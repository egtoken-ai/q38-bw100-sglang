"""BF16 correctness and timing for ratio-3 GDN projection views on BW100.
Run with HIP_VISIBLE_DEVICES=4 (root controls GPU scheduling).
No model weights, no server; this is diagnostic, never an acceptance score.
"""
import argparse
import json
from pathlib import Path

import torch
import triton
from sglang.srt.layers.attention.fla.layernorm_gated import rms_norm_gated
from sglang.srt.layers.attention.fla.fused_gdn_gating import fused_gdn_gating
from sglang.srt.layers.attention.mamba.causal_conv1d_triton import causal_conv1d_fn


def views(qkvz, ba):
    return qkvz[:, :5120], qkvz[:, 5120:].view(-1, 24, 128), ba[:, :24], ba[:, 24:]


def legacy(qkvz, ba):
    q, k, v, z = qkvz.split([1024, 1024, 3072, 3072], dim=-1)
    b, a = ba.split([24, 24], dim=-1)
    return torch.cat((q, k, v), -1), z.reshape(-1, 128), b.contiguous(), a.contiguous()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.manual_seed(3713)
    results = []
    for lengths in ([63, 129], [4096], [4096, 4096], [8192]):
        tokens = sum(lengths)
        qkvz = torch.randn(tokens, 8192, dtype=torch.bfloat16, device='cuda')
        ba = torch.randn(tokens, 48, dtype=torch.bfloat16, device='cuda')
        x = torch.randn(tokens * 24, 128, dtype=torch.bfloat16, device='cuda')
        weight = torch.randn(128, dtype=torch.bfloat16, device='cuda')
        alo = torch.randn(24, device='cuda')
        dt = torch.randn(24, dtype=torch.bfloat16, device='cuda')
        old = legacy(qkvz, ba)
        new = views(qkvz, ba)
        for i in (0, 2, 3):
            torch.testing.assert_close(old[i], new[i], rtol=0, atol=0)
        torch.testing.assert_close(old[1], new[1].reshape(-1, 128), rtol=0, atol=0)
        old_g = fused_gdn_gating(alo, old[3], old[2], dt)
        new_g = fused_gdn_gating(alo, new[3], new[2], dt)
        for left, right in zip(old_g, new_g):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

        # Verify both norm ordering modes, retaining BF16 intermediate semantics.
        errors = []
        for norm_before in (True, False):
            kw = dict(x=x, weight=weight, bias=None, eps=1e-6,
                      is_rms_norm=True, norm_before_gate=norm_before)
            baseline = rms_norm_gated(z=old[1], **kw)
            candidate = rms_norm_gated(z=new[1], **kw)
            torch.testing.assert_close(baseline, candidate, rtol=0.015625, atol=0.015625)
            errors.append(float((baseline.float() - candidate.float()).abs().max()))

        # Production conv consumes transpose views; verify outputs and final state.
        batch = len(lengths)
        initial = torch.randn(batch, 5120, 3, dtype=torch.bfloat16, device='cuda')
        conv_weight = torch.randn(5120, 4, dtype=torch.bfloat16, device='cuda')
        starts = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0).tolist()),
                              dtype=torch.int32, device='cuda')
        common = dict(weight=conv_weight, bias=None, query_start_loc=starts,
                      seq_lens_cpu=list(lengths),
                      cache_indices=torch.arange(batch, dtype=torch.int32, device='cuda'),
                      has_initial_state=torch.ones(batch, dtype=torch.bool, device='cuda'),
                      activation='silu', validate_data=True)
        state_old = initial.clone()
        state_new = initial.clone()
        conv_old = causal_conv1d_fn(old[0].T, conv_states=state_old, **common)
        conv_new = causal_conv1d_fn(new[0].T, conv_states=state_new, **common)
        torch.testing.assert_close(conv_old, conv_new, rtol=0, atol=0)
        torch.testing.assert_close(state_old, state_new, rtol=0, atol=0)

        def run(split):
            _, z, b, a = split(qkvz, ba)
            fused_gdn_gating(alo, a, b, dt)
            return rms_norm_gated(x=x, weight=weight, bias=None, z=z,
                                  eps=1e-6, is_rms_norm=True)
        base_ms = triton.testing.do_bench(lambda: run(legacy), warmup=100, rep=300)
        view_ms = triton.testing.do_bench(lambda: run(views), warmup=100, rep=300)

        # empty_like can densify a view with holes. Record the actual device
        # output layout rather than infer it from input contiguity alone.
        layout = dict(
            baseline_conv_input_stride=list(old[0].T.stride()),
            candidate_conv_input_stride=list(new[0].T.stride()),
            baseline_conv_output_stride=list(conv_old.stride()),
            candidate_conv_output_stride=list(conv_new.stride()),
            baseline_backend_qkv_stride=list(conv_old.T.stride()),
            candidate_backend_qkv_stride=list(conv_new.T.stride()),
            baseline_backend_qkv_contiguous=conv_old.T.is_contiguous(),
            candidate_backend_qkv_contiguous=conv_new.T.is_contiguous(),
            baseline_q_stride=list(conv_old.T[:, :1024].view(1, tokens, 8, 128).stride()),
            candidate_q_stride=list(conv_new.T[:, :1024].view(1, tokens, 8, 128).stride()),
        )
        timing_common = dict(common, validate_data=False)

        def run_conv(projected, state):
            # Same state reset/launch in both arms; includes reset cost.
            state.copy_(initial)
            return causal_conv1d_fn(projected.T, conv_states=state, **timing_common)

        def run_segment(split, state):
            projected, z, b, a = split(qkvz, ba)
            conv_out = run_conv(projected, state)
            g, beta = fused_gdn_gating(alo, a, b, dt)
            # x stands in for the recurrence output shared by both arms. This
            # segment intentionally excludes recurrence and its QKV preparation.
            norm_out = rms_norm_gated(x=x, weight=weight, bias=None, z=z,
                                     eps=1e-6, is_rms_norm=True)
            return conv_out, g, beta, norm_out

        old_conv_ms = triton.testing.do_bench(
            lambda: run_conv(old[0], state_old), warmup=100, rep=300)
        new_conv_ms = triton.testing.do_bench(
            lambda: run_conv(new[0], state_new), warmup=100, rep=300)
        old_segment_ms = triton.testing.do_bench(
            lambda: run_segment(legacy, state_old), warmup=100, rep=300)
        new_segment_ms = triton.testing.do_bench(
            lambda: run_segment(views, state_new), warmup=100, rep=300)
        item = dict(lengths=lengths, dtype='bfloat16', all_checks_pass=True,
                    layout=layout,
                    baseline_conv_with_state_reset_ms=old_conv_ms,
                    candidate_conv_with_state_reset_ms=new_conv_ms,
                    baseline_split_conv_gate_norm_ms=old_segment_ms,
                    candidate_split_conv_gate_norm_ms=new_segment_ms,
                    segment_excludes='recurrence and post-conv QKV preparation',
                    norm_max_abs=errors, baseline_ms=base_ms, views_ms=view_ms,
                    baseline_projection_stride=list(old[0].stride()),
                    candidate_projection_stride=list(new[0].stride()))
        print(json.dumps(item), flush=True)
        results.append(item)
    Path(args.output).write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
