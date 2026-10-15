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
        item = dict(lengths=lengths, dtype='bfloat16', all_checks_pass=True,
                    norm_max_abs=errors, baseline_ms=base_ms, views_ms=view_ms,
                    baseline_projection_stride=list(old[0].stride()),
                    candidate_projection_stride=list(new[0].stride()))
        print(json.dumps(item), flush=True)
        results.append(item)
    Path(args.output).write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
