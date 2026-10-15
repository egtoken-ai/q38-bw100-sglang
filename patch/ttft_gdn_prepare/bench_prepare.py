"""BF16 postconv preparation parity, state recurrence check, and timing.
Root schedules this serially on GPU4. Diagnostic only, not TTFT acceptance.
"""
import argparse
import json
from pathlib import Path

import torch
import triton
from sglang.srt.layers.attention.fla.fused_gdn_gating import fused_gdn_gating
from sglang.srt.layers.attention.fla.gdn_prefill_prepare import fused_gdn_prefill_prepare
from sglang.srt.layers.attention.fla.l2norm import l2norm_fwd
from sglang.srt.layers.attention.fla.chunk import chunk_gated_delta_rule


def old_prepare(x, a_log, a, b, dt):
    tokens = x.shape[0]
    q, k, v = x.split([1024, 1024, 3072], dim=-1)
    q = q.view(1, tokens, 8, 128).contiguous()
    k = k.view(1, tokens, 8, 128).contiguous()
    v = v.view(1, tokens, 24, 128).contiguous()
    q = l2norm_fwd(q)
    k = l2norm_fwd(k)
    g, beta = fused_gdn_gating(a_log, a, b, dt)
    return q, k, v, g, beta


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    args = p.parse_args()
    torch.manual_seed(38806)
    rows = []
    for tokens in (192, 4096, 8192):
        for layout in ('token_major', 'feature_major'):
            if layout == 'token_major':
                x = torch.randn(tokens, 5120, dtype=torch.bfloat16, device='cuda')
            else:
                x = torch.randn(5120, tokens, dtype=torch.bfloat16, device='cuda').T
            ba = torch.randn(tokens, 48, dtype=torch.bfloat16, device='cuda')
            b, a = ba[:, :24], ba[:, 24:]
            a_log = torch.randn(24, dtype=torch.float32, device='cuda')
            dt = torch.randn(24, dtype=torch.bfloat16, device='cuda')
            reference = old_prepare(x, a_log, a, b, dt)
            qkv_diff = []
            launches = []
            for warps in (4, 8, 16):
                result = fused_gdn_prefill_prepare(x, a_log, a, b, dt, 8, 24, 128, 128,
                                                  num_warps=warps)
                assert all(t.dtype == torch.bfloat16 for t in result[:3])
                assert all(t.dtype == torch.float32 for t in result[3:])
                for idx in (0, 1):
                    torch.testing.assert_close(reference[idx], result[idx], rtol=0.015625, atol=0.001)
                # V, g, and beta must retain the vendor formula and rounding exactly.
                for idx in (2, 3, 4):
                    torch.testing.assert_close(reference[idx], result[idx], rtol=0, atol=0)
                qkv_diff.append(max(float((reference[i].float() - result[i].float()).abs().max())
                                    for i in (0, 1)))
                ms = triton.testing.do_bench(
                    lambda: fused_gdn_prefill_prepare(x, a_log, a, b, dt, 8, 24, 128, 128,
                                                      num_warps=warps), warmup=100, rep=200)
                launches.append(dict(num_warps=warps, ms=ms))
            base_ms = triton.testing.do_bench(lambda: old_prepare(x, a_log, a, b, dt),
                                              warmup=100, rep=200)
            # Compare chunk outputs and final in-place state for the short tail.
            # Both consume already-normalized BF16 Q/K so no double normalization.
            recurrence = None
            if tokens == 192:
                state0 = torch.randn(2, 24, 128, 128, dtype=torch.float32, device='cuda') * 0.01
                state1, state2 = state0.clone(), state0.clone()
                starts = torch.tensor([0, 63, 192], dtype=torch.int32, device='cuda')
                indices = torch.tensor([0, 1], dtype=torch.int32, device='cuda')
                def recur(prepared, state):
                    q, k, v, g, beta = prepared
                    return chunk_gated_delta_rule(
                        q=q, k=k, v=v, g=g, beta=beta, initial_state=state,
                        initial_state_indices=indices, cu_seqlens=starts,
                        head_first=False, use_qk_l2norm_in_kernel=False)
                old = recur(reference, state1)
                new = recur(result, state2)
                torch.testing.assert_close(old[0], new[0], rtol=0.02, atol=0.02)
                torch.testing.assert_close(state1, state2, rtol=0.02, atol=0.02)
                recurrence = dict(output_max_abs=float((old[0].float()-new[0].float()).abs().max()),
                                  state_max_abs=float((state1-state2).abs().max()))
            row = dict(tokens=tokens, layout=layout, stride=list(x.stride()),
                       all_checks_pass=True, max_qk_abs=qkv_diff, baseline_ms=base_ms,
                       candidates=launches, recurrence=recurrence)
            rows.append(row)
            print(json.dumps(row), flush=True)
    Path(args.output).write_text(json.dumps(rows, indent=2) + '\n')


if __name__ == '__main__':
    main()
