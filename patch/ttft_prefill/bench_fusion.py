"""Screen vendor RMSNorm/SiLU plus INT8 quantization fusion for Qwen Prefill."""
import argparse
import json
from pathlib import Path
import statistics

import torch
from lightop.norm import gemma_fused_add_rmsnorm, lm_faster_rmsquant
from lightop.activation import fuse_silu_and_mul, lm_fuse_silu_mul_quant
from lightop.quant import per_token_quant_int8


def deployed_silu_and_mul(x):
    """The exact HCU SiluAndMul.forward_cuda operation, without server state."""
    out = torch.empty((*x.shape[:-1], x.shape[-1] // 2), dtype=x.dtype, device=x.device)
    fuse_silu_and_mul(x, out)
    return out


def measure(fn, cuda_graph=False):
    # Time many executions per event pair. A captured replay additionally
    # isolates kernel/device cost from Python/dispatcher launch latency.
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    if cuda_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_output = fn()
        fn = graph.replay
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
    samples = []
    for _ in range(5):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(20):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / 20)
    return statistics.median(samples)


def measure_pair(baseline, candidate):
    return dict(
        baseline_ms=measure(baseline, cuda_graph=True),
        candidate_ms=measure(candidate, cuda_graph=True),
        baseline_eager_ms=measure(baseline),
        candidate_eager_ms=measure(candidate),
        timing_method='median of 5 event pairs, each with 20 executions; '
                      'primary values use CUDA graph replay',
    )


def compare_quant(actual, expected):
    aq, asc = actual
    eq, esc = expected
    a = aq.float() * asc.reshape(-1, 1)
    e = eq.float() * esc.reshape(-1, 1)
    rel = ((a-e).norm() / e.norm()).item()
    if not torch.isfinite(a).all() or rel > .02:
        raise AssertionError(f'Quantized relative L2 error {rel}')
    return dict(relative_l2_error=rel,
                max_quant_step_diff=(aq.int()-eq.int()).abs().max().item(),
                quant_mismatch_fraction=(aq!=eq).float().mean().item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    args = p.parse_args()
    torch.manual_seed(20261015)
    torch.set_num_threads(4)
    rows = []
    activation = deployed_silu_and_mul
    for m in [128, 4096, 8192]:
        x = torch.randn(m, 5120, device='cuda', dtype=torch.bfloat16)
        residual = torch.randn_like(x)
        weight = torch.randn(5120, device='cuda', dtype=torch.bfloat16) * .1
        gamma = weight + 1

        def norm_base():
            out, res = gemma_fused_add_rmsnorm(x.clone(), residual.clone(), weight, 1e-6)
            return per_token_quant_int8(out), out, res

        def norm_fused():
            out, res = x.clone(), residual.clone()
            quant = lm_faster_rmsquant(out, gamma, 1e-6, torch.int8, res, True)
            return quant, out, res

        row = dict(tokens=m, stage='gemma-rmsnorm-quant')
        try:
            ref, out_ref, res_ref = norm_base()
            q, out, res = norm_fused()
            row.update(compare_quant(q, ref))
            row['norm_max_abs_error'] = (out.float()-out_ref.float()).abs().max().item()
            row['residual_max_abs_error'] = (res.float()-res_ref.float()).abs().max().item()
            torch.testing.assert_close(out, out_ref, atol=.025, rtol=.015)
            torch.testing.assert_close(res, res_ref, atol=0, rtol=0)
            row.update(status='PASS', **measure_pair(norm_base, norm_fused))
        except Exception as exc:
            row.update(status='FAIL', error=f'{type(exc).__name__}: {exc}'[:700])
        rows.append(row);print(json.dumps(row), flush=True)
        gate = torch.randn(m, 17408, device='cuda', dtype=torch.bfloat16)
        base = lambda: per_token_quant_int8(activation(gate))
        fused = lambda: lm_fuse_silu_mul_quant(gate)
        row = dict(tokens=m, stage='silu-mul-quant')
        try:
            row.update(compare_quant(fused(), base()))
            row.update(status='PASS', **measure_pair(base, fused))
        except Exception as exc:
            row.update(status='FAIL', error=f'{type(exc).__name__}: {exc}'[:700])
        rows.append(row);print(json.dumps(row), flush=True)
        Path(args.output).write_text(json.dumps(rows, indent=2)+'\n')


if __name__ == '__main__':
    main()
