"""Screen INT8 GEMM backends at the Qwen27B TP2 projection shapes."""
import argparse
import gc
import json
from pathlib import Path
import statistics

import torch
from lightop import gemm_ops


def timed(fn):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(5):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(20261015)
    results = []
    for m in [4096, 8192, 16384]:
        for n, k in [(17408, 5120), (5120, 8704), (8192, 5120), (5120, 3072)]:
            a = torch.randint(-32, 32, (m, k), device='cuda', dtype=torch.int8)
            weight = torch.randint(-32, 32, (n, k), device='cuda', dtype=torch.int8)
            b = weight.t()
            sa = torch.rand(m, 1, device='cuda', dtype=torch.float32) * .01
            sb = torch.rand(n, 1, device='cuda', dtype=torch.float32) * .01
            baseline = lambda: gemm_ops.triton_scaled_mm(a, b, sa, sb, torch.bfloat16)
            reference = baseline()
            independent = ((a[:32].float() @ weight[:64].float().t())
                           * sa[:32] * sb[:64].t()).to(torch.bfloat16)
            torch.testing.assert_close(reference[:32, :64], independent, atol=.01, rtol=.02)
            choices = {
                'baseline-lightop': baseline,
                'hipblaslt': lambda: gemm_ops.hipblaslt_w8a8_gemm(
                    a, weight, sa, sb, m, n, k, 'NT', torch.bfloat16)[1],
                # Match the strategy-2 column-major repacking used by SGLang.
                'rocblas': lambda: gemm_ops.rocblas_scaled_mm_nn(
                    a, packed_rocblas, sa, sb, torch.bfloat16),
                'triton-only': lambda: gemm_ops.triton_scaled_mm.__globals__['matmul_int8'](
                    a, sa, b, sb, torch.bfloat16, None),
            }
            packed_rocblas = weight.t().contiguous().reshape(n, k).t()
            for name, fn in choices.items():
                row = dict(m=m, n=n, k=k, candidate=name)
                try:
                    actual = fn()
                    row['max_abs_error'] = (actual.float() - reference.float()).abs().max().item()
                    torch.testing.assert_close(actual, reference, atol=.01, rtol=.02)
                    row.update(median_ms=timed(fn), status='PASS')
                except Exception as exc:
                    row.update(status='FAIL', error=f'{type(exc).__name__}: {exc}'[:600])
                results.append(row)
                print(json.dumps(row), flush=True)
                Path(args.output).write_text(json.dumps(results, indent=2) + '\n')
            del a, b, weight, sa, sb, reference
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
