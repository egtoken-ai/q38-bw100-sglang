#!/usr/bin/env python3
"""Compare cached explicit hipBLASLt algorithms with current public backends."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library', default='/tmp/libq38_hipblaslt_cached.so')
    p.add_argument('--output', required=True)
    p.add_argument('--m', type=int, default=8192)
    p.add_argument('--n', type=int, default=17408)
    p.add_argument('--k', type=int, default=5120)
    p.add_argument('--index', type=int, default=-1, help='-1 uses first heuristic; otherwise exact library solution index')
    p.add_argument('--workspace-mib', type=int, default=128, choices=[0, 128])
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if args.dry_run:
        print(json.dumps(vars(args)))
        return
    import torch
    from lightop import gemm_ops
    from hipblaslt_cached import CachedHipblasLt

    torch.set_num_threads(4)
    torch.manual_seed(20261015)
    m, n, k = args.m, args.n, args.k
    a = torch.randint(-127, 128, (m, k), dtype=torch.int8, device='cuda')
    w = torch.randint(-127, 128, (n, k), dtype=torch.int8, device='cuda')
    sa = torch.rand(m, 1, device='cuda') * .01 + .0001
    sw = torch.rand(n, 1, device='cuda') * .01 + .0001
    backend = CachedHipblasLt(args.library)
    asm = lambda: gemm_ops.triton_scaled_mm(a, w.t(), sa, sw, torch.bfloat16)
    legacy = lambda: gemm_ops.hipblaslt_w8a8_gemm(a, w, sa, sw, m, n, k, 'NT', torch.bfloat16)[1]
    tuned = lambda: backend.run(a, w, sa, sw, index=args.index, workspace_mib=args.workspace_mib)
    reference = asm()
    independent = ((a[:8].cpu().double() @ w[:16].cpu().double().t())
                   * sa[:8].cpu().double() * sw[:16].cpu().double().t()).to(torch.bfloat16)
    torch.testing.assert_close(reference[:8, :16].cpu(), independent, rtol=.02, atol=.01)
    result = dict(scope='Microbenchmark only; not model TTFT or accuracy acceptance',
                  config=vars(args), library_sha256=hashlib.sha256(Path(args.library).read_bytes()).hexdigest(), rows=[])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for name, fn in [('lightop-asm', asm), ('legacy-hipblaslt', legacy), ('cached-hipblaslt', tuned)]:
        row = dict(backend=name)
        try:
            actual = fn()
            torch.cuda.synchronize()
            delta = actual.float() - reference.float()
            row.update(max_abs_error=delta.abs().max().item(),
                       relative_l2_error=(delta.norm() / reference.float().norm()).item(),
                       exact_fraction=(actual == reference).float().mean().item())
            torch.testing.assert_close(actual, reference, rtol=.02, atol=.01)
            torch.testing.assert_close(actual[:8, :16].cpu(), independent, rtol=.02, atol=.01)
            del actual, delta
            for _ in range(3):fn()
            torch.cuda.synchronize()
            times=[]
            for _ in range(7):
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(3):fn()
                end.record();end.synchronize()
                times.append(start.elapsed_time(end)/3)
            row.update(status='PASS',median_ms=statistics.median(times),samples_ms=times)
            if name=='cached-hipblaslt':row.update(backend.last_selection)
        except Exception as exc:
            row.update(status='FAIL',error=f'{type(exc).__name__}: {exc}')
            result['rows'].append(row)
            output.write_text(json.dumps(result,indent=2)+'\n')
            raise
        result['rows'].append(row)
        output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(row),flush=True)
    backend.clear()


if __name__=='__main__':
    main()
