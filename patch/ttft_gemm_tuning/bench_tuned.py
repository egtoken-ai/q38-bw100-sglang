#!/usr/bin/env python3
"""Screen vendor ASM INT8 mode IDs; never edits the serving process or cache."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import os

SHAPES = [(17408, 5120), (5120, 8704), (8192, 5120), (5120, 3072)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--m', type=int, nargs='+', default=[4096, 8192])
    p.add_argument('--shapes', type=int, nargs='+', choices=range(4), default=list(range(4)))
    p.add_argument('--modes', type=int, nargs='+', choices=range(11), default=list(range(11)))
    p.add_argument('--repeats', type=int, default=7)
    p.add_argument('--calls', type=int, default=3)
    p.add_argument('--timeout', type=float, default=120, help='Per-mode isolated worker deadline')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if any(m <= 16 or m % 128 for m in args.m) or args.repeats < 3 or args.calls < 1:
        p.error('Use prefill M divisible by 128 and at least 3 timing samples')
    if args.dry_run:
        print(json.dumps(dict(shapes=[(m, *SHAPES[i]) for m in args.m for i in args.shapes],
                              modes=args.modes, dtype='INT8 inputs / BF16 output',
                              baseline='public triton_scaled_mm', gpu_executed=False)))
        return

    if not args.worker:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        summary = dict(scope='Isolated ASM mode screening, not model acceptance', workers=[], rows=[])
        for mode in args.modes:
            child_output = output.with_name(output.stem + f'-mode{mode}' + output.suffix)
            log = child_output.with_suffix(child_output.suffix + '.log')
            if child_output.exists() or log.exists():
                raise FileExistsError(f'Use a fresh --output path; worker artifact exists: {child_output}')
            command = [sys.executable, str(Path(__file__).resolve()), '--worker',
                       '--output', str(child_output), '--modes', str(mode),
                       '--m', *map(str, args.m), '--shapes', *map(str, args.shapes),
                       '--repeats', str(args.repeats), '--calls', str(args.calls)]
            try:
                with log.open('w') as stream:
                    process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                             timeout=args.timeout, check=False)
                worker = dict(mode=mode, exit_code=process.returncode, log=str(log))
            except subprocess.TimeoutExpired:
                worker = dict(mode=mode, status='TIMEOUT', log=str(log))
            if child_output.exists():
                child = json.loads(child_output.read_text())
                summary['rows'].extend(child['rows'])
                worker['metadata'] = {k: v for k, v in child.items() if k != 'rows'}
            summary['workers'].append(worker)
            output.write_text(json.dumps(summary, indent=2) + '\n')
            print(json.dumps(worker), flush=True)
        return

    import torch
    from lightop import gemm_ops, gemmopt
    from lightop.config_smooth_gemm import gemm_w8a8_smooth_config

    actual_cus = torch.cuda.get_device_properties(0).multi_processor_count
    selected_cus = int(os.environ.get('LIGHTOP_GPU_CUS', '80'))
    if actual_cus != selected_cus:
        raise RuntimeError(f'CU mismatch: device={actual_cus}, config={selected_cus}')
    allowed_modes = set(gemm_w8a8_smooth_config.get_gemm_w8a8_smooth_asm_dict().values())
    if not set(args.modes) <= allowed_modes:
        raise RuntimeError(f'Modes absent in current CU config: {set(args.modes) - allowed_modes}')

    torch.set_num_threads(4)
    torch.manual_seed(20261015)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(scope='Microbenchmark only; no model TTFT claim',
                  source_sha256=hashlib.sha256(Path(gemmopt.__file__).read_bytes()).hexdigest(),
                  device=torch.cuda.get_device_name(), actual_cus=actual_cus, selected_cus=selected_cus,
                  allowed_modes=sorted(allowed_modes), calls_per_sample=args.calls, rows=[])

    def save(row):
        result['rows'].append(row)
        output.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(row), flush=True)

    def timed(fn):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        values = []
        for _ in range(args.repeats):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.calls):
                fn()
            end.record()
            end.synchronize()
            values.append(start.elapsed_time(end) / args.calls)
        return statistics.median(values), values

    for m in args.m:
        for shape_id in args.shapes:
            n, k = SHAPES[shape_id]
            a = torch.randint(-127, 128, (m, k), device='cuda', dtype=torch.int8)
            weight = torch.randint(-127, 128, (n, k), device='cuda', dtype=torch.int8)
            b = weight.t()
            sa = torch.rand(m, 1, device='cuda') * .01 + .0001
            sb = torch.rand(n, 1, device='cuda') * .01 + .0001
            default_config = gemmopt.w8a8_smooth_asm_config(m, n, k)
            baseline = lambda: gemm_ops.triton_scaled_mm(a, b, sa, sb, torch.bfloat16)
            status, asm_reference = gemmopt.gemm_w8a8_smooth(a, b, sa, sb, None, torch.bfloat16)
            if not status:
                raise RuntimeError('Public baseline is not using the vendor ASM path')
            reference = baseline()
            torch.testing.assert_close(reference, asm_reference, rtol=0, atol=0)
            del asm_reference
            # Independent CPU FP64 check of a small output slice. Outside all timing.
            exact = ((a[:8].cpu().double() @ weight[:16].cpu().double().t())
                     * sa[:8].cpu().double() * sb[:16].cpu().double().t()).to(torch.bfloat16)
            torch.testing.assert_close(reference[:8, :16].cpu(), exact, rtol=.02, atol=.01)
            base_ms, samples = timed(baseline)
            save(dict(m=m, n=n, k=k, mode='public-baseline', default_config=default_config,
                      status='PASS', median_ms=base_ms, samples_ms=samples))
            for mode in args.modes:
                # Same dimensions/layout/scales; select an ID present in vendor config tables.
                fn = lambda: gemmopt.op.gemm_w8a8_smooth_asm(
                    a, b, sa, sb, None, torch.bfloat16, m, n, k, True, mode)
                row = dict(m=m, n=n, k=k, mode=mode)
                try:
                    actual = fn()
                    torch.cuda.synchronize()
                    diff = actual.float() - reference.float()
                    row['max_abs_error'] = diff.abs().max().item()
                    row['relative_l2_error'] = (diff.norm() / reference.float().norm()).item()
                    row['exact_fraction'] = (actual == reference).float().mean().item()
                    torch.testing.assert_close(actual, reference, rtol=.02, atol=.01)
                    torch.testing.assert_close(actual[:8, :16].cpu(), exact, rtol=.02, atol=.01)
                    del actual, diff
                    ms, samples = timed(fn)
                    row.update(status='PASS', median_ms=ms, samples_ms=samples,
                               speedup_vs_public=base_ms / ms)
                except Exception as exc:
                    row.update(status='FAIL', error=f'{type(exc).__name__}: {exc}'[:1000])
                    save(row)
                    if any(word in str(exc).lower() for word in ['illegal memory', 'device-side', 'memory access fault']):
                        raise
                    continue
                save(row)
            del a, weight, b, sa, sb, reference
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
