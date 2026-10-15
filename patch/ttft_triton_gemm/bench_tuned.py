#!/usr/bin/env python3
"""Bounded eight-config INT8 Triton screen; only the parent schedules GPU work."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

# (name, BM, BN, BK, warps, stages, grouped-M). SPLIT_K=1 preserves integer sum.
SPECS = [
    ('vendor-80cu-up-candidate', 256, 128, 64, 4, 1, 4),
    ('previous-default', 256, 256, 64, 8, 0, 8),
    ('vendor-80cu-down-candidate', 128, 256, 64, 4, 2, 8),
    ('128x128-k128', 128, 128, 128, 4, 2, 4),
    ('256x128-k128', 256, 128, 128, 8, 1, 4),
    ('128x256-k128', 128, 256, 128, 8, 1, 8),
    ('128x128-k256', 128, 128, 256, 8, 1, 4),
    ('128x128-k64-s3', 128, 128, 64, 4, 3, 8),
]


def configuration(index):
    name, bm, bn, bk, warps, stages, group = SPECS[index]
    return name, dict(BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk,
                      GROUP_SIZE_M=group, SPLIT_K=1, num_warps=warps, num_stages=stages)


def worker(args):
    import torch
    from lightop import gemm_ops

    torch.set_num_threads(4)
    torch.manual_seed(20261015)
    fn = gemm_ops.triton_scaled_mm.__globals__['matmul_int8']
    kernel = fn.__globals__['matmul_kernel']
    name, config = configuration(args.worker)
    row = dict(name=name, config=config, m=args.m, n=args.n, k=args.k,
               source_file=fn.__code__.co_filename,
               source_sha256=hashlib.sha256(Path(fn.__code__.co_filename).read_bytes()).hexdigest())
    outpath = Path(args.output)

    def save():
        outpath.write_text(json.dumps(row, indent=2) + '\n')
        print(json.dumps(row), flush=True)

    try:
        props = torch.cuda.get_device_properties(0)
        arch = getattr(props, 'gcnArchName', '').split(':')[0]
        if arch != 'gfx936' or props.multi_processor_count != 64:
            raise RuntimeError(f'Expected gfx936/64cu, got {arch}/{props.multi_processor_count}')
        row.update(arch=arch, cu=props.multi_processor_count)
        m, n, k = args.m, args.n, args.k
        a = torch.randint(-127, 128, (m, k), dtype=torch.int8, device='cuda')
        w = torch.randint(-127, 128, (n, k), dtype=torch.int8, device='cuda')
        b = w.t()
        sa = torch.rand(m, 1, device='cuda') * .01 + .0001
        sb = torch.rand(n, 1, device='cuda') * .01 + .0001
        c = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
        grid = ((m + config['BLOCK_SIZE_M'] - 1) // config['BLOCK_SIZE_M']
                * ((n + config['BLOCK_SIZE_N'] - 1) // config['BLOCK_SIZE_N']), 1)
        kernel_args = (a, sa, b, sb, c, m, n, k,
                       a.stride(0), a.stride(1), sa.stride(0),
                       b.stride(0), b.stride(1), sb.stride(0), c.stride(0), c.stride(1))
        # warmup compiles without launching. Reject LDS use before any kernel launch.
        compiled = kernel.warmup(*kernel_args, grid=grid, **config)
        shared = int(compiled.metadata.shared)
        row.update(compiled_shared_bytes=shared, num_regs=getattr(compiled, 'n_regs', None))
        if shared > 64 * 1024:
            row.update(status='SKIP_SHARED_MEMORY', limit_bytes=64 * 1024)
            save()
            return

        asm = lambda: gemm_ops.triton_scaled_mm(a, b, sa, sb, torch.bfloat16)
        api = lambda: fn(a, sa, b, sb, torch.bfloat16, config)
        def raw():
            kernel[grid](*kernel_args, **config)
            return c

        reference = asm()
        independent = ((a[:8].cpu().double() @ w[:16].cpu().double().t())
                       * sa[:8].cpu().double() * sb[:16].cpu().double().t()).to(torch.bfloat16)
        torch.testing.assert_close(reference[:8, :16].cpu(), independent, rtol=.02, atol=.01)
        actual = api()
        torch.cuda.synchronize()
        delta = actual.float() - reference.float()
        row.update(max_abs_error=delta.abs().max().item(),
                   relative_l2_error=(delta.norm() / reference.float().norm()).item(),
                   exact_fraction=(actual == reference).float().mean().item())
        torch.testing.assert_close(actual, reference, rtol=.02, atol=.01)
        torch.testing.assert_close(actual[:8, :16].cpu(), independent, rtol=.02, atol=.01)
        raw_output = raw()
        torch.testing.assert_close(raw_output, actual, rtol=0, atol=0)
        del actual, delta, raw_output

        def timed(call):
            for _ in range(3):call()
            torch.cuda.synchronize()
            samples = []
            for _ in range(5):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(3):call()
                end.record();end.synchronize()
                samples.append(start.elapsed_time(end) / 3)
            return dict(median_ms=statistics.median(samples), samples_ms=samples)

        row['asm'] = timed(asm)
        row['triton_api'] = timed(api)  # Includes original torch.zeros allocation/init.
        row['triton_kernel_only'] = timed(raw)  # Preallocated output, full overwrite.
        row['api_speedup_vs_asm'] = row['asm']['median_ms'] / row['triton_api']['median_ms']
        row['kernel_speedup_vs_asm'] = row['asm']['median_ms'] / row['triton_kernel_only']['median_ms']
        row['status'] = 'PASS'
    except Exception as exc:
        row.update(status='FAIL', error=f'{type(exc).__name__}: {exc}'[:3000])
    save()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--m', type=int, default=8192)
    p.add_argument('--n', type=int, default=17408)
    p.add_argument('--k', type=int, default=5120)
    p.add_argument('--configs', type=int, nargs='+', choices=range(8), default=list(range(8)))
    p.add_argument('--worker', type=int, choices=range(8), help=argparse.SUPPRESS)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if min(args.m, args.n, args.k) <= 0 or any(x % 128 for x in (args.m, args.n, args.k)):
        p.error('This bounded screen requires positive M/N/K divisible by 128')
    if len(args.configs) > 8 or len(set(args.configs)) != len(args.configs):
        p.error('At most eight distinct configurations are allowed')
    if args.dry_run:
        print(json.dumps(dict(shape=[args.m,args.n,args.k],
                              configurations=[configuration(i) for i in args.configs],
                              global_deadline_seconds=450, worker_timeout_seconds=50,
                              gpu_executed=False)))
        return
    if args.worker is not None:
        worker(args)
        return
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError('Use a fresh output path')
    deadline = time.monotonic() + 450
    started = time.monotonic()
    result = dict(scope='Eight-config micro screening, not model acceptance', shape=[args.m,args.n,args.k], workers=[])
    for index in args.configs:
        remaining = deadline - time.monotonic()
        if remaining <= 2:
            result['deadline_reached'] = True
            break
        child_json = output.with_name(output.stem + f'-config{index}' + output.suffix)
        child_log = child_json.with_suffix(child_json.suffix + '.log')
        if child_json.exists() or child_log.exists():
            raise FileExistsError(f'Existing worker artifacts: {child_json}')
        command = [sys.executable,str(Path(__file__).resolve()),'--worker',str(index),
                   '--output',str(child_json),'--m',str(args.m),'--n',str(args.n),'--k',str(args.k)]
        with child_log.open('w') as log:
            process = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                process.wait(timeout=min(50, max(1,remaining-2)))
                state = dict(config_id=index, exit_code=process.returncode)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL)
                try:process.wait(timeout=1)
                except subprocess.TimeoutExpired:pass
                state = dict(config_id=index,status='TIMEOUT')
        state['log'] = str(child_log)
        if child_json.exists():
            state['result'] = json.loads(child_json.read_text())
        result['workers'].append(state)
        result['elapsed_seconds'] = time.monotonic()-started
        output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(state),flush=True)
    result['elapsed_seconds'] = time.monotonic()-started
    output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
