"""TP2 MLP-only overlap: AR(output A) overlaps MLP(B), no input all-reduce.

Run under torchrun with exactly two visible GPUs. All ranks issue collectives in
identical order. Fixed iteration counts avoid divergent rank collective counts.
"""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import statistics
import time

import torch
import torch.distributed as dist
from lightop import gemm_ops
from lightop.activation import fuse_silu_and_mul
from lightop.quant import per_token_quant_int8


def error_metrics(actual, reference, exact=False):
    delta = actual.float() - reference.float()
    metrics = dict(max_abs=float(delta.abs().max()),
                   relative_l2=float(delta.norm()/reference.float().norm().clamp_min(1e-12)),
                   mismatch_fraction=float((actual != reference).float().mean()))
    torch.testing.assert_close(actual, reference,
                               atol=0 if exact else .015625,
                               rtol=0 if exact else .015625)
    if metrics['relative_l2'] > .002:
        raise AssertionError(f'Relative L2 exceeds 0.2%: {metrics}')
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--micro-tokens', type=int, nargs='+', default=[2048,4096,8192])
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--trace-dir')
    args = parser.parse_args()
    rank = int(os.environ['LOCAL_RANK'])
    if rank == 0:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if int(os.environ['WORLD_SIZE']) != 2:
        raise RuntimeError('Exactly TP2 is required')
    if os.environ.get('TORCH_NCCL_BLOCKING_WAIT', '0') != '0':
        raise RuntimeError('Blocking NCCL wait would invalidate this overlap experiment')
    torch.cuda.set_device(rank)
    torch.set_num_threads(2)
    dist.init_process_group('gloo', timeout=timedelta(seconds=180))
    nccl = dist.new_group(backend='nccl', timeout=timedelta(seconds=180))
    torch.manual_seed(20261015 + rank)
    # Actual dense 27B TP2 local MLP projection shapes.
    gate_weight = torch.randint(-16,16,(17408,5120),device='cuda',dtype=torch.int8).T
    down_weight = torch.randint(-16,16,(5120,8704),device='cuda',dtype=torch.int8).T
    gate_scale = torch.full((17408,1), .001, device='cuda', dtype=torch.float32)
    down_scale = torch.full((5120,1), .001, device='cuda', dtype=torch.float32)
    sa, sb = torch.cuda.Stream(), torch.cuda.Stream()
    default = torch.cuda.current_stream()
    after_a_compute = torch.cuda.Event()

    def mlp(x):
        qx, sx = per_token_quant_int8(x)
        gate = gemm_ops.triton_scaled_mm(qx, gate_weight, sx, gate_scale, torch.bfloat16)
        activated = torch.empty((x.shape[0],8704),device=x.device,dtype=torch.bfloat16)
        fuse_silu_and_mul(gate, activated)
        qa, scales = per_token_quant_int8(activated)
        return gemm_ops.triton_scaled_mm(qa, down_weight, scales, down_scale, torch.bfloat16)

    def single_batch(inp):
        x = inp.clone()
        y = mlp(x)
        dist.all_reduce(y, group=nccl)
        return y

    def serial_microbatches(inp):
        outputs = []
        for source in inp.chunk(2, dim=0):
            x = source.clone()
            y = mlp(x)
            dist.all_reduce(y, group=nccl)
            outputs.append(y)
        return torch.cat(outputs, dim=0)

    def overlapped_microbatches(inp):
        source_a, source_b = inp.chunk(2, dim=0)
        sa.wait_stream(default)
        sb.wait_stream(default)
        # Input is already replicated after attention AR + norm. Only the
        # MLP output is reduced; both ranks issue outA, then outB.
        with torch.cuda.stream(sa):
            xa = source_a.clone()
            ya = mlp(xa)
            after_a_compute.record(sa)
            out_a = dist.all_reduce(ya, group=nccl, async_op=True)
        with torch.cuda.stream(sb):
            xb = source_b.clone()
            # Keep GEMMs serialized while permitting outA communication to
            # overlap MLP(B). There is no request/state slicing in this micro.
            sb.wait_event(after_a_compute)
            yb = mlp(xb)
            out_b = dist.all_reduce(yb, group=nccl, async_op=True)
        with torch.cuda.stream(sa):
            out_a.wait()
        with torch.cuda.stream(sb):
            out_b.wait()
        default.wait_stream(sa)
        default.wait_stream(sb)
        ya.record_stream(default)
        yb.record_stream(default)
        return torch.cat((ya,yb), dim=0)

    modes = [('single_batch',single_batch),('serial_two_batches',serial_microbatches),
             ('overlapped_two_batches',overlapped_microbatches)]
    rows = []
    for micro_tokens in args.micro_tokens:
        if micro_tokens not in (1024,2048,4096,8192):
            raise ValueError('Keep the initial sweep bounded to audited MLP shapes')
        # Post-attention norm input is replicated across TP ranks; weights remain sharded.
        torch.manual_seed(20261015 + micro_tokens)
        inp = torch.randn(2*micro_tokens,5120,device='cuda',dtype=torch.bfloat16)
        torch.cuda.synchronize()
        dist.barrier()
        reference = single_batch(inp)
        serial = serial_microbatches(inp)
        overlap = overlapped_microbatches(inp)
        torch.cuda.synchronize()
        check = {'rank':rank}
        try:
            check['serial_vs_single'] = error_metrics(serial,reference)
            check['overlap_vs_serial'] = error_metrics(overlap,serial,exact=True)
            check['status'] = 'PASS'
        except Exception as exc:
            check['status'] = 'FAIL'
            check['error'] = f'{type(exc).__name__}: {exc}'
        checks = [None,None]
        dist.all_gather_object(checks,check)
        row = dict(total_tokens=2*micro_tokens,micro_tokens=micro_tokens,checks=checks,
                   torch_version=torch.__version__, device=torch.cuda.get_device_name(),
                   hip_visible_devices=os.environ.get('HIP_VISIBLE_DEVICES'),
                   weight_dtype='int8', activation_dtype='bfloat16',
                   collective_order=['outA','outB'], input_state='already replicated after attention AR and norm',
                   gemm_shapes=[[micro_tokens,17408,5120],[micro_tokens,5120,8704]],timings={})
        rows.append(row)
        if any(x['status'] != 'PASS' for x in checks):
            if rank == 0:
                Path(args.output).write_text(json.dumps(rows,indent=2)+'\n')
            raise RuntimeError('Numerical mismatch; do not benchmark an invalid overlap candidate')
        # Rotate timing order on successive repeats, keeping both ranks identical.
        samples = {name:[] for name,_ in modes}
        for name,fn in modes:
            for _ in range(args.warmup):
                fn(inp)
        torch.cuda.synchronize()
        dist.barrier()
        for repeat in range(args.repeats):
            order = modes[repeat % len(modes):] + modes[:repeat % len(modes)]
            for name,fn in order:
                torch.cuda.synchronize()
                dist.barrier()
                start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                wall_start = time.perf_counter()
                start.record()
                for _ in range(args.iterations):
                    result = fn(inp)
                end.record()
                end.synchronize()
                sample = dict(rank=rank,gpu_ms=start.elapsed_time(end)/args.iterations,
                              wall_ms=(time.perf_counter()-wall_start)*1000/args.iterations)
                gathered = [None,None]
                dist.all_gather_object(gathered,sample)
                samples[name].append(dict(max_rank_gpu_ms=max(x['gpu_ms'] for x in gathered),
                                          max_rank_wall_ms=max(x['wall_ms'] for x in gathered),
                                          ranks=gathered))
        for name in samples:
            row['timings'][name] = dict(samples=samples[name],
                median_gpu_ms=statistics.median(x['max_rank_gpu_ms'] for x in samples[name]),
                median_wall_ms=statistics.median(x['max_rank_wall_ms'] for x in samples[name]))
        baseline = row['timings']['single_batch']['median_gpu_ms']
        serial_ms = row['timings']['serial_two_batches']['median_gpu_ms']
        overlap_ms = row['timings']['overlapped_two_batches']['median_gpu_ms']
        row['overlap_improvement_vs_single_pct'] = (1-overlap_ms/baseline)*100
        row['overlap_improvement_vs_serial_pct'] = (1-overlap_ms/serial_ms)*100
        if rank == 0:
            Path(args.output).parent.mkdir(parents=True,exist_ok=True)
            Path(args.output).write_text(json.dumps(rows,indent=2)+'\n')
            print(json.dumps(row),flush=True)
        if args.trace_dir and micro_tokens == 4096:
            trace = Path(args.trace_dir)
            trace.mkdir(parents=True,exist_ok=True)
            dist.barrier()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as prof:
                with torch.profiler.record_function('serial_two_mlp_batches'):
                    serial_microbatches(inp)
                torch.cuda.synchronize()
                with torch.profiler.record_function('overlapped_two_mlp_batches'):
                    overlapped_microbatches(inp)
                torch.cuda.synchronize()
            prof.export_chrome_trace(str(trace/f'mlp-only-overlap-rank{rank}.json'))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
