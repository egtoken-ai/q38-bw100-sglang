"""Isolated M32768 comparison of 4/8-way post-attention boundary pipelines.

Only diagnostic. Inputs/weights and GEMM precision are fixed across modes.
Both baseline and candidates reset the same preallocated input/residual buffers.
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
from lightop.norm import gemma_fused_add_rmsnorm
from lightop.quant import per_token_quant_int8


def metrics(actual, reference, exact=False):
    diff = actual.float() - reference.float()
    result = dict(max_abs=float(diff.abs().max()),
                  relative_l2=float(diff.norm()/reference.float().norm().clamp_min(1e-12)),
                  mismatches=int(torch.count_nonzero(actual != reference)))
    torch.testing.assert_close(actual,reference,rtol=0 if exact else .015625,
                               atol=0 if exact else .015625)
    if result['relative_l2'] > .002:
        raise AssertionError(f'Relative L2 exceeds 0.2%: {result}')
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--total-tokens',type=int,nargs='+',default=[32768],choices=[32768])
    p.add_argument('--parts',type=int,nargs='+',default=[4],choices=[4,8])
    p.add_argument('--warmup',type=int,default=3)
    p.add_argument('--iterations',type=int,default=5)
    p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--trace-dir')
    args = p.parse_args()
    rank = int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE']) != 2:
        raise RuntimeError('Requires exactly TP2')
    if os.environ.get('TORCH_NCCL_BLOCKING_WAIT','0') != '0':
        raise RuntimeError('Blocking NCCL wait invalidates this overlap diagnostic')
    torch.cuda.set_device(rank)
    torch.set_num_threads(2)
    dist.init_process_group('gloo',timeout=timedelta(seconds=180))
    nccl = dist.new_group(backend='nccl',timeout=timedelta(seconds=180))
    if rank == 0:
        Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    # Rank-specific INT8 column/row-parallel MLP weight shards.
    torch.manual_seed(20261015 + rank)
    gate_weight = torch.randint(-16,16,(17408,5120),device='cuda',dtype=torch.int8).T
    down_weight = torch.randint(-16,16,(5120,8704),device='cuda',dtype=torch.int8).T
    gate_scale = torch.full((17408,1),.001,device='cuda',dtype=torch.float32)
    down_scale = torch.full((5120,1),.001,device='cuda',dtype=torch.float32)
    # Same actual Gemma weight parameter on both TP ranks. Pass the original
    # BF16 parameter, NOT a separately rounded BF16 (1 + weight) gamma.
    torch.manual_seed(50001)
    norm_weight = (torch.randn(5120,device='cuda')*.1).bfloat16()
    norm_weight[:8] = torch.tensor([0,.001953125,-.001953125,.00390625,
                                    -.00390625,.0078125,-.0078125,.1],
                                   device='cuda',dtype=torch.bfloat16)
    streams = [torch.cuda.Stream() for _ in range(max(args.parts))]
    compute_done = [torch.cuda.Event() for _ in range(max(args.parts))]
    default = torch.cuda.current_stream()

    def mlp(x):
        qx,sx = per_token_quant_int8(x)
        gate = gemm_ops.triton_scaled_mm(qx,gate_weight,sx,gate_scale,torch.bfloat16)
        activation = torch.empty((x.shape[0],8704),device=x.device,dtype=torch.bfloat16)
        fuse_silu_and_mul(gate,activation)
        qa,sa = per_token_quant_int8(activation)
        return gemm_ops.triton_scaled_mm(qa,down_weight,sa,down_scale,torch.bfloat16)

    results = []
    for total_tokens in args.total_tokens:
        if total_tokens != 32768:
            raise ValueError('This experiment is restricted to the serving shape M32768')
        torch.manual_seed(61000 + total_tokens + rank)
        attention_partial = torch.randn(total_tokens,5120,device='cuda',dtype=torch.bfloat16)
        torch.manual_seed(62000 + total_tokens)
        residual_input = torch.randn_like(attention_partial)
        work_x = torch.empty_like(attention_partial)
        work_r = torch.empty_like(residual_input)

        def reset():
            work_x.copy_(attention_partial)
            work_r.copy_(residual_input)

        def single():
            reset()
            dist.all_reduce(work_x,group=nccl)
            normalized,residual = gemma_fused_add_rmsnorm(work_x,work_r,norm_weight,1e-6)
            output = mlp(normalized)
            dist.all_reduce(output,group=nccl)
            return output,residual,normalized

        def serial(parts):
            reset()
            outputs = []
            for x,r in zip(work_x.chunk(parts,0),work_r.chunk(parts,0)):
                dist.all_reduce(x,group=nccl)
                normalized,_ = gemma_fused_add_rmsnorm(x,r,norm_weight,1e-6)
                output = mlp(normalized)
                dist.all_reduce(output,group=nccl)
                outputs.append(output)
            return torch.cat(outputs,0),work_r,work_x

        def pipeline(parts):
            reset()
            xs,rs = work_x.chunk(parts,0),work_r.chunk(parts,0)
            input_works,output_works,outputs = [],[],[]
            # Identical rank issue order: every in_i, then every out_i.
            for i in range(parts):
                streams[i].wait_stream(default)
                with torch.cuda.stream(streams[i]):
                    input_works.append(dist.all_reduce(xs[i],group=nccl,async_op=True))
            for i in range(parts):
                with torch.cuda.stream(streams[i]):
                    input_works[i].wait()
                    if i:
                        streams[i].wait_event(compute_done[i-1])
                    normalized,_ = gemma_fused_add_rmsnorm(xs[i],rs[i],norm_weight,1e-6)
                    output = mlp(normalized)
                    compute_done[i].record(streams[i])
                    output_works.append(dist.all_reduce(output,group=nccl,async_op=True))
                    outputs.append(output)
            for i in range(parts):
                with torch.cuda.stream(streams[i]):
                    output_works[i].wait()
                default.wait_stream(streams[i])
                outputs[i].record_stream(default)
            return torch.cat(outputs,0),work_r,work_x

        modes = [('single', single)]
        for parts in args.parts:
            modes.extend([(f'serial_{parts}', lambda parts=parts: serial(parts)),
                          (f'pipeline_{parts}', lambda parts=parts: pipeline(parts))])
        torch.cuda.synchronize()
        dist.barrier()
        reference = tuple(t.clone() for t in single())
        torch.cuda.synchronize()
        row = dict(total_tokens=total_tokens,parts=args.parts,checks={},timings={},
                   norm_impl='lightop.norm.gemma_fused_add_rmsnorm(original BF16 weight, eps=1e-6)',
                   dtypes=dict(weights='int8',activation='bfloat16',residual='bfloat16'),
                   timing='same input/residual buffer reset included in all arms',
                   collective_order='in_0..in_N-1, out_0..out_N-1; GEMMs serialized by events')
        results.append(row)
        serial_refs = {}
        for name,fn in modes[1:]:
            actual = fn()
            torch.cuda.synchronize()
            check = dict(rank=rank)
            try:
                check['output_vs_single'] = metrics(actual[0],reference[0])
                # Norm and residual must agree exactly, not merely final MLP.
                check['residual_vs_single'] = metrics(actual[1],reference[1],exact=True)
                check['normalized_vs_single'] = metrics(actual[2],reference[2],exact=True)
                if name.startswith('serial'):
                    serial_refs[name[-1]] = tuple(t.clone() for t in actual)
                else:
                    serial_ref = serial_refs[name[-1]]
                    check['output_vs_serial'] = metrics(actual[0],serial_ref[0],exact=True)
                check['status'] = 'PASS'
            except Exception as exc:
                check['status'] = 'FAIL'
                check['error'] = f'{type(exc).__name__}: {exc}'
            checks = [None,None]
            dist.all_gather_object(checks,check)
            row['checks'][name] = checks
            if any(x['status'] != 'PASS' for x in checks):
                if rank == 0:
                    Path(args.output).write_text(json.dumps(results,indent=2)+'\n')
                raise RuntimeError(f'{name} failed numerical parity; no performance claim')
        # Also prove that residual checking is not comparing unchanged buffers.
        delta = reference[1].float()-residual_input.float()
        row['residual_update'] = dict(changed_elements=int(torch.count_nonzero(delta)),
                                      max_abs=float(delta.abs().max()))
        if row['residual_update']['changed_elements'] == 0:
            raise AssertionError('Vacuous residual update check')
        samples = {name:[] for name,_ in modes}
        for _,fn in modes:
            for _ in range(args.warmup):
                fn()
        torch.cuda.synchronize()
        dist.barrier()
        for repeat in range(args.repeats):
            offset=repeat%len(modes)
            for name,fn in modes[offset:]+modes[:offset]:
                torch.cuda.synchronize()
                dist.barrier()
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                allocated_before=torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                wall_start=time.perf_counter()
                start.record()
                for _ in range(args.iterations):
                    output=fn()
                end.record()
                end.synchronize()
                sample=dict(rank=rank,gpu_ms=start.elapsed_time(end)/args.iterations,
                            wall_ms=(time.perf_counter()-wall_start)*1000/args.iterations,
                            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                            peak_increment_bytes=max(0,torch.cuda.max_memory_allocated()-allocated_before))
                gathered=[None,None]
                dist.all_gather_object(gathered,sample)
                samples[name].append(dict(max_rank_gpu_ms=max(x['gpu_ms'] for x in gathered),
                                          max_rank_wall_ms=max(x['wall_ms'] for x in gathered),ranks=gathered))
        for name,_ in modes:
            row['timings'][name]=dict(samples=samples[name],
                median_gpu_ms=statistics.median(x['max_rank_gpu_ms'] for x in samples[name]),
                median_wall_ms=statistics.median(x['max_rank_wall_ms'] for x in samples[name]))
        single_ms=row['timings']['single']['median_gpu_ms']
        for n in args.parts:
            pipeline_ms=row['timings'][f'pipeline_{n}']['median_gpu_ms']
            serial_ms=row['timings'][f'serial_{n}']['median_gpu_ms']
            row[f'pipeline_{n}_improvement_vs_single_pct']=(1-pipeline_ms/single_ms)*100
            row[f'pipeline_{n}_improvement_vs_serial_pct']=(1-pipeline_ms/serial_ms)*100
        if rank==0:
            Path(args.output).write_text(json.dumps(results,indent=2)+'\n')
            print(json.dumps(row),flush=True)
        if args.trace_dir and total_tokens==32768:
            trace=Path(args.trace_dir);trace.mkdir(parents=True,exist_ok=True)
            dist.barrier()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as prof:
                for name,fn in modes:
                    with torch.profiler.record_function(name):
                        fn()
                    torch.cuda.synchronize()
            prof.export_chrome_trace(str(trace/f'norm-mlp-parts48-rank{rank}.json'))
    dist.barrier()
    dist.destroy_process_group()


if __name__=='__main__':
    main()
