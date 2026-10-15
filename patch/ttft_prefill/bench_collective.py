"""Two-rank BF16 RCCL vs unquantized QuickReduce at Prefill tensor sizes."""
import argparse
import json
import os
from pathlib import Path
import statistics

import torch
import torch.distributed as dist


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    args = p.parse_args()
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    torch.set_num_threads(2)
    os.environ['ROCM_QUICK_REDUCE_QUANTIZATION'] = 'FP'
    os.environ['ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16'] = '0'
    os.environ['ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB'] = '256'
    dist.init_process_group('gloo')
    nccl = dist.new_group(backend='nccl')
    from sglang.srt.distributed.device_communicators.quick_all_reduce import QuickAllReduce
    qr = QuickAllReduce(dist.group.WORLD, torch.device('cuda', rank))
    if qr.disabled:
        if rank == 0:
            Path(args.output).write_text(json.dumps({'status': 'UNAVAILABLE'}) + '\n')
            print('QuickReduce unavailable', flush=True)
        dist.destroy_process_group()
        return
    rows = []
    for tokens in [1024, 4096, 8192, 16384]:
        torch.manual_seed(20261015 + rank)
        inp = torch.randn(tokens, 5120, device='cuda', dtype=torch.bfloat16)
        work = inp.clone()
        ref = inp.clone()
        dist.all_reduce(ref, group=nccl)
        qr_output = torch.empty_like(inp)
        actual = qr.quick_all_reduce(inp, out=qr_output)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, ref, atol=0, rtol=0)
        for name in ['rccl', 'quick-fp']:
            samples = []
            for i in range(8):
                work.copy_(inp)
                torch.cuda.synchronize()
                dist.barrier()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                if name == 'rccl':
                    dist.all_reduce(work, group=nccl)
                else:
                    qr.quick_all_reduce(inp, out=qr_output)
                end.record()
                end.synchronize()
                if i >= 3:
                    samples.append(start.elapsed_time(end))
            row = dict(tokens=tokens, bytes=inp.numel()*2, rank=rank,
                       candidate=name, median_ms=statistics.median(samples), exact=True)
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, row)
            if rank == 0:
                rows.extend(gathered)
                Path(args.output).write_text(json.dumps(rows, indent=2) + '\n')
                print(json.dumps(gathered), flush=True)
    qr.close()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
