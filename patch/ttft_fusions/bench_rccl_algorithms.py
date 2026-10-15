"""Run one RCCL environment candidate in a fresh two-rank torchrun process.

No dtype conversion/quantization: validates BF16 sum against an FP32 reference
then times BF16 all_reduce. Caller owns GPU reservation and candidate sequencing.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import statistics

import torch
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tokens", default="1024,4096,8192,16384")
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.set_num_threads(2)
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=90))
    if dist.get_world_size() != 2:
        raise RuntimeError("This diagnostic is restricted to TP2")
    group = dist.new_group(backend="nccl", timeout=datetime.timedelta(seconds=90))
    rows = []
    for tokens in [int(value) for value in args.tokens.split(",")]:
        # Deterministic, nonconstant BF16 operands with known FP32 reference.
        # Check correctness on actual data; use zeros for repeated timing so an
        # in-place sum cannot overflow after many back-to-back collectives.
        pattern = (torch.arange(tokens * 5120, device="cuda") % 257).float() / 32 - 4
        lhs = pattern.to(torch.bfloat16)
        rhs = (pattern * -.5 + .015625).to(torch.bfloat16)
        expected = (lhs.float() + rhs.float()).to(torch.bfloat16)
        work = (lhs if rank == 0 else rhs).clone()
        dist.all_reduce(work, group=group)
        torch.testing.assert_close(work, expected, atol=0, rtol=0)
        work.zero_()
        for _ in range(10):
            dist.all_reduce(work, group=group)
        torch.cuda.synchronize()
        samples = []
        for _ in range(5):
            dist.barrier()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(20):
                dist.all_reduce(work, group=group)
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) / 20)
        row = dict(tokens=tokens, bytes=work.numel() * work.element_size(),
                   rank=rank, median_ms=statistics.median(samples),
                   samples_ms=samples, exact_bf16_sum=True)
        gathered = [None, None]
        dist.all_gather_object(gathered, row)
        if rank == 0:
            rows.extend(gathered)
            metadata = {
                "environment": {key: os.environ.get(key) for key in [
                    "NCCL_ALGO", "NCCL_PROTO", "NCCL_MIN_NCHANNELS",
                    "NCCL_MAX_NCHANNELS", "HIP_VISIBLE_DEVICES"]},
                "dtype": "bfloat16", "ranks": 2, "rows": rows,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(metadata, indent=2) + "\n")
            print(json.dumps(gathered), flush=True)
    dist.destroy_process_group(group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
