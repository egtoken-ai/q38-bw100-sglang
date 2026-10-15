"""Screen lossy INT8 QuickReduce with direct BF16 input and output on TP2.

Only the experiment owner runs this on reserved GPU 4/5. This diagnostic never
enables a model-serving path and never interprets numeric error as model quality.
It refuses FP16 casting instead of falling back to an unsupported dtype.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import statistics

# QuickAllReduce uses CUDA_VISIBLE_DEVICES to identify physical devices, even
# when running on HIP. Set both before importing torch or initializing HIP.
if os.environ.get("HIP_VISIBLE_DEVICES") and not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["HIP_VISIBLE_DEVICES"]
os.environ["ROCM_QUICK_REDUCE_QUANTIZATION"] = "INT8"
os.environ["ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16"] = "0"
os.environ["ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB"] = "256"

import torch
import torch.distributed as dist


def numeric_error(actual, reference):
    a, r = actual.float(), reference.float()
    difference = a - r
    return {
        "relative_l2_error": (difference.norm() / r.norm().clamp_min(1e-30)).item(),
        "max_absolute_error": difference.abs().max().item(),
        "mean_absolute_error": difference.abs().mean().item(),
        "mean_signed_error": difference.mean().item(),
        "mismatch_fraction": (actual != reference).float().mean().item(),
        "all_finite": bool(torch.isfinite(actual).all().item()),
    }


def timed(fn, reset):
    # Each event pair covers one collective. Reset copies are explicitly outside
    # the timed interval, preventing repeated in-place BF16 sums from overflowing.
    # Launch 20 pairs before synchronization, then aggregate their device times;
    # this is not a single noisy host wall-time measurement.
    for _ in range(10):
        reset()
        fn()
    torch.cuda.synchronize()
    batch_means = []
    for _ in range(5):
        dist.barrier()
        events = []
        for _ in range(20):
            reset()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            events.append((start, end))
        torch.cuda.synchronize()
        batch_means.append(statistics.mean(start.elapsed_time(end) for start, end in events))
    return {"median_ms": statistics.median(batch_means), "batch_mean_ms": batch_means}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    torch.set_num_threads(2)
    dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=90))
    if dist.get_world_size() != 2:
        raise RuntimeError("This candidate is restricted to two ranks")
    nccl = dist.new_group(backend="nccl", timeout=datetime.timedelta(seconds=90))
    from sglang.srt.distributed.device_communicators.quick_all_reduce import (
        QuickAllReduce, QuickReduceRegime,
    )
    qr = QuickAllReduce(dist.group.WORLD, torch.device("cuda", rank))
    metadata = {
        "status": "RUNNING",
        "dtype": "bfloat16",
        "communication_quantization": "INT8",
        "cast_bf16_to_fp16": False,
        "lossy": True,
        "model_quality_evaluated": False,
        "timing": "10 warmups; 5 batches of 20 CUDA-event pairs; reset copies excluded",
        "environment": {key: os.environ.get(key) for key in (
            "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
            "ROCM_QUICK_REDUCE_QUANTIZATION", "ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16",
            "ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB", "NCCL_ALGO", "NCCL_PROTO",
        )},
        "rows": [],
    }

    def save():
        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(metadata, indent=2) + "\n")

    def available_on_both(local):
        statuses = [None, None]
        dist.all_gather_object(statuses, bool(local))
        return all(statuses)

    available = (
        not qr.disabled
        and not getattr(qr, "use_fp16_kernels", True)
        and getattr(qr, "qr_quant_level", None) == QuickReduceRegime.INT8
        and torch.bfloat16 in qr._SUPPORTED_DTYPES
    )
    if not available_on_both(available):
        metadata.update(status="UNAVAILABLE", reason="Direct BF16 INT8 QuickReduce unavailable; FP16 fallback forbidden")
        save()
        qr.close()
        dist.destroy_process_group(nccl)
        dist.destroy_process_group()
        return
    save()
    try:
        for tokens in (4096, 8192, 16384):
            torch.manual_seed(20261015 + rank)
            inp = torch.randn(tokens, 5120, device="cuda", dtype=torch.bfloat16)
            if not available_on_both(qr.should_quick_allreduce(inp)):
                metadata.update(status="UNAVAILABLE", reason=f"BF16 input shape {list(inp.shape)} rejected")
                save()
                return
            work = inp.clone()
            reference = inp.clone()
            dist.all_reduce(reference, group=nccl)
            out = torch.empty_like(inp)
            actual = qr.quick_all_reduce(inp, out=out)
            torch.cuda.synchronize()
            if actual.dtype != torch.bfloat16 or qr.use_fp16_kernels:
                metadata.update(status="UNAVAILABLE", reason="BF16 direct-path contract not retained")
                save()
                return
            error = numeric_error(actual, reference)
            # This is descriptive error reporting, not an exactness/quality gate.
            # Non-finite output is not a usable numerical result and stops timing.
            if not available_on_both(error["all_finite"]):
                metadata.update(status="NUMERIC_FAILURE", reason="Non-finite INT8 QuickReduce output")
                metadata["rows"].append(dict(rank=rank, tokens=tokens, **error))
                save()
                return
            timings = {}
            for name in ("rccl_bf16", "quick_int8_bf16"):
                if name == "rccl_bf16":
                    fn = lambda: dist.all_reduce(work, group=nccl)
                else:
                    fn = lambda: qr.quick_all_reduce(work, out=out)
                # Same reset/readiness dependency on both paths; copy not timed.
                timings[name] = timed(fn, lambda: work.copy_(inp))
            row = dict(rank=rank, tokens=tokens, bytes=inp.numel() * 2,
                       numeric_error=error, timings=timings)
            gathered = [None, None]
            dist.all_gather_object(gathered, row)
            if rank == 0:
                metadata["rows"].extend(gathered)
                save()
                print(json.dumps(gathered), flush=True)
        metadata["status"] = "MEASURED_LOSSY_CANDIDATE"
        save()
    except (RuntimeError, NotImplementedError) as exc:
        # Do not retry through FP16. A distributed kernel failure may still require
        # torchrun/outer timeout to terminate a peer blocked inside the collective.
        metadata.update(status="UNAVAILABLE", reason=f"{type(exc).__name__}: {exc}")
        save()
        raise
    finally:
        qr.close()
        dist.destroy_process_group(nccl)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
