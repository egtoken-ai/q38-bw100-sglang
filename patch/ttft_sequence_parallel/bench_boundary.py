"""Bounded TP2 boundary diagnostic: AR+norm+quant versus RS+norm+quant+AG.

BF16 partials/residuals; INT8 quantized activations; original FP32 token scales.
No production patch or approximation. Each candidate must exactly match the
same baseline quantization and its corresponding residual token slice.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import statistics

if os.environ.get("HIP_VISIBLE_DEVICES") and not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["HIP_VISIBLE_DEVICES"]

import torch
import torch.distributed as dist
from lightop.norm import gemma_fused_add_rmsnorm
from lightop.quant import per_token_quant_int8


def error_summary(actual, reference):
    delta = actual.float() - reference.float()
    return {
        "exact": bool(torch.equal(actual, reference)),
        "max_absolute_error": delta.abs().max().item(),
        "mismatch_fraction": (actual != reference).float().mean().item(),
        "all_finite": bool(torch.isfinite(actual).all().item()),
    }


def measure_pair(pipelines, resets):
    # Both paths reset every mutable input/residual before every invocation.
    # Resets are outside the event interval; no accumulated residual drift.
    for name in pipelines:
        for _ in range(10):
            resets[name]()
            pipelines[name]()
        torch.cuda.synchronize()
    batches = {name: [] for name in pipelines}
    for batch in range(5):
        order = list(pipelines)
        if batch % 2:
            order.reverse()
        for name in order:
            dist.barrier()
            events = []
            for _ in range(20):
                resets[name]()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                pipelines[name]()
                end.record()
                events.append((start, end))
            torch.cuda.synchronize()
            batches[name].append(statistics.mean(a.elapsed_time(b) for a, b in events))
    return {
        name: {"median_ms": statistics.median(values), "batch_mean_ms": values}
        for name, values in batches.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.set_num_threads(2)
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=90))
    if dist.get_world_size() != 2 or torch.version.hip is None:
        raise RuntimeError("Requires HIP TP2")
    if os.environ.get("HIP_VISIBLE_DEVICES") != "4,5" or os.environ.get("CUDA_VISIBLE_DEVICES") != "4,5":
        raise RuntimeError("Restricted to the experiment's physical GPUs 4,5")
    group = dist.new_group(backend="nccl", timeout=datetime.timedelta(seconds=90))
    report = {
        "status": "RUNNING",
        "scope": "boundary microbenchmark only; not model integration or TTFT acceptance",
        "partial_dtype": "bfloat16", "residual_dtype": "bfloat16",
        "weight_dtype": "bfloat16", "activation_dtype": "int8", "scale_dtype": "float32",
        "world_size": 2, "hidden_size": 5120, "epsilon": 1e-6,
        "timing": "10 warmups/path, 5 alternating batches/path, 20 event pairs/batch; resets excluded",
        "rows": [],
        "environment": {key: os.environ.get(key) for key in (
            "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "NCCL_ALGO", "NCCL_PROTO",
        )},
    }

    def save():
        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        for tokens in (4096, 8192, 16384):
            # Rank-specific partials and identical residual/gamma on both ranks.
            torch.manual_seed(20261015 + rank)
            partial = torch.randn(tokens, 5120, device="cuda", dtype=torch.bfloat16)
            torch.manual_seed(20261030)
            residual = torch.randn_like(partial)
            weight = torch.randn(5120, device="cuda", dtype=torch.bfloat16) * .1
            local_tokens = tokens // 2
            token_slice = slice(rank * local_tokens, (rank + 1) * local_tokens)
            full_work = torch.empty_like(partial)
            full_residual = torch.empty_like(residual)
            local_work = torch.empty(local_tokens, 5120, device="cuda", dtype=torch.bfloat16)
            local_residual = torch.empty_like(local_work)
            full_quant = torch.empty(tokens, 5120, device="cuda", dtype=torch.int8)
            full_scale = torch.empty(tokens, 1, device="cuda", dtype=torch.float32)

            def reset_baseline():
                full_work.copy_(partial)
                full_residual.copy_(residual)

            def baseline():
                dist.all_reduce(full_work, group=group)
                normalized, updated_residual = gemma_fused_add_rmsnorm(
                    full_work, full_residual, weight, 1e-6)
                quant, scale = per_token_quant_int8(normalized)
                if quant.dtype != torch.int8 or scale.dtype != torch.float32:
                    raise RuntimeError("Unexpected deployed quantization dtype")
                return quant, scale, updated_residual, normalized

            def reset_candidate():
                # reduce_scatter overwrites every local_work element. Its input
                # partial is read-only, so only residual needs to be restored.
                local_residual.copy_(residual[token_slice])

            def candidate():
                dist.reduce_scatter_tensor(local_work, partial, group=group)
                normalized, updated_residual = gemma_fused_add_rmsnorm(
                    local_work, local_residual, weight, 1e-6)
                local_quant, local_scale = per_token_quant_int8(normalized)
                if local_quant.dtype != torch.int8 or local_scale.dtype != torch.float32:
                    raise RuntimeError("Unexpected deployed quantization dtype")
                # all_gather_into_tensor concatenates rank-major token slices,
                # reproducing the baseline's original full token order.
                dist.all_gather_into_tensor(full_quant, local_quant, group=group)
                dist.all_gather_into_tensor(full_scale, local_scale, group=group)
                return full_quant, full_scale, updated_residual, normalized

            reset_baseline()
            reference = baseline()
            reset_candidate()
            actual = candidate()
            torch.cuda.synchronize()
            correctness = {
                "full_quant": error_summary(actual[0], reference[0]),
                "full_scale": error_summary(actual[1], reference[1]),
                "local_residual": error_summary(actual[2], reference[2][token_slice]),
                "local_normalized": error_summary(actual[3], reference[3][token_slice]),
            }
            local_ok = all(item["exact"] and item["all_finite"] for item in correctness.values())
            statuses = [None, None]
            dist.all_gather_object(statuses, local_ok)
            row = {"rank": rank, "tokens": tokens, "correctness": correctness}
            if all(statuses):
                row["timings"] = measure_pair(
                    {"allreduce_norm_quant": baseline, "reduce_scatter_norm_quant_allgather": candidate},
                    {"allreduce_norm_quant": reset_baseline, "reduce_scatter_norm_quant_allgather": reset_candidate},
                )
                old = row["timings"]["allreduce_norm_quant"]["median_ms"]
                new = row["timings"]["reduce_scatter_norm_quant_allgather"]["median_ms"]
                row["boundary_latency_improvement_percent"] = (old-new)/old*100
            gathered = [None, None]
            dist.all_gather_object(gathered, row)
            if rank == 0:
                report["rows"].extend(gathered)
                print(json.dumps(gathered), flush=True)
                save()
            if not all(statuses):
                report["status"] = "NUMERICAL_MISMATCH"
                save()
                return
        report["status"] = "MEASURED_EXACT_BOUNDARY"
        save()
    except Exception as exc:
        report.update(status="UNAVAILABLE_OR_FAILED", reason=f"{type(exc).__name__}: {exc}")
        save()
        raise
    finally:
        dist.destroy_process_group(group)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
