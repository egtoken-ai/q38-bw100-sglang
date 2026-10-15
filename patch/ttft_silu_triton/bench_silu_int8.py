"""Compare deployed two-kernel baseline, native fused op, and Triton variants.

Standalone GPU4 diagnostic. CPU reference checks sampled rows, including zeros,
large finite gates, and quantization halfway values. Full tensors are compared
against the actual vendor baseline. No tolerance is labeled model correctness.
"""
import argparse
import json
import os
from pathlib import Path
import statistics

import torch
from lightop.activation import fuse_silu_and_mul, lm_fuse_silu_mul_quant
from lightop.quant import per_token_quant_int8

from fused_silu_int8 import fused_silu_int8


def baseline(x):
    out = torch.empty(x.shape[0], x.shape[1] // 2, device=x.device, dtype=x.dtype)
    fuse_silu_and_mul(x, out)
    return per_token_quant_int8(out)


def torch_cpu_reference(x):
    # Match the two BF16 roundings verified by probe_activation.py: first SiLU,
    # then multiply. Neither BF16 intermediate may be skipped.
    # torch.round uses ties-to-even, independently of Triton's nearbyint.
    gate, up = x.cpu().float().chunk(2, dim=-1)
    silu = (gate / (1 + torch.exp(-gate))).to(torch.bfloat16).float()
    activation = (silu * up).to(torch.bfloat16).float()
    absmax = activation.abs().amax(-1, keepdim=True).clamp_min(1e-10)
    return torch.round(activation * (127 / absmax)).to(torch.int8), absmax / 127


def compare(actual, reference):
    aq, asc = actual
    rq, rsc = reference
    a, r = aq.float() * asc, rq.float() * rsc
    return {
        "quant_exact": bool(torch.equal(aq, rq)),
        "scale_exact": bool(torch.equal(asc, rsc)),
        "quant_mismatch_fraction": (aq != rq).float().mean().item(),
        "max_quant_step_difference": (aq.int() - rq.int()).abs().max().item(),
        "scale_max_absolute_error": (asc-rsc).abs().max().item(),
        "dequant_relative_l2_error": ((a-r).norm() / r.norm().clamp_min(1e-30)).item(),
        "dequant_max_absolute_error": (a-r).abs().max().item(),
        "all_finite": bool(torch.isfinite(a).all().item()),
    }


def measure(fn, use_graph):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    captured_output = None
    if use_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            # One replay submits ten kernels/pipelines, reducing host-launch
            # gaps. Both baseline and all candidates use the same arrangement.
            for _ in range(10):
                captured_output = fn()
        run = graph.replay
        calls_per_run, runs_per_sample = 10, 10
        graph.replay()
        torch.cuda.synchronize()
    else:
        run = fn
        calls_per_run, runs_per_sample = 1, 20
    times = []
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(runs_per_sample):
            run()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / (calls_per_run*runs_per_sample))
    return dict(median_ms=statistics.median(times), samples_ms=times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if os.environ.get("HIP_VISIBLE_DEVICES") != "4":
        raise RuntimeError("This diagnostic is restricted to physical GPU4")
    torch.set_num_threads(2)
    report = {
        "status": "RUNNING", "dtype": "bfloat16", "output_dtype": "int8",
        "scale_dtype": "float32", "width_in": 17408, "width_out": 8704,
        "model_quality_evaluated": False, "rows": [],
        "activation_rounding": "SiLU FP32 -> BF16 -> multiply FP32 -> BF16 -> quantize",
        "performance_admission": "full INT8 output and FP32 scales must exactly match vendor baseline",
        "theoretical_removed_activation_hbm_bytes_per_row": 4*8704,
        "note": "Traffic reduction is relative to the unfused baseline, not proven relative to the native fused kernel",
    }

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    for tokens in (128, 4096, 8192, 16384):
        torch.manual_seed(20261015)
        x = torch.randn(tokens, 17408, device="cuda", dtype=torch.bfloat16)
        x[0].zero_()
        x[1, :8704].zero_()
        # Large positive/negative finite gates and BF16-representable half steps.
        x[2, :8704] = torch.linspace(-80, 80, 8704, device="cuda", dtype=torch.float32).to(torch.bfloat16)
        x[3, :8704].fill_(16)
        x[3, 8704:] = ((torch.arange(8704, device="cuda") % 509 - 254).float()/32).to(torch.bfloat16)
        ref = baseline(x)
        sample_rows = 8
        cpu_ref = torch_cpu_reference(x[:sample_rows])
        gpu_ref_sample = tuple(part[:sample_rows].cpu() for part in ref)
        report.setdefault("baseline_cpu_reference", []).append({
            "tokens": tokens, "sample_rows": sample_rows,
            "error": compare(gpu_ref_sample, cpu_ref),
        })
        functions = {"vendor_unfused": lambda: baseline(x), "vendor_fused": lambda: lm_fuse_silu_mul_quant(x)}
        functions.update({f"triton_warps_{warps}": lambda w=warps: fused_silu_int8(x, num_warps=w) for warps in (4, 8, 16)})
        for name, fn in functions.items():
            row = {"tokens": tokens, "candidate": name}
            try:
                actual = fn()
                row["against_vendor_baseline"] = compare(actual, ref)
                row["against_cpu_reference_sample"] = compare(tuple(part[:sample_rows].cpu() for part in actual), cpu_ref)
                # Exact zero row is a necessary formula edge case, independent of
                # the tolerance used to describe finite random-input differences.
                row["zero_row_quant_zero"] = bool((actual[0][0] == 0).all().item())
                row["zero_row_scale"] = actual[1][0].item()
                row["zero_row_scale_matches_vendor"] = bool(torch.equal(actual[1][0], ref[1][0]))
                if not row["against_vendor_baseline"]["all_finite"]:
                    raise RuntimeError("Non-finite reconstructed output")
                if not row["zero_row_quant_zero"] or not row["zero_row_scale_matches_vendor"]:
                    raise RuntimeError("Zero-row quantization formula mismatch")
                if not (
                    row["against_vendor_baseline"]["quant_exact"]
                    and row["against_vendor_baseline"]["scale_exact"]
                ):
                    row["status"] = "NUMERICAL_MISMATCH_NOT_TIMED"
                    row["performance_candidate_admitted"] = False
                    report["rows"].append(row)
                    print(json.dumps(row), flush=True)
                    save()
                    continue
                row["performance_candidate_admitted"] = True
                if name.startswith("triton"):
                    _, compiled = fused_silu_int8(x, num_warps=int(name.rsplit("_", 1)[1]), return_kernel=True)
                    row["compiled_kernel"] = {
                        "n_regs": getattr(compiled, "n_regs", None),
                        "n_spills": getattr(compiled, "n_spills", None),
                    }
                row["eager"] = measure(fn, False)
                row["cuda_graph"] = measure(fn, True)
                row["status"] = "MEASURED_EXACT_QUANT_AND_SCALE"
            except Exception as exc:
                row.update(status="FAILED", error=f"{type(exc).__name__}: {exc}"[:1500])
            report["rows"].append(row)
            print(json.dumps(row), flush=True)
            save()
    report["status"] = "COMPLETE_WITH_DIAGNOSTICS"
    save()


if __name__ == "__main__":
    main()
