"""Require full exactness against vendor residual GemmaNorm+INT8 quant first.

No GPU work is performed merely by importing this file. The experiment owner
runs the bounded diagnostic after reserving GPU4. Candidate timing is admitted
only after full quant/scales/residual/norm match, never from CPU formula alone.
"""
import argparse
import json
import os
from pathlib import Path
import statistics

import torch
from lightop.norm import gemma_fused_add_rmsnorm
from lightop.quant import per_token_quant_int8

from fused_norm_quant import fused_norm_quant


def stats(a, b):
    delta = a.float()-b.float()
    return dict(exact=bool(torch.equal(a, b)),
                mismatch_fraction=(a != b).float().mean().item(),
                max_absolute_error=delta.abs().max().item(),
                relative_l2_error=(delta.norm()/b.float().norm().clamp_min(1e-30)).item(),
                all_finite=bool(torch.isfinite(a).all().item()))


def cpu_reference(x, residual, weight, round_residual):
    combined = x.cpu().float() + residual.cpu().float()
    stored_residual = combined.to(torch.bfloat16)
    if round_residual:
        combined = stored_residual.float()
    variance = combined.square().mean(dim=-1, keepdim=True)
    normalized = ((combined * torch.rsqrt(variance + 1e-6)) * (weight.cpu().float()+1)).to(torch.bfloat16)
    value = normalized.float()
    absmax = value.abs().amax(-1, keepdim=True).clamp_min(1e-10)
    return torch.round(value*(127/absmax)).to(torch.int8), absmax/127, stored_residual, normalized


def timing(fn, reset, use_graph):
    for _ in range(10):
        reset()
        fn()
    torch.cuda.synchronize()
    if use_graph:
        # Capture reset and operation separately. Replay reset before start event
        # on every sample to preserve the same state without timing reset work.
        reset_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(reset_graph):
            reset()
        operation_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(operation_graph):
            held_output = fn()
        reset_fn, operation = reset_graph.replay, operation_graph.replay
    else:
        reset_fn, operation = reset, fn
    means = []
    for _ in range(5):
        events = []
        for _ in range(20):
            reset_fn()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            operation()
            end.record()
            events.append((start, end))
        torch.cuda.synchronize()
        means.append(statistics.mean(a.elapsed_time(b) for a, b in events))
    return dict(median_ms=statistics.median(means), batch_mean_ms=means)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--tokens", default="128,4096,8192,16384")
    args = p.parse_args()
    if os.environ.get("HIP_VISIBLE_DEVICES") != "4":
        raise RuntimeError("Diagnostic restricted to physical GPU4")
    torch.set_num_threads(2)
    report = dict(status="RUNNING", scope="residual-present GemmaNorm boundary only",
                  exact_admission="full INT8 quant, FP32 scale, BF16 residual and optional norm",
                  no_service_integration=True, rows=[], cpu_diagnostics=[])

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+"\n")

    save()
    for tokens in (int(v) for v in args.tokens.split(",")):
        if tokens not in (128, 4096, 8192, 16384):
            raise ValueError("Unsupported bounded shape")
        torch.manual_seed(20261015)
        source_x = torch.randn(tokens, 5120, device="cuda", dtype=torch.bfloat16)
        source_r = torch.randn_like(source_x)
        weight = (torch.randn(5120, device="cuda", dtype=torch.bfloat16)*.1).contiguous()
        source_x[0].zero_(); source_r[0].zero_()
        source_r[1].copy_(-source_x[1])  # Exact cancellation and scale-floor case.
        # Small offsets expose rounding loss from precomputing gamma in BF16.
        weight[::8] = .001953125
        bx, br = torch.empty_like(source_x), torch.empty_like(source_r)
        cx, cr = torch.empty_like(source_x), torch.empty_like(source_r)

        def reset_b():
            bx.copy_(source_x); br.copy_(source_r)

        def reset_c():
            cx.copy_(source_x); cr.copy_(source_r)

        def baseline():
            norm, residual = gemma_fused_add_rmsnorm(bx, br, weight, 1e-6)
            quant, scale = per_token_quant_int8(norm)
            return quant, scale, residual, norm

        reset_b()
        reference = tuple(t.clone() for t in baseline())
        for rounded in (False, True):
            cpu = cpu_reference(source_x[:8], source_r[:8], weight, rounded)
            report["cpu_diagnostics"].append(dict(tokens=tokens, round_residual_for_norm=rounded,
                comparisons={name: stats(a, b[:8].cpu()) for name, a, b in zip(("quant", "scale", "residual", "norm"), cpu, reference)}))
        row = dict(tokens=tokens, candidate="vendor_norm_then_quant", status="MEASURED_BASELINE")
        row["eager"] = timing(baseline, reset_b, False)
        row["cuda_graph"] = timing(baseline, reset_b, True)
        report["rows"].append(row); print(json.dumps(row), flush=True); save()
        for rounded in (False, True):
            for warps in (4, 8):
                # First require norm output too, even when timing q-only mode.
                reset_c()
                row = dict(tokens=tokens, candidate="triton_norm_quant", num_warps=warps,
                           round_residual_for_norm=rounded)
                try:
                    actual = fused_norm_quant(cx, cr, weight, emit_norm=True,
                        round_residual_for_norm=rounded, num_warps=warps)
                    row["correctness"] = {name: stats(a, b) for name, a, b in zip(("quant", "scale", "residual", "norm"), actual, reference)}
                    exact = all(v["exact"] and v["all_finite"] for v in row["correctness"].values())
                    row["status"] = "EXACT" if exact else "NUMERICAL_MISMATCH_NOT_TIMED"
                    if exact:
                        row["timings"] = {}
                        for emit in (False, True):
                            # Also verify the no-norm-store specialization itself.
                            reset_c()
                            verify = fused_norm_quant(cx, cr, weight, emit_norm=emit,
                                round_residual_for_norm=rounded, num_warps=warps)
                            if not all(torch.equal(a, b) for a, b in zip(verify[:3], reference[:3])):
                                raise RuntimeError("Specialization changed exact quant/scale/residual")
                            fn = lambda e=emit: fused_norm_quant(cx, cr, weight, emit_norm=e,
                                round_residual_for_norm=rounded, num_warps=warps)
                            row["timings"]["emit_norm" if emit else "quant_only"] = {
                                "eager": timing(fn, reset_c, False), "cuda_graph": timing(fn, reset_c, True)}
                except Exception as exc:
                    row.update(status="FAILED", error=f"{type(exc).__name__}: {exc}"[:1500])
                report["rows"].append(row); print(json.dumps(row), flush=True); save()
    report["status"] = "COMPLETE_WITH_DIAGNOSTICS"
    save()


if __name__ == "__main__":
    main()
