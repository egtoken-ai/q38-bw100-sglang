"""Correctness and paired device timings; no model weights required."""

import argparse
import importlib
import itertools
import json
import statistics
import time
from pathlib import Path

import torch
import triton

from sglang.srt.layers.attention.fla import chunk_fwd as candidate
from sglang.srt.layers.attention.fla import chunk_fwd_baseline as baseline
from sglang.srt.layers.attention.fla.chunk_scaled_dot_kkt import (
    chunk_scaled_dot_kkt_fwd,
)
from sglang.srt.layers.attention.fla.cumsum import chunk_local_cumsum
from sglang.srt.layers.attention.fla.index import prepare_chunk_indices
from sglang.srt.layers.attention.fla.l2norm import l2norm_fwd
from sglang.srt.layers.attention.fla.solve_tril import solve_tril


def inputs(lengths, dtype, hg=2, h=6, kdim=128, vdim=128, packed=True):
    b, t = (1, sum(lengths)) if packed else (len(lengths), lengths[0])
    k = torch.nn.functional.normalize(
        torch.randn(b, t, hg, kdim, device="cuda"), dim=-1
    ).to(dtype)
    v = torch.randn(b, t, h, vdim, device="cuda", dtype=dtype)
    beta = torch.rand(b, t, h, device="cuda", dtype=torch.float32)
    raw_g = -torch.rand(b, t, h, device="cuda") * 0.1
    cu = (
        torch.tensor(
            [0] + list(itertools.accumulate(lengths)), device="cuda", dtype=torch.int32
        )
        if packed
        else None
    )
    idx = prepare_chunk_indices(cu, 64) if cu is not None else None
    g = chunk_local_cumsum(raw_g, chunk_size=64, cu_seqlens=cu, chunk_indices=idx)
    return dict(k=k, v=v, beta=beta, g=g, cu_seqlens=cu, chunk_indices=idx), raw_g


def inverse_reference(x, lengths):
    k, g, beta = (x[name].double().cpu() for name in ("k", "g", "beta"))
    b, t, hg, kd = k.shape
    h = beta.shape[-1]
    out = torch.zeros(b, t, h, 64, dtype=torch.float64)
    offset = 0
    for seq, length in enumerate(lengths):
        bi, start = (0, offset) if x["cu_seqlens"] is not None else (seq, 0)
        for pos in range(0, length, 64):
            n = min(64, length - pos)
            sl = slice(start + pos, start + pos + n)
            kk = k[bi, sl].repeat_interleave(h // hg, dim=1).transpose(0, 1)
            gg = g[bi, sl].T
            bb = beta[bi, sl].T
            lower = torch.tril(
                (kk @ kk.transpose(-1, -2))
                * torch.exp(gg[:, :, None] - gg[:, None, :])
                * bb[:, :, None],
                diagonal=-1,
            )
            eye = torch.eye(n, dtype=torch.float64).expand(h, n, n)
            inv = torch.linalg.solve_triangular(
                eye + lower, eye, upper=False, unitriangular=True
            )
            out[bi, sl, :, :n] = inv.transpose(0, 1)
        offset += length
    return out


def launch(mod, x, a):
    b, t, hg, kd = x["k"].shape
    h = x["beta"].shape[-1]
    idx = x["chunk_indices"]
    nt = len(idx) if idx is not None else triton.cdiv(t, 64)
    mod.chunk_gated_delta_rule_fwd_kkt_solve_kernel[(nt, b * h)](
        k=x["k"],
        g=x["g"],
        beta=x["beta"],
        A=a,
        cu_seqlens=x["cu_seqlens"],
        chunk_indices=idx,
        T=t,
        H=h,
        Hg=hg,
        K=kd,
        BT=64,
        BC=16,
    )
    return a


def separated(x):
    a = chunk_scaled_dot_kkt_fwd(
        x["k"], x["beta"], x["g"], x["cu_seqlens"], output_dtype=torch.float32
    )
    return solve_tril(a, x["cu_seqlens"], output_dtype=x["k"].dtype)


def correctness():
    reports = []
    for dtype in [torch.bfloat16, torch.float16]:
        atol, rtol = (0.004, 0.02) if dtype == torch.bfloat16 else (0.001, 0.004)
        for packed, lengths in [
            (False, [n, n]) for n in [1, 15, 16, 17, 63, 64, 65, 127, 128, 129]
        ] + [(True, [1, 15, 16, 17, 63, 64, 65, 127, 128, 129])]:
            x, _ = inputs(lengths, dtype, packed=packed)
            wb, ub, ab = baseline.chunk_gated_delta_rule_fwd_intra(**x)
            wc, uc, ac = candidate.chunk_gated_delta_rule_fwd_intra(**x)
            ref = inverse_reference(x, lengths).to(device="cuda", dtype=dtype)
            torch.testing.assert_close(ac, ab, atol=atol, rtol=rtol)
            torch.testing.assert_close(wc, wb, atol=atol, rtol=rtol)
            torch.testing.assert_close(uc, ub, atol=atol, rtol=rtol)
            torch.testing.assert_close(ac, ref, atol=atol, rtol=rtol)
            # Poison every output cell, then invoke the kernel directly.
            # This catches missing upper-block stores and packed-tail writes.
            poison = torch.full_like(ac, float("nan"))
            launch(candidate, x, poison)
            assert torch.isfinite(poison).all().item()
            torch.testing.assert_close(poison, ac, atol=0, rtol=0)
            zero_mask = ref == 0
            assert torch.count_nonzero(poison[zero_mask]).item() == 0
            reports.append(
                {
                    "dtype": str(dtype),
                    "packed": packed,
                    "lengths": lengths,
                    "max_inverse_abs_error": (ac.float() - ref.float())
                    .abs()
                    .max()
                    .item(),
                    "max_baseline_abs_error": (ac.float() - ab.float())
                    .abs()
                    .max()
                    .item(),
                }
            )
            print(json.dumps({"correctness_case": reports[-1]}), flush=True)
        # Actual TP=2 GDN head geometry, including a sequence boundary tail.
        x, _ = inputs([65, 129], dtype, hg=8, h=24)
        b = baseline.chunk_gated_delta_rule_fwd_intra(**x)
        c = candidate.chunk_gated_delta_rule_fwd_intra(**x)
        for actual, expected in zip(c, b):
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        # Legacy separated path: compare on aligned input (its tail behavior
        # is not used as independent correctness authority).
        x, _ = inputs([128], dtype, hg=8, h=24)
        torch.testing.assert_close(
            candidate.chunk_gated_delta_rule_fwd_intra(**x)[2],
            separated(x),
            atol=atol,
            rtol=rtol,
        )
    return reports


def full_gdr_correctness():
    mod = importlib.import_module("sglang.srt.layers.attention.fla.chunk")
    reports = []
    for dtype in [torch.bfloat16, torch.float16]:
        x, raw_g = inputs([65, 129], dtype, hg=8, h=24)
        q = torch.randn_like(x["k"])
        normalized = []
        for tensor in [q, x["k"]]:
            actual_norm = l2norm_fwd(tensor)
            expected_norm = (
                tensor.float()
                / torch.sqrt(tensor.float().square().sum(-1, keepdim=True) + 1e-6)
            ).to(dtype)
            assert torch.isfinite(actual_norm).all().item()
            torch.testing.assert_close(
                actual_norm,
                expected_norm,
                atol=0.002 if dtype == torch.bfloat16 else 0.0003,
                rtol=0.01 if dtype == torch.bfloat16 else 0.002,
            )
            normalized.append(actual_norm.double().cpu().repeat_interleave(3, dim=2))
        state = torch.randn(4, 24, 128, 128, device="cuda", dtype=torch.float32) * 0.1
        indices = torch.tensor([2, 0], device="cuda", dtype=torch.int32)
        saved = mod.chunk_gated_delta_rule_fwd_intra
        results = []
        try:
            for impl in [baseline, candidate]:
                mod.chunk_gated_delta_rule_fwd_intra = (
                    impl.chunk_gated_delta_rule_fwd_intra
                )
                pool = state.clone()
                output = mod.chunk_gated_delta_rule(
                    q,
                    x["k"],
                    x["v"],
                    raw_g,
                    x["beta"],
                    initial_state=pool,
                    initial_state_indices=indices,
                    cu_seqlens=x["cu_seqlens"],
                    use_qk_l2norm_in_kernel=True,
                )
                results.append((output[0], pool, output[2]))
                torch.testing.assert_close(pool[[1, 3]], state[[1, 3]], atol=0, rtol=0)
        finally:
            mod.chunk_gated_delta_rule_fwd_intra = saved
        for actual, expected in zip(results[1], results[0]):
            assert torch.isfinite(actual).all().item()
            torch.testing.assert_close(
                actual,
                expected,
                atol=0.01 if dtype == torch.bfloat16 else 0.002,
                rtol=0.03 if dtype == torch.bfloat16 else 0.006,
            )
        # Independent sequential delta-rule reference, using the checked
        # rounded Q,K that the chunk implementation actually consumes.
        ref_pool = state.double().cpu()
        ref_out = torch.empty_like(x["v"], device="cpu", dtype=torch.float64)
        vv, gg, bb = (t.double().cpu() for t in [x["v"], raw_g, x["beta"]])
        pos = 0
        for length, slot in zip([65, 129], [2, 0]):
            hh = ref_pool[slot].clone()
            for t in range(pos, pos + length):
                kk, qq = normalized[1][0, t], normalized[0][0, t]
                hh *= gg[0, t].exp()[:, None, None]
                residual = (vv[0, t] - (hh * kk[:, None, :]).sum(-1)) * bb[
                    0, t, :, None
                ]
                hh += residual[:, :, None] * kk[:, None, :]
                ref_out[0, t] = (hh * qq[:, None, :]).sum(-1) / (128**0.5)
            ref_pool[slot] = hh
            pos += length
        ref_atol, ref_rtol = (0.02, 0.05) if dtype == torch.bfloat16 else (0.004, 0.01)
        for output, pool, _ in results:
            torch.testing.assert_close(
                output.double().cpu(), ref_out, atol=ref_atol, rtol=ref_rtol
            )
            torch.testing.assert_close(
                pool.double().cpu(), ref_pool, atol=ref_atol, rtol=ref_rtol
            )
        reports.append(
            {
                "dtype": str(dtype),
                "l2norm_reference_checked": True,
                "reference_output_max_abs_error": (
                    results[1][0].double().cpu() - ref_out
                )
                .abs()
                .max()
                .item(),
                "reference_state_max_abs_error": (
                    results[1][1].double().cpu() - ref_pool
                )
                .abs()
                .max()
                .item(),
                "output_max_abs_diff": (results[1][0].float() - results[0][0].float())
                .abs()
                .max()
                .item(),
                "state_max_abs_diff": (results[1][1] - results[0][1])
                .abs()
                .max()
                .item(),
            }
        )
    return reports


def elapsed(fn, repeats):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / repeats


def paired(functions, rounds, repeats):
    for fn in functions.values():
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    samples = {name: [] for name in functions}
    names = list(functions)
    for round_id in range(rounds):
        for name in (names if round_id % 2 == 0 else names[::-1]):
            samples[name].append(elapsed(functions[name], repeats))
    return {
        name: {
            "median_us": statistics.median(values),
            "min_us": min(values),
            "max_us": max(values),
            "samples_us": values,
        }
        for name, values in samples.items()
    }


def graph_functions(functions, calls=4):
    """Capture several calls per replay so host launch gaps do not dominate."""
    graphs = {}
    for name, fn in functions.items():
        # Warm each function (including autotune/metadata) before capture.
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(calls):
                fn()
        graphs[name] = graph.replay
    return graphs


def benchmark(rounds, repeats):
    rows = []
    for lengths in [[64], [256], [4096], [4096] * 8, [65536]]:
        x, _ = inputs(lengths, torch.bfloat16, hg=8, h=24)
        shape = (*x["beta"].shape, 64)
        ab = torch.zeros(shape, device="cuda", dtype=x["k"].dtype)
        ac = torch.empty_like(ab)

        def allocated(mod, zero):
            a = (
                torch.zeros(shape, device="cuda", dtype=ab.dtype)
                if zero
                else torch.empty(shape, device="cuda", dtype=ab.dtype)
            )
            return launch(mod, x, a)

        stages = {
            "kernel_only": {
                "baseline": lambda: launch(baseline, x, ab),
                "candidate": lambda: launch(candidate, x, ac),
            },
            "kkt_solve_with_allocation": {
                "baseline": lambda: allocated(baseline, True),
                "candidate": lambda: allocated(candidate, False),
                "separated": lambda: separated(x),
            },
            "intra_with_wu": {
                "baseline": lambda: baseline.chunk_gated_delta_rule_fwd_intra(**x),
                "candidate": lambda: candidate.chunk_gated_delta_rule_fwd_intra(**x),
            },
        }
        for stage, fns in stages.items():
            for mode in ["eager", "graph"]:
                measured = fns if mode == "eager" else graph_functions(fns)
                result = paired(measured, rounds, repeats)
                if mode == "graph":
                    for values in result.values():
                        for key in ["median_us", "min_us", "max_us"]:
                            values[key] /= 4
                        values["samples_us"] = [v / 4 for v in values["samples_us"]]
                row = {
                    "lengths": lengths,
                    "dtype": "bfloat16",
                    "key_heads": 8,
                    "value_heads": 24,
                    "head_dim": 128,
                    "stage": stage,
                    "mode": mode,
                    "timing": result,
                    "speedup": result["baseline"]["median_us"]
                    / result["candidate"]["median_us"],
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/tmp/gdn-validation.json")
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--correctness-only", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(20261014)
    torch.set_num_threads(4)
    props = torch.cuda.get_device_properties(0)
    report = {
        "device": str(props),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "triton": triton.__version__,
        "installed": json.loads(Path(__file__).with_name("installed.json").read_text()),
        "seed": 20261014,
        "rounds": args.rounds,
        "repeats": args.repeats,
    }
    print(json.dumps(report), flush=True)
    started = time.monotonic()
    report["correctness"] = correctness()
    report["full_gdr_correctness"] = full_gdr_correctness()
    report["correctness_seconds_including_compilation"] = time.monotonic() - started
    print("correctness passed", flush=True)
    report["benchmarks"] = (
        [] if args.correctness_only else benchmark(args.rounds, args.repeats)
    )
    report["model_end_to_end_tested"] = False
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print("saved " + args.output, flush=True)


if __name__ == "__main__":
    main()
