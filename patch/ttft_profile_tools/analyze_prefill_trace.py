#!/usr/bin/env python3
"""CPU-only analysis of an SGLang torch-profiler trace (ROCm/DTK, activity GPU).

Usage:  python3 -I analyze_prefill_trace.py TRACE.json.gz [--step SUBSTR] [--json OUT]

Selects one GPU step window (gpu_user_annotation 'step[...]', default the EXTEND
step with the most tokens), then reports:
  * device time by category / layer role (GEMM roles inferred from Tensile grid)
  * top kernels
  * device idle gaps (union of all streams) and the CPU ops launching the kernel
    right after each large gap
  * RCCL vs compute overlap (exposed comm = comm time with no compute kernel active)
  * per-layer-type timeline (GDN vs full attention) on the compute stream
"""
import argparse
import bisect
import collections
import gzip
import json
import re
import sys

HIDDEN = 5120


def load(path):
    with gzip.open(path, "rt") if path.endswith(".gz") else open(path) as f:
        return json.load(f)


def gemm_role(name, grid, M_hint):
    """Infer (role, MxNxK) for Tensile INT8/BF16 GEMMs from tile 256x256 grids."""
    g = grid or [0, 0, 0]
    if "I8BS" in name:  # hipBLASLt (whitelisted shapes), grid x = Ntiles*768
        nt, mt = g[0] // 768, g[1]
        n, m = nt * 256, mt * 256
        if n == 17408:
            return "INT8 GEMM MLP gate_up (hipBLASLt)", f"{m}x17408x5120"
        if n == 5120:
            return "INT8 GEMM attn/GDN out_proj (hipBLASLt)", f"{m}x5120x3072"
        if n == 8192:
            return "INT8 GEMM GDN in_proj_qkvz (hipBLASLt)", f"{m}x8192x5120"
        return "INT8 GEMM other (hipBLASLt)", f"{m}x{n}x?"
    if "I8II" in name:  # lightop ASM-like Tensile, grid = (Ntiles, Mtiles)
        n, m = g[0] * 256, g[1] * 256
        if n == 5120:
            return "INT8 GEMM MLP down (lightop)", f"{m}x5120x8704"
        if n == 8192:
            return "INT8 GEMM GDN in_proj_qkvz (lightop)", f"{m}x8192x5120"
        if n == 7168:
            return "INT8 GEMM FA qkv_proj (lightop)", f"{m}x7168x5120"
        if n == 17408:
            return "INT8 GEMM MLP gate_up (lightop)", f"{m}x17408x5120"
        return "INT8 GEMM other (lightop)", f"{m}x{n}x?"
    if "BBH" in name:
        if "MT64x64x64" in name:
            return "BF16 GEMM GDN in_proj_ba", "Mx48x5120"
        return "BF16 GEMM other (lm_head/misc)", "?"
    return None, None


def category(e):
    n = e["name"]
    g = e.get("args", {}).get("grid")
    if e.get("cat") == "gpu_memcpy":
        return "memcpy", None
    if e.get("cat") == "gpu_memset":
        return "memset", None
    if n.startswith("ncclDevKernel") or "rccl" in n.lower():
        return "RCCL", None
    if n.startswith("Cijk_") or n == "matmul_kernel":
        r, shape = gemm_role(n, g, None)
        if r:
            return r, shape
        return "GEMM other", None
    gdn = {
        "chunk_gated_delta_rule_fwd_kernel_h_blockdim64": "GDN chunk_h (state recurrence)",
        "chunk_gated_delta_rule_fwd_kkt_solve_kernel": "GDN kkt+solve",
        "recompute_w_u_fwd_kernel": "GDN recompute_w_u",
        "chunk_fwd_kernel_o": "GDN chunk_o",
        "chunk_local_cumsum_scalar_kernel": "GDN cumsum",
        "_causal_conv1d_fwd_kernel": "GDN causal_conv1d",
        "fused_gdn_prefill_prepare_kernel": "GDN prepare (split/l2norm/gating)",
        "_layer_norm_fwd_1pass_kernel": "GDN gated RMSNorm (out norm)",
    }
    if n in gdn:
        return gdn[n], None
    if n.startswith("fused_recurrent") or n.startswith("_causal_conv1d_update"):
        return "GDN decode", None
    if "flash_fwd" in n:
        return "FullAttn FlashAttention fwd", None
    if n in ("_pack",):
        return "FullAttn KV pack", None
    if n in ("_gate",):
        return "FullAttn output gate", None
    if n == "_fused_qk_gemma_rmsnorm_gate_kernel":
        return "FullAttn QK-norm+gate split", None
    if "mrope" in n or "rotary" in n:
        return "FullAttn mRoPE", None
    if "paged_attention" in n:
        return "FullAttn decode", None
    if "fused_add_rms" in n:
        rows = (g or [0])[0]
        if rows >= 65536:
            return "Norm: fused_add_rms full-M (layer input norm)", None
        return "Norm: fused_add_rms MLP sub-batch (post-attn norm)", None
    if "fused_rms" in n:
        return "Norm: final rms", None
    if n == "_per_token_quant_int8":
        rows = (g or [0])[0]
        if rows >= 65536:
            return "Quant: per-token INT8 full-M (attn in/out)", None
        return "Quant: per-token INT8 MLP sub-batch", None
    if "silu_and_mul" in n:
        return "Act: SiLU*mul", None
    if "CatArrayBatchedCopy" in n:
        gy = (g or [0, 0])[1] if g and len(g) > 1 else 0
        if gy == 8:
            return "Copy: torch.cat of 8 MLP sub-batches", None
        return "Copy: cat other", None
    if "index_elementwise" in n or "index_put" in n:
        return "Index/scatter (KV store etc.)", None
    if "FillFunctor" in n:
        return "Fill/zero", None
    if n.startswith("void rocprim") or "radix_sort" in n:
        return "Sampling/sort misc", None
    return "Other elementwise/misc", None


def union_intervals(iv):
    iv = sorted(iv)
    out = []
    for s, e in iv:
        if out and s <= out[-1][1]:
            if e > out[-1][1]:
                out[-1][1] = e
        else:
            out.append([s, e])
    return out


def intersect_len(a, b):
    """a, b: sorted disjoint interval lists. returns total overlap length."""
    i = j = 0
    tot = 0.0
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0])
        e = min(a[i][1], b[j][1])
        if e > s:
            tot += e - s
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--step", default=None, help="substring of step annotation")
    ap.add_argument("--nth", type=int, default=None,
                    help="pick the n-th EXTEND step in GPU time order (0-based)")
    ap.add_argument("--json", default=None)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--gaps", type=int, default=15)
    a = ap.parse_args()

    d = load(a.trace)
    ev = d["traceEvents"]
    xs = [e for e in ev if e.get("ph") == "X"]
    t0 = min(e["ts"] for e in xs)

    steps = [e for e in xs if e.get("cat") == "gpu_user_annotation" and e["name"].startswith("step[")]
    if not steps:
        print("no gpu step annotations", file=sys.stderr)
        sys.exit(1)

    def toks(e):
        m = re.search(r"toks=(\d+)", e["name"])
        return int(m.group(1)) if m else 0

    if a.nth is not None:
        step = sorted((e for e in steps if "EXTEND" in e["name"]), key=lambda e: e["ts"])[a.nth]
    elif a.step:
        step = [e for e in steps if a.step in e["name"]][0]
    else:
        step = max((e for e in steps if "EXTEND" in e["name"]), key=toks)
    w0, w1 = step["ts"], step["ts"] + step["dur"]

    dev = [e for e in xs if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and w0 <= e["ts"] < w1]
    dev.sort(key=lambda e: e["ts"])
    win_ms = (w1 - w0) / 1e3

    # correlation -> runtime launch -> enclosing CPU ops
    rt = {}
    for e in xs:
        if e.get("cat") == "cuda_runtime":
            c = e.get("args", {}).get("correlation")
            if c is not None:
                rt[c] = e
    cpu_ops = sorted((e for e in xs if e.get("cat") in ("cpu_op", "user_annotation")), key=lambda e: e["ts"])
    cpu_by_tid = collections.defaultdict(list)
    for e in cpu_ops:
        cpu_by_tid[e["tid"]].append(e)
    cpu_ts = {t: [e["ts"] for e in L] for t, L in cpu_by_tid.items()}

    def enclosing(r, depth=4):
        L = cpu_by_tid.get(r["tid"], [])
        T = cpu_ts.get(r["tid"], [])
        i = bisect.bisect_right(T, r["ts"])
        out = []
        k = i - 1
        scanned = 0
        while k >= 0 and scanned < 4000 and len(out) < depth:
            e = L[k]
            if e["ts"] <= r["ts"] and e["ts"] + e["dur"] >= r["ts"] + r["dur"]:
                out.append(e["name"])
            k -= 1
            scanned += 1
        return out  # innermost first

    # ---- categories
    cat = collections.OrderedDict()
    kern = collections.defaultdict(lambda: [0, 0.0, None])
    for e in dev:
        c, shape = category(e)
        e["_cat"] = c
        e["_shape"] = shape
        v = cat.setdefault(c, [0, 0.0, set()])
        v[0] += 1
        v[1] += e["dur"]
        if shape:
            v[2].add(shape)
        key = (e["name"][:110], str(e.get("args", {}).get("grid")))
        kern[key][0] += 1
        kern[key][1] += e["dur"]
        kern[key][2] = c
    total_dev = sum(v[1] for v in cat.values())

    streams = collections.Counter(e["tid"] for e in dev)
    comm = [e for e in dev if e["_cat"] == "RCCL"]
    comp = [e for e in dev if e["_cat"] != "RCCL"]
    comm_iv = union_intervals([[e["ts"], e["ts"] + e["dur"]] for e in comm])
    comp_iv = union_intervals([[e["ts"], e["ts"] + e["dur"]] for e in comp])
    all_iv = union_intervals([[e["ts"], e["ts"] + e["dur"]] for e in dev])
    comm_t = sum(e - s for s, e in comm_iv)
    comp_t = sum(e - s for s, e in comp_iv)
    busy = sum(e - s for s, e in all_iv)
    ov = intersect_len(comm_iv, comp_iv)

    # gaps (device fully idle) within window
    gaps = []
    prev = w0
    for s, e in all_iv:
        if s > prev:
            gaps.append((s - prev, prev, s))
        prev = max(prev, e)
    if w1 > prev:
        gaps.append((w1 - prev, prev, w1))
    gaps.sort(reverse=True)
    # find kernel starting at end of each gap
    starts = [e["ts"] for e in dev]

    def gap_info(g):
        i = bisect.bisect_left(starts, g[2] - 0.01)
        nxt = dev[i] if i < len(dev) else None
        prv = None
        # previous ending kernel
        for k in range(i - 1, max(-1, i - 50), -1):
            if dev[k]["ts"] + dev[k]["dur"] <= g[1] + 0.01:
                prv = dev[k]
                break
        info = {"gap_us": round(g[0], 1), "at_ms": round((g[1] - w0) / 1e3, 2)}
        if prv:
            info["prev_kernel"] = prv["name"][:70]
        if nxt:
            info["next_kernel"] = nxt["name"][:70]
            r = rt.get(nxt.get("args", {}).get("correlation"))
            if r:
                info["next_launch_api"] = r["name"]
                info["next_launch_cpu_ops"] = enclosing(r)
                info["launch_to_start_us"] = round(nxt["ts"] - r["ts"], 1)
        return info

    gap_hist = collections.Counter()
    for g in gaps:
        b = "<5us" if g[0] < 5 else "5-20us" if g[0] < 20 else "20-100us" if g[0] < 100 else "0.1-1ms" if g[0] < 1000 else ">=1ms"
        gap_hist[b] += 1
    gap_sum = collections.Counter()
    for g in gaps:
        b = "<5us" if g[0] < 5 else "5-20us" if g[0] < 20 else "20-100us" if g[0] < 100 else "0.1-1ms" if g[0] < 1000 else ">=1ms"
        gap_sum[b] += g[0]

    # per comm kernel: exposed portion
    comp_iv_sorted = comp_iv
    comp_starts = [s for s, _ in comp_iv_sorted]
    exposed_by_name = collections.defaultdict(float)
    for e in comm:
        s, t = e["ts"], e["ts"] + e["dur"]
        i = max(0, bisect.bisect_right(comp_starts, s) - 1)
        cov = 0.0
        while i < len(comp_iv_sorted) and comp_iv_sorted[i][0] < t:
            a0, a1 = comp_iv_sorted[i]
            cov += max(0.0, min(a1, t) - max(a0, s))
            i += 1
        exposed_by_name[e["name"][:45]] += (t - s) - cov

    # what compute runs while comm runs (slowdown of overlapped kernels)
    def overlap_frac(e):
        s, t = e["ts"], e["ts"] + e["dur"]
        i = max(0, bisect.bisect_right([x[0] for x in comm_iv], s) - 1)
        cov = 0.0
        while i < len(comm_iv) and comm_iv[i][0] < t:
            a0, a1 = comm_iv[i]
            cov += max(0.0, min(a1, t) - max(a0, s))
            i += 1
        return cov / max(e["dur"], 1e-9)

    contention = collections.defaultdict(lambda: {"with": [], "without": []})
    for e in comp:
        if e["dur"] < 50:
            continue
        key = (e["_cat"], e["name"][:50], str(e.get("args", {}).get("grid")))
        f = overlap_frac(e)
        if f > 0.8:
            contention[key]["with"].append(e["dur"])
        elif f < 0.05:
            contention[key]["without"].append(e["dur"])

    # ---- report
    out = {
        "trace": a.trace,
        "step": step["name"],
        "window_ms": round(win_ms, 2),
        "streams": dict(streams),
        "device_event_sum_ms": round(total_dev / 1e3, 2),
        "busy_union_ms": round(busy / 1e3, 2),
        "idle_ms": round(win_ms - busy / 1e3, 2),
        "compute_union_ms": round(comp_t / 1e3, 2),
        "comm_union_ms": round(comm_t / 1e3, 2),
        "comm_overlapped_ms": round(ov / 1e3, 2),
        "comm_exposed_ms": round((comm_t - ov) / 1e3, 2),
        "categories": [],
        "top_kernels": [],
        "gap_hist": {k: [gap_hist[k], round(gap_sum[k] / 1e3, 2)] for k in ["<5us", "5-20us", "20-100us", "0.1-1ms", ">=1ms"]},
        "largest_gaps": [gap_info(g) for g in gaps[: a.gaps]],
        "exposed_comm_by_kernel_ms": {k: round(v / 1e3, 2) for k, v in exposed_by_name.items()},
        "contention": [],
    }
    for c, (n, t, shapes) in sorted(cat.items(), key=lambda x: -x[1][1]):
        out["categories"].append({
            "category": c, "count": n, "total_ms": round(t / 1e3, 2),
            "pct_of_window": round(100 * t / 1e3 / win_ms, 2),
            "avg_us": round(t / n, 1), "shapes": sorted(shapes)})
    for (n, g), (c, t, cg) in sorted(kern.items(), key=lambda x: -x[1][1])[: a.top]:
        out["top_kernels"].append({"name": n, "grid": g, "category": cg, "count": c,
                                   "total_ms": round(t / 1e3, 2), "avg_us": round(t / c, 1),
                                   "pct_of_window": round(100 * t / 1e3 / win_ms, 2)})
    for key, v in contention.items():
        if v["with"] and v["without"]:
            w = sorted(v["with"])[len(v["with"]) // 2]
            wo = sorted(v["without"])[len(v["without"]) // 2]
            out["contention"].append({"category": key[0], "kernel": key[1], "grid": key[2],
                                      "n_with_comm": len(v["with"]), "median_us_with_comm": round(w, 1),
                                      "n_without_comm": len(v["without"]), "median_us_without_comm": round(wo, 1),
                                      "slowdown_pct": round(100 * (w / wo - 1), 1)})
        elif v["with"]:
            out["contention"].append({"category": key[0], "kernel": key[1], "grid": key[2],
                                      "n_with_comm": len(v["with"]),
                                      "median_us_with_comm": round(sorted(v["with"])[len(v["with"]) // 2], 1),
                                      "n_without_comm": 0})
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
