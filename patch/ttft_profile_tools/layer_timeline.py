#!/usr/bin/env python3
"""Per-layer timeline of a prefill step (CPU-only trace analysis).

Usage: python3 -I layer_timeline.py TRACE.json.gz [--dump-layer N] [--json OUT]

Layers are delimited by the full-M fused_add_rms kernel (layer input norm).
For each layer reports: type (GDN/FA), wall time, compute-stream busy time,
exposed-comm time split into head (before first MLP sub-batch norm), mid, and
tail (after the last down GEMM), and per-phase timings.
Also fits, per kernel class, duration ~ base*(1+alpha*overlap_with_comm) to
estimate the CU/bandwidth "contention tax" paid by kernels overlapped with RCCL.
"""
import argparse
import bisect
import collections
import gzip
import json
import re
import statistics


def load(p):
    with gzip.open(p, "rt") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--dump-layer", type=int, nargs="*", default=[])
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    d = load(a.trace)
    xs = [e for e in d["traceEvents"] if e.get("ph") == "X"]
    steps = [e for e in xs if e.get("cat") == "gpu_user_annotation" and e["name"].startswith("step[")]
    step = max((e for e in steps if "EXTEND" in e["name"]),
               key=lambda e: int(re.search(r"toks=(\d+)", e["name"]).group(1)))
    w0, w1 = step["ts"], step["ts"] + step["dur"]
    dev = sorted((e for e in xs if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and w0 <= e["ts"] < w1),
                 key=lambda e: e["ts"])
    comm = [e for e in dev if e["name"].startswith("ncclDevKernel")]
    comp = [e for e in dev if not e["name"].startswith("ncclDevKernel")]
    starts = [e for e in comp if "fused_add_rms" in e["name"] and e["args"]["grid"][0] >= 65536]
    bounds = [e["ts"] for e in starts] + [w1]

    def short(n):
        if n.startswith("Cijk"):
            return n[:40]
        return re.sub(r"<.*", "", n.replace("void ", ""))[:48]

    comm_iv = [(e["ts"], e["ts"] + e["dur"]) for e in comm]
    comm_iv.sort()
    cstarts = [s for s, _ in comm_iv]

    def comm_cov(s, t):
        i = max(0, bisect.bisect_right(cstarts, s) - 1)
        cov = 0.0
        while i < len(comm_iv) and comm_iv[i][0] < t:
            cov += max(0.0, min(comm_iv[i][1], t) - max(comm_iv[i][0], s))
            i += 1
        return cov

    layers = []
    for li in range(len(starts)):
        s, t = bounds[li], bounds[li + 1]
        ks = [e for e in comp if s <= e["ts"] < t]
        cs = [e for e in comm if s <= e["ts"] < t]
        names = [e["name"] for e in ks]
        typ = "FA" if any("flash_fwd" in n for n in names) else "GDN" if any("chunk_fwd_kernel_o" in n for n in names) else "?"
        mlp_norm = [e for e in ks if "fused_add_rms" in e["name"] and e["args"]["grid"][0] < 65536]
        down = [e for e in ks if "I8II" in e["name"] and e["args"]["grid"][0] == 20]
        cat = [e for e in ks if "CatArrayBatchedCopy" in e["name"] and e["args"]["grid"][1] == 8]
        # compute-stream idle intervals
        idle = []
        prev = s
        for e in ks:
            if e["ts"] > prev:
                idle.append((prev, e["ts"]))
            prev = max(prev, e["ts"] + e["dur"])
        if t > prev:
            idle.append((prev, t))
        head = mid = tail = other = 0.0
        first_norm = mlp_norm[0]["ts"] if mlp_norm else t
        last_down_end = down[-1]["ts"] + down[-1]["dur"] if down else s
        for i0, i1 in idle:
            c = comm_cov(i0, i1)
            if i1 <= first_norm + 0.01 and i0 >= (mlp_norm[0]["ts"] - 50000 if mlp_norm else s):
                head += c
            elif i0 >= last_down_end - 0.01:
                tail += c
            elif mlp_norm and i0 >= first_norm:
                mid += c
            else:
                other += c
        attn_end = first_norm
        rec = {
            "layer": li, "type": typ, "wall_ms": (t - s) / 1e3,
            "attn_phase_ms": (attn_end - s) / 1e3,
            "mlp_phase_ms": (last_down_end - first_norm) / 1e3 if mlp_norm and down else 0,
            "tail_phase_ms": (t - last_down_end) / 1e3 if down else 0,
            "exposed_comm_head_ms": head / 1e3, "exposed_comm_mid_ms": mid / 1e3,
            "exposed_comm_tail_ms": tail / 1e3, "exposed_comm_other_ms": other / 1e3,
            "idle_total_ms": sum(i1 - i0 for i0, i1 in idle) / 1e3,
            "cat_ms": sum(e["dur"] for e in cat) / 1e3,
            "n_comm": len(cs),
        }
        layers.append(rec)
        if li in a.dump_layer:
            print(f"=== layer {li} {typ} wall {rec['wall_ms']:.2f} ms")
            evs = sorted(ks + cs, key=lambda e: e["ts"])
            for e in evs:
                tag = "  COMM" if e in cs else "      "
                print(f"{tag} {(e['ts'] - s) / 1e3:8.3f} +{e['dur'] / 1e3:7.3f}  {short(e['name'])} {e['args'].get('grid')}")

    def agg(typ):
        L = [r for r in layers if r["type"] == typ]
        keys = [k for k in L[0] if k.endswith("_ms")]
        return {"n": len(L), **{k: round(statistics.mean(r[k] for r in L), 3) for k in keys},
                **{"sum_" + k: round(sum(r[k] for r in L), 2) for k in keys}}

    summary = {"step": step["name"], "n_layers": len(layers), "pre_first_layer_ms": round((starts[0]["ts"] - w0) / 1e3, 2),
               "GDN": agg("GDN"), "FA": agg("FA")}

    # contention fit: dur = b0 + b1*frac
    cls = collections.defaultdict(list)
    for e in comp:
        if e["dur"] < 100:
            continue
        key = (short(e["name"]), str(e["args"].get("grid")))
        f = comm_cov(e["ts"], e["ts"] + e["dur"]) / e["dur"]
        cls[key].append((f, e["dur"]))
    fits = []
    for key, pts in cls.items():
        if len(pts) < 16:
            continue
        fs = [p[0] for p in pts]
        ds = [p[1] for p in pts]
        mf, md = statistics.mean(fs), statistics.mean(ds)
        var = sum((f - mf) ** 2 for f in fs)
        lo = [dd for f, dd in pts if f < 0.05]
        hi = [dd for f, dd in pts if f > 0.95]
        if var < 1e-6:
            b1 = 0.0
            b0 = md
        else:
            b1 = sum((f - mf) * (dd - md) for f, dd in pts) / var
            b0 = md - b1 * mf
        tax = sum(dd for _, dd in pts) - b0 * len(pts)
        fits.append({"kernel": key[0], "grid": key[1], "n": len(pts), "mean_overlap_frac": round(mf, 3),
                     "fit_base_us": round(b0, 1), "fit_full_overlap_us": round(b0 + b1, 1),
                     "median_no_comm_us": round(statistics.median(lo), 1) if lo else None, "n_no_comm": len(lo),
                     "median_full_comm_us": round(statistics.median(hi), 1) if hi else None, "n_full_comm": len(hi),
                     "est_tax_ms": round(tax / 1e3, 2)})
    fits.sort(key=lambda r: -r["est_tax_ms"])
    summary["contention_fits"] = fits[:12]
    summary["contention_tax_total_ms"] = round(sum(max(0, r["est_tax_ms"]) for r in fits), 1)
    print(json.dumps(summary, indent=1))
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"summary": summary, "layers": layers}, f, indent=1)


if __name__ == "__main__":
    main()
