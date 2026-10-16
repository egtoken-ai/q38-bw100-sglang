"""BW100 mRoPE tiled-kernel sweep with bitwise exactness gate.

Single process, one GPU (HIP_VISIBLE_DEVICES selects it). Nothing installed is
modified: the reference is the audited vendor ``triton_kernels.py`` (or its
``.py.ttft_mrope_original`` backup on an installed image) imported from a temp
copy; the candidate is ``install.transform`` of that text plus the patch-dir
helper. ``--installed`` tests the real installed sglang module instead.

Q and K (whole tensors, including the untouched non-rotary dims and any
surrounding buffer for strided views) must be bitwise equal to the vendor
kernel on every case before a config is timed/ranked.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
SGLANG_SRT = Path("/usr/local/lib/python3.10/dist-packages/sglang/srt")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


INSTALL = load_module("ttft_mrope_install", HERE / "install.py")


def model_dims(config_path):
    dims = dict(n_qh=12, n_kh=2, head=256, rotary=64, section=[11, 11, 10], interleaved=True,
                theta=10000000.0, max_pos=262144, source="builtin (Qwen3.8-27B TP2)")
    try:
        cfg = json.loads(Path(config_path).read_text())
        tc = cfg.get("text_config", cfg)
        rp = tc.get("rope_parameters") or tc.get("rope_scaling") or {}
        tp = 2
        head = tc["head_dim"]
        prf = rp.get("partial_rotary_factor", tc.get("partial_rotary_factor", 1.0))
        dims.update(n_qh=tc["num_attention_heads"] // tp, n_kh=max(tc["num_key_value_heads"] // tp, 1),
                    head=head, rotary=int(head * prf), section=list(rp["mrope_section"]),
                    interleaved=bool(rp.get("mrope_interleaved", False)),
                    theta=float(rp.get("rope_theta", tc.get("rope_theta", 10000))),
                    max_pos=tc["max_position_embeddings"], source=str(config_path))
    except (OSError, KeyError, ValueError) as exc:
        print(f"[warn] model config unavailable ({exc}); using builtin dims", flush=True)
    return dims


def cos_sin_cache(dims, device):
    # Same construction as RotaryEmbedding._compute_cos_sin_cache, then the
    # MRotaryEmbedding._match_cos_sin_cache_dtype cast to the query dtype.
    rd = dims["rotary"]
    inv_freq = 1.0 / (dims["theta"] ** (torch.arange(0, rd, 2, dtype=torch.float) / rd))
    t = torch.arange(dims["max_pos"], dtype=torch.float)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(device=device, dtype=torch.bfloat16)


def bitwise_equal(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(
        a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))


def text_positions(lens, offsets, device):
    rows = [torch.arange(o, o + n, dtype=torch.int64) for n, o in zip(lens, offsets)]
    pos = torch.cat(rows)
    return pos.unsqueeze(0).expand(3, -1).contiguous().to(device)


def build_cases(dims, device, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    qd, kd = dims["n_qh"] * dims["head"], dims["n_kh"] * dims["head"]

    def tensors(ntok, layout):
        if layout == "contig":   # production: QK-norm/gate path writes contiguous q/k
            q = (torch.randn(ntok, qd, generator=g) * 3).to(torch.bfloat16).to(device)
            k = (torch.randn(ntok, kd, generator=g) * 3).to(torch.bfloat16).to(device)
            return dict(q=q, k=k, qbuf=q, kbuf=k)
        width = 2 * qd + 2 * kd  # fallback-path-like split views of one qkv buffer
        buf = (torch.randn(ntok, width, generator=g) * 3).to(torch.bfloat16).to(device)
        return dict(q=buf[:, :qd], k=buf[:, 2 * qd:2 * qd + kd], qbuf=buf, kbuf=buf)

    def case(name, positions, layout="contig", timed=False):
        ntok = positions.shape[1]
        d = tensors(ntok, layout)
        d.update(name=name, positions=positions, layout=layout, timed=timed, ntok=ntok)
        return d

    rnd = torch.randint(0, dims["max_pos"], (3, 777), generator=g, dtype=torch.int64).to(device)
    return [
        case("prod_16x4096", text_positions([4096] * 16, [0] * 16, device), timed=True),
        case("prod_65536", text_positions([65536], [0], device), timed=True),
        case("bs1_4096", text_positions([4096], [0], device), timed=True),
        case("decode_16", text_positions([1] * 16, [4096 + 37 * i for i in range(16)], device)),
        case("odd_63_129_prefix", text_positions([63, 129], [100, 70000], device)),
        case("one_token", text_positions([1], [5], device)),
        case("random_thw_777", rnd),
        case("strided_63_129", text_positions([63, 129], [0, 4096], device), layout="strided"),
        case("strided_random_777", rnd.clone(), layout="strided"),
    ]


def default_grid():
    grid = []
    for bt in (4, 8, 16, 32, 64):
        for w in (1, 2, 4):
            grid.append(dict(BLOCK_T=bt, num_warps=w, PER_HEAD=False))
    grid += [dict(BLOCK_T=128, num_warps=4, PER_HEAD=False), dict(BLOCK_T=64, num_warps=8, PER_HEAD=False)]
    for bt, w in ((8, 1), (16, 1), (32, 2), (64, 4)):
        grid.append(dict(BLOCK_T=bt, num_warps=w, PER_HEAD=True))
    return grid


def key(cfg):
    return f'T{cfg["BLOCK_T"]}:w{cfg["num_warps"]}:{"perhead" if cfg["PER_HEAD"] else "allheads"}'


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model-config", default="/model/config.json")
    p.add_argument("--grid-json", type=Path)
    p.add_argument("--only-config", type=Path)
    p.add_argument("--installed", action="store_true")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--write-config", type=Path)
    args = p.parse_args()

    t_start = time.time()
    device = torch.device("cuda")
    target = SGLANG_SRT / INSTALL.TARGET_REL
    backup = target.with_name(target.name + INSTALL.BACKUP_SUFFIX)
    original_path = backup if backup.exists() else target
    original = original_path.read_text()
    sha = hashlib.sha256(original.encode()).hexdigest()
    if sha != INSTALL.ORIGINAL_SHA256:
        raise SystemExit(f"Vendor source {original_path} SHA {sha} != audited {INSTALL.ORIGINAL_SHA256}")

    tmp = Path(tempfile.mkdtemp(prefix="ttft_mrope_"))
    (tmp / "mrope_vendor_ref.py").write_text(original)
    ref_mod = load_module("ttft_mrope_vendor_ref", tmp / "mrope_vendor_ref.py")
    if args.installed:
        if not backup.exists():
            raise SystemExit("--installed requires the .py.ttft_mrope_original backup")
        from sglang.srt.layers.rotary_embedding import triton_kernels as cand_mod
        from sglang.srt.layers.rotary_embedding import mrope_tiled as helper
    else:
        helper = load_module(INSTALL.HELPER_MODULE, HERE / INSTALL.HELPER_NAME)
        helper.ENABLED = True
        (tmp / "mrope_candidate.py").write_text(INSTALL.transform(original))
        cand_mod = load_module("ttft_mrope_candidate", tmp / "mrope_candidate.py")

    dims = model_dims(args.model_config)
    if not dims["interleaved"]:
        raise SystemExit("Model mRoPE is not interleaved; tiled kernel would not be taken")
    cache = cos_sin_cache(dims, device)
    cases = build_cases(dims, device, args.seed)

    def call(fn, c, q, k):
        fn(q, k, cache, c["positions"], dims["section"], dims["head"], dims["rotary"],
           True, False, True, None)

    def fresh(c):
        qb = c["qbuf"].clone()
        kb = qb if c["kbuf"] is c["qbuf"] else c["kbuf"].clone()
        if c["layout"] == "contig":
            return qb, kb, qb, kb
        qd, kd = c["q"].shape[1], c["k"].shape[1]
        return qb, kb, qb[:, :qd], qb[:, 2 * qd:2 * qd + kd]

    refs = {}
    for c in cases:
        outs = []
        for _ in range(2):
            qb, kb, q, k = fresh(c)
            call(ref_mod.triton_mrope_fused, c, q, k)
            outs.append((qb, kb))
        if not (bitwise_equal(outs[0][0], outs[1][0]) and bitwise_equal(outs[0][1], outs[1][1])):
            raise SystemExit(f"Vendor kernel not deterministic on {c['name']}")
        refs[c["name"]] = outs[0]

    def time_case(fn, c):
        _, _, q, k = fresh(c)
        for _ in range(args.warmup):
            call(fn, c, q, k)
        torch.cuda.synchronize()
        ts = []
        for _ in range(args.iters):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            call(fn, c, q, k)
            e.record()
            e.synchronize()
            ts.append(s.elapsed_time(e))
        ts.sort()
        return ts[len(ts) // 2], ts[0]

    base_times = {c["name"]: time_case(ref_mod.triton_mrope_fused, c) for c in cases if c["timed"]}

    if args.installed:
        grid = [dict(helper.CONFIG)]
    elif args.only_config:
        grid = [INSTALL.validate_config(json.loads(args.only_config.read_text()))]
    elif args.grid_json:
        grid = [INSTALL.validate_config(c) for c in json.loads(args.grid_json.read_text())]
    else:
        grid = [INSTALL.validate_config(c) for c in default_grid()]

    results = []
    for cfg in grid:
        row = dict(config=cfg, key=key(cfg))
        if not args.installed:
            helper.CONFIG = cfg
        try:
            t0 = time.time()
            bad = []
            for c in cases:
                qb, kb, q, k = fresh(c)
                call(cand_mod.triton_mrope_fused, c, q, k)
                if not bitwise_equal(qb, refs[c["name"]][0]):
                    bad.append(f'{c["name"]}:q')
                if kb is not qb and not bitwise_equal(kb, refs[c["name"]][1]):
                    bad.append(f'{c["name"]}:k')
            row.update(exact=not bad, mismatch=bad)
            if not bad:
                row["times_ms"] = {c["name"]: time_case(cand_mod.triton_mrope_fused, c) for c in cases if c["timed"]}
                row["total_ms"] = row["times_ms"]["prod_16x4096"][0] + row["times_ms"]["prod_65536"][0]
            row["wall_s"] = round(time.time() - t0, 2)
        except Exception as exc:
            row.update(exact=False, error=f"{type(exc).__name__}: {exc}"[:400])
            torch.cuda.synchronize()
        print(json.dumps({k: row.get(k) for k in ("key", "exact", "mismatch", "total_ms", "error")}), flush=True)
        results.append(row)

    exact = sorted((r for r in results if r.get("exact")), key=lambda r: r["total_ms"])
    base_total = base_times["prod_16x4096"][0] + base_times["prod_65536"][0]
    print("\nrank  config               16x4096 ms  65536 ms  bs1_4096 ms  speedup")
    for i, r in enumerate(exact, 1):
        t = r["times_ms"]
        print(f'{i:>4}  {r["key"]:<19}  {t["prod_16x4096"][0]:>10.4f}  {t["prod_65536"][0]:>8.4f}  '
              f'{t["bs1_4096"][0]:>11.4f}  {base_total / r["total_ms"]:>6.2f}x')
    print(f'vendor: 16x4096 {base_times["prod_16x4096"][0]:.4f} ms, 65536 {base_times["prod_65536"][0]:.4f} ms, '
          f'bs1_4096 {base_times["bs1_4096"][0]:.4f} ms; exact {len(exact)}/{len(results)}')
    recommended = exact[0] if exact else None
    summary = dict(
        tool="bench_mrope.py", device=torch.cuda.get_device_name(0),
        hip_visible_devices=os.environ.get("HIP_VISIBLE_DEVICES"), torch=torch.__version__,
        vendor_source=str(original_path), vendor_sha256=sha, installed_mode=args.installed, dims=dims,
        warmup=args.warmup, iters=args.iters, seed=args.seed,
        cases=[dict(name=c["name"], tokens=c["ntok"], layout=c["layout"], timed=c["timed"]) for c in cases],
        vendor_times_ms={k: dict(median=v[0], min=v[1]) for k, v in base_times.items()},
        results=results, ranking=[r["key"] for r in exact],
        recommended=None if recommended is None else dict(
            config=recommended["config"], total_ms=recommended["total_ms"], vendor_total_ms=base_total,
            saving_ms_per_layer_call=base_total - recommended["total_ms"]),
        wall_s=round(time.time() - t_start, 1),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(f"recommended: {summary['recommended']}\nwrote {args.output} ({summary['wall_s']} s)")
    if args.write_config and recommended is not None:
        body = dict(status="selected", note=f"bench_mrope.py {args.output}", **recommended["config"])
        args.write_config.write_text(json.dumps(body, indent=2) + "\n")
        print(f"wrote {args.write_config}")
    if args.installed and not exact:
        raise SystemExit("Installed mRoPE is NOT bitwise identical")


if __name__ == "__main__":
    main()
