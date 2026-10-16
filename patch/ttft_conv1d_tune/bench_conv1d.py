"""BW100 causal_conv1d prefill launch sweep with bitwise exactness gate.

Single process, one GPU (HIP_VISIBLE_DEVICES selects it). Nothing installed is
modified: the reference is the audited vendor source (from the image file, or
its ``.py.ttft_conv1d_original`` backup on an installed image) imported from a
temp copy; the candidate is ``install.transform`` of that same text with the
patch-dir helper. ``--installed`` instead tests the real sglang module of an
installed image against the backup.

Output AND final conv_states (whole pool) must be bitwise equal (int16 views)
to the vendor launch on every case; only then is a config timed/ranked.
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


INSTALL = load_module("ttft_conv1d_install", HERE / "install.py")


def model_dims(config_path):
    dims = dict(conv_dim=5120, width=4, activation="silu", source="builtin (Qwen3.8-27B TP2)")
    try:
        cfg = json.loads(Path(config_path).read_text())
        tc = cfg.get("text_config", cfg)
        tp = 2
        conv = (2 * tc["linear_num_key_heads"] * tc["linear_key_head_dim"]
                + tc["linear_num_value_heads"] * tc["linear_value_head_dim"])
        dims.update(conv_dim=conv // tp, width=tc["linear_conv_kernel_dim"],
                    activation=tc.get("hidden_act", "silu"), source=str(config_path),
                    z_dim=tc["linear_num_value_heads"] * tc["linear_value_head_dim"] // tp)
    except (OSError, KeyError, ValueError) as exc:
        print(f"[warn] model config unavailable ({exc}); using builtin dims", flush=True)
    dims.setdefault("z_dim", 3072)
    return dims


def bitwise_equal(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))


def build_cases(dims, device, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    dim, width, zdim = dims["conv_dim"], dims["width"], dims["z_dim"]
    total = 65536
    # Production layout: mixed_qkv = projected_qkvz[:, :dim] with row stride dim+z.
    proj = (torch.randn(total, dim + zdim, generator=g) * 2).to(torch.bfloat16).to(device)
    weight = (torch.randn(dim, 1, width, generator=g) * 0.5).to(torch.bfloat16).to(device)
    weight = weight.view(dim, width)
    bias = (torch.randn(dim, generator=g) * 0.1).to(torch.bfloat16).to(device)
    slots = 48
    pool = (torch.randn(2, slots, dim, width - 1, generator=g)).to(torch.bfloat16).to(device)

    def case(name, lens, has_init, timed=False, layout="prod", with_bias=False):
        ntok = sum(lens)
        if layout == "prod":
            x = proj[:ntok, :dim].transpose(0, 1)
        else:
            x = proj[:ntok, :dim].contiguous().transpose(0, 1)
        qsl = torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()), dtype=torch.int32, device=device)
        perm = torch.randperm(slots - 1, generator=g)[: len(lens)] + 1
        return dict(name=name, x=x, lens=list(lens), qsl=qsl,
                    cache_indices=perm.to(torch.int32).to(device),
                    has_init=torch.tensor(has_init, dtype=torch.bool, device=device),
                    bias=bias if with_bias else None, timed=timed, layout=layout)

    main = [
        case("prod_16x4096", [4096] * 16, [False] * 16, timed=True),
        case("prod_65536", [65536], [False], timed=True),
        case("mixed_16x4096", [4096] * 16, [i % 2 == 1 for i in range(16)]),
        case("init_65536", [65536], [True]),
        case("odd_63_129", [63, 129], [True, False]),
        case("short_1_2_3_5_4096", [1, 2, 3, 5, 4096], [True, False, True, True, False]),
    ]
    extra = [
        case("odd_bias_63_129", [63, 129, 7], [True, False, True], with_bias=True),
        case("contig_63_129_4096", [63, 129, 4096], [False, True, True], layout="contig"),
    ]
    return dict(weight=weight, pool=pool, main=main, extra=extra)


def run_once(fn, data, c, activation, pool_init):
    pool = pool_init.clone()
    conv_states = pool[1]
    out = fn(c["x"], data["weight"], c["bias"], conv_states=conv_states,
             query_start_loc=c["qsl"], seq_lens_cpu=c["lens"], cache_indices=c["cache_indices"],
             has_initial_state=c["has_init"], activation=activation)
    return out, pool


def time_fn(fn, data, c, activation, warmup, iters):
    pool = data["pool"].clone()
    conv_states = pool[1]

    def call():
        fn(c["x"], data["weight"], c["bias"], conv_states=conv_states,
           query_start_loc=c["qsl"], seq_lens_cpu=c["lens"], cache_indices=c["cache_indices"],
           has_initial_state=c["has_init"], activation=activation)
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        call()
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2], times[0]


def default_grid():
    grid = []
    for bm in (8, 16, 32, 64):
        for bn in (256, 512):
            grid.append(dict(kernel="orig", BLOCK_M=bm, BLOCK_N=bn, num_warps=4, num_stages=2))
    grid += [
        dict(kernel="orig", BLOCK_M=16, BLOCK_N=1024, num_warps=4, num_stages=2),
        dict(kernel="orig", BLOCK_M=32, BLOCK_N=512, num_warps=8, num_stages=2),
        dict(kernel="orig", BLOCK_M=32, BLOCK_N=1024, num_warps=8, num_stages=2),
        dict(kernel="orig", BLOCK_M=32, BLOCK_N=256, num_warps=4, num_stages=1),
        dict(kernel="orig", BLOCK_M=64, BLOCK_N=256, num_warps=2, num_stages=1),
    ]
    for bm, bn, w in ((8, 256, 4), (8, 512, 4), (8, 1024, 4), (16, 256, 4), (16, 512, 4),
                      (32, 128, 4), (32, 256, 4), (64, 128, 4), (16, 256, 2), (8, 256, 2),
                      (16, 512, 8), (32, 256, 8), (32, 512, 8), (64, 256, 8)):
        grid.append(dict(kernel="tile2d", BLOCK_M=bm, BLOCK_N=bn, num_warps=w, num_stages=1))
    return grid


def key(cfg):
    return f'{cfg["kernel"]}:M{cfg["BLOCK_M"]}:N{cfg["BLOCK_N"]}:w{cfg["num_warps"]}:s{cfg["num_stages"]}'


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model-config", default="/model/config.json")
    p.add_argument("--grid-json", type=Path, help="JSON list of configs to sweep instead of the default grid")
    p.add_argument("--only-config", type=Path, help="Check just this config.json (e.g. the selected one)")
    p.add_argument("--installed", action="store_true", help="Test the installed sglang module vs the backup")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--top-extra", type=int, default=4, help="Top configs that also run extra coverage cases")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--write-config", type=Path, help="Write the recommended exact config here (status=selected)")
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

    tmp = Path(tempfile.mkdtemp(prefix="ttft_conv1d_"))
    (tmp / "conv1d_vendor_ref.py").write_text(original)
    ref_mod = load_module("ttft_conv1d_vendor_ref", tmp / "conv1d_vendor_ref.py")
    if args.installed:
        if not backup.exists():
            raise SystemExit("--installed requires the .py.ttft_conv1d_original backup")
        from sglang.srt.layers.attention.mamba import causal_conv1d_triton as cand_mod
        from sglang.srt.layers.attention.mamba import causal_conv1d_ttft as helper
    else:
        helper = load_module(INSTALL.HELPER_MODULE, HERE / INSTALL.HELPER_NAME)
        helper.ENABLED = True
        (tmp / "conv1d_candidate.py").write_text(INSTALL.transform(original))
        cand_mod = load_module("ttft_conv1d_candidate", tmp / "conv1d_candidate.py")

    dims = model_dims(args.model_config)
    data = build_cases(dims, device, args.seed)
    act = dims["activation"]

    # Reference outputs (vendor launch, vendor source).
    refs = {}
    for c in data["main"] + data["extra"]:
        out, pool = run_once(ref_mod.causal_conv1d_fn, data, c, act, data["pool"])
        # Determinism of the reference itself.
        out2, pool2 = run_once(ref_mod.causal_conv1d_fn, data, c, act, data["pool"])
        if not (bitwise_equal(out, out2) and bitwise_equal(pool, pool2)):
            raise SystemExit(f"Vendor kernel not deterministic on {c['name']}")
        refs[c["name"]] = (out, pool)
    torch.cuda.synchronize()
    base_times = {c["name"]: time_fn(ref_mod.causal_conv1d_fn, data, c, act, args.warmup, args.iters)
                  for c in data["main"] if c["timed"]}

    if args.installed:
        grid = [dict(helper.CONFIG)]
    elif args.only_config:
        grid = [INSTALL.validate_config(json.loads(args.only_config.read_text()))]
    elif args.grid_json:
        grid = [INSTALL.validate_config(c) for c in json.loads(args.grid_json.read_text())]
    else:
        grid = [INSTALL.validate_config(c) for c in default_grid()]

    def check(cfg, cases):
        if not args.installed:
            helper.CONFIG = cfg
        bad = []
        for c in cases:
            out, pool = run_once(cand_mod.causal_conv1d_fn, data, c, act, data["pool"])
            if not bitwise_equal(out, refs[c["name"]][0]):
                bad.append(f'{c["name"]}:out')
            if not bitwise_equal(pool, refs[c["name"]][1]):
                bad.append(f'{c["name"]}:conv_state')
        return bad

    results = []
    for cfg in grid:
        row = dict(config=cfg, key=key(cfg))
        try:
            t0 = time.time()
            bad = check(cfg, data["main"])
            row["exact_main"] = not bad
            row["mismatch"] = bad
            if not bad:
                row["times_ms"] = {}
                for c in data["main"]:
                    if c["timed"]:
                        row["times_ms"][c["name"]] = time_fn(cand_mod.causal_conv1d_fn, data, c, act,
                                                             args.warmup, args.iters)
                row["total_ms"] = sum(v[0] for v in row["times_ms"].values())
            row["wall_s"] = round(time.time() - t0, 2)
        except Exception as exc:  # compile/launch failure of one candidate
            row.update(exact_main=False, error=f"{type(exc).__name__}: {exc}"[:400])
            torch.cuda.synchronize()
        print(json.dumps({k: row.get(k) for k in ("key", "exact_main", "mismatch", "total_ms", "error")}), flush=True)
        results.append(row)

    exact = sorted((r for r in results if r.get("exact_main")), key=lambda r: r["total_ms"])
    for r in exact[: max(args.top_extra, 1)]:
        try:
            bad = check(r["config"], data["extra"])
            r["exact_extra"] = not bad
            r["mismatch_extra"] = bad
        except Exception as exc:
            r["exact_extra"] = False
            r["error_extra"] = f"{type(exc).__name__}: {exc}"[:400]
    base_total = sum(v[0] for v in base_times.values())
    recommended = next((r for r in exact if r.get("exact_extra")), None)

    print("\nrank  config                              16x4096 ms  65536 ms  total ms  speedup  extra")
    for i, r in enumerate(exact, 1):
        t = r["times_ms"]
        print(f'{i:>4}  {r["key"]:<34}  {t["prod_16x4096"][0]:>10.3f}  {t["prod_65536"][0]:>8.3f}  '
              f'{r["total_ms"]:>8.3f}  {base_total / r["total_ms"]:>6.3f}x  {r.get("exact_extra", "-")}')
    print(f'vendor launch (reference): 16x4096 {base_times["prod_16x4096"][0]:.3f} ms, '
          f'65536 {base_times["prod_65536"][0]:.3f} ms; exact {len(exact)}/{len(results)}')
    nbytes = 2 * 2 * dims["conv_dim"] * 65536
    summary = dict(
        tool="bench_conv1d.py", device=torch.cuda.get_device_name(0),
        hip_visible_devices=os.environ.get("HIP_VISIBLE_DEVICES"), torch=torch.__version__,
        vendor_source=str(original_path), vendor_sha256=sha, installed_mode=args.installed,
        dims=dims, warmup=args.warmup, iters=args.iters, seed=args.seed,
        cases=[dict(name=c["name"], lens=c["lens"] if len(c["lens"]) < 8 else f'{len(c["lens"])}x{c["lens"][0]}',
                    has_init=c["has_init"].tolist(), layout=c["layout"], bias=c["bias"] is not None,
                    timed=c["timed"]) for c in data["main"] + data["extra"]],
        vendor_times_ms={k: dict(median=v[0], min=v[1], GBps=nbytes / v[0] / 1e6) for k, v in base_times.items()},
        results=results,
        ranking=[r["key"] for r in exact],
        recommended=None if recommended is None else dict(
            config=recommended["config"], total_ms=recommended["total_ms"],
            vendor_total_ms=base_total, saving_ms_per_layer_call=base_total - recommended["total_ms"]),
        wall_s=round(time.time() - t_start, 1),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(f"recommended: {summary['recommended']}\nwrote {args.output} ({summary['wall_s']} s)")
    if args.write_config and recommended is not None:
        body = dict(status="selected", note=f"bench_conv1d.py {args.output}", **recommended["config"])
        args.write_config.write_text(json.dumps(body, indent=2) + "\n")
        print(f"wrote {args.write_config}")
    if args.installed and not (exact and exact[0].get("exact_extra")):
        raise SystemExit("Installed launch is NOT bitwise identical")


if __name__ == "__main__":
    main()
