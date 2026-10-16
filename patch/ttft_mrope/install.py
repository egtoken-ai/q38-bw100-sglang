"""Install the token-tiled mRoPE kernel behind SGLANG_QWEN27B_MROPE_TILED.

Only ``layers/rotary_embedding/triton_kernels.py`` (SHA256-guarded) changes: an
import plus an early-return dispatch inserted right before the single vendor
``_triton_mrope_forward_fused[(num_tokens,)](`` launch in ``triton_mrope_fused``.
The vendor launch stays verbatim as the fallback. ``mrope_tiled.py`` is
installed next to it with ``config.json`` embedded; the rollback copy uses a
non-.py suffix.

CPU simulation (no install): ``python3 install.py --source FILE --out-dir DIR``.
"""
import argparse
import hashlib
import json
from pathlib import Path

ORIGINAL_SHA256 = "49b3a2ca2784b4ffb0773947d2e26055f55a520c9cd847b86bcc3f6c2bf94d05"
TARGET_REL = "layers/rotary_embedding/triton_kernels.py"
HELPER_NAME = "mrope_tiled.py"
HELPER_MODULE = "sglang.srt.layers.rotary_embedding.mrope_tiled"
BACKUP_SUFFIX = ".ttft_mrope_original"
CONFIG_MARKER = "CONFIG = dict(DEFAULT_CONFIG)  # TTFT_MROPE_CONFIG\n"

OLD_IMPORT = "import triton.language as tl\n"
NEW_IMPORT = OLD_IMPORT + f"\nfrom {HELPER_MODULE} import try_mrope_tiled as _ttft_try_mrope_tiled\n"
ANCHOR = (
    "    pad_hd = triton.next_power_of_2(head_size)\n"
    "    _triton_mrope_forward_fused[(num_tokens,)](\n"
)
DISPATCH = (
    "    if _ttft_try_mrope_tiled(\n"
    "        q, k, cos_sin_cache, positions, mrope_section, head_size, rotary_dim,\n"
    "        mrope_interleaved, mrope_interleaved_glm, is_neox_style,\n"
    "    ):\n"
    "        return\n"
)


def transform(source):
    if hashlib.sha256(source.encode()).hexdigest() != ORIGINAL_SHA256:
        raise RuntimeError("triton_kernels.py does not match the audited vendor SHA256")
    if source.count(OLD_IMPORT) != 1 or source.count(ANCHOR) != 1:
        raise RuntimeError("Unexpected import/launch anchors")
    fn = source.index("def triton_mrope_fused(")
    pos = source.index(ANCHOR)
    if pos < fn or "\ndef " in source[fn + 1:pos]:
        raise RuntimeError("Anchor is not inside triton_mrope_fused")
    head, launch = ANCHOR.split("\n", 1)
    updated = source[:pos] + head + "\n" + DISPATCH + launch + source[pos + len(ANCHOR):]
    updated = updated.replace(OLD_IMPORT, NEW_IMPORT, 1)
    compile(updated, "triton_kernels.py", "exec")
    return updated


def validate_config(cfg):
    if not isinstance(cfg, dict):
        raise ValueError("config must be a dict")
    out = {"BLOCK_T": cfg.get("BLOCK_T"), "num_warps": cfg.get("num_warps"), "PER_HEAD": cfg.get("PER_HEAD")}
    for key, lo, hi in (("BLOCK_T", 1, 256), ("num_warps", 1, 16)):
        v = out[key]
        if not isinstance(v, int) or isinstance(v, bool) or not lo <= v <= hi or v & (v - 1):
            raise ValueError(f"{key} must be a power of two in [{lo}, {hi}]: {v!r}")
    if not isinstance(out["PER_HEAD"], bool):
        raise ValueError(f"PER_HEAD must be true/false: {out['PER_HEAD']!r}")
    return out


def load_config(path, allow_placeholder=False):
    raw = json.loads(Path(path).read_text())
    if raw.get("status") != "selected" and not allow_placeholder:
        raise RuntimeError(
            f"{path}: status={raw.get('status')!r}; run bench_mrope.py, copy the recommended "
            "exact config and set status to 'selected' (or pass --allow-placeholder)"
        )
    return raw


def render_helper(helper_source, cfg):
    launch = validate_config(cfg)
    if helper_source.count(CONFIG_MARKER) != 1:
        raise RuntimeError("Helper config marker missing")
    rendered = helper_source.replace(CONFIG_MARKER, f"CONFIG = {launch!r}  # TTFT_MROPE_CONFIG (installed)\n")
    compile(rendered, HELPER_NAME, "exec")
    return rendered, launch


def main():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("/usr/local/lib/python3.10/dist-packages/sglang/srt"))
    p.add_argument("--config", type=Path, default=here / "config.json")
    p.add_argument("--allow-placeholder", action="store_true")
    p.add_argument("--source", type=Path, help="CPU simulation: vendor file to transform")
    p.add_argument("--out-dir", type=Path, help="CPU simulation output directory")
    args = p.parse_args()

    cfg = load_config(args.config, args.allow_placeholder)
    helper_source, launch = render_helper((here / HELPER_NAME).read_text(), cfg)
    if args.source is not None:
        if args.out_dir is None:
            raise SystemExit("--source requires --out-dir")
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "triton_kernels.py").write_text(transform(args.source.read_text()))
        (args.out_dir / HELPER_NAME).write_text(helper_source)
        print(f"Simulated install into {args.out_dir}; launch={launch}")
        return

    target = args.root / TARGET_REL
    backup = target.with_name(target.name + BACKUP_SUFFIX)
    helper = target.with_name(HELPER_NAME)
    if backup.exists() or helper.exists():
        raise RuntimeError("Refusing to overwrite a prior installation")
    original_bytes = target.read_bytes()
    actual = hashlib.sha256(original_bytes).hexdigest()
    if actual != ORIGINAL_SHA256:
        raise RuntimeError(f"Audited parent source changed: {TARGET_REL}: {actual}")
    updated = transform(original_bytes.decode())
    backup.write_bytes(original_bytes)
    helper.write_text(helper_source)
    target.write_text(updated)
    print(f"Installed tiled mRoPE {launch}; disable with SGLANG_QWEN27B_MROPE_TILED=0; backup {backup.name}")


if __name__ == "__main__":
    main()
