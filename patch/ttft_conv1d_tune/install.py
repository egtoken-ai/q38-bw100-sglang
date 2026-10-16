"""Install the tuned causal_conv1d prefill launch behind SGLANG_QWEN27B_CONV1D_TUNE.

Only ``layers/attention/mamba/causal_conv1d_triton.py`` (SHA256-guarded) is
rewritten: one import line plus a dispatch around the single
``_causal_conv1d_fwd_kernel[grid](...)`` launch in ``causal_conv1d_fn``. The
vendor launch text is kept verbatim as the fallback branch. The helper
``causal_conv1d_ttft.py`` is installed next to it with ``config.json`` embedded.
The rollback copy uses a non-.py suffix so it can never be imported/scanned.

CPU simulation (no install): ``python3 install.py --source FILE --out-dir DIR``.
"""
import argparse
import hashlib
import json
from pathlib import Path

ORIGINAL_SHA256 = "e87f568526b175f9e7e49e742ae2e6e0b36093d455882ed7a45b5593743d5413"
TARGET_REL = "layers/attention/mamba/causal_conv1d_triton.py"
HELPER_NAME = "causal_conv1d_ttft.py"
BACKUP_SUFFIX = ".ttft_conv1d_original"
HELPER_MODULE = "sglang.srt.layers.attention.mamba.causal_conv1d_ttft"
CONFIG_MARKER = "CONFIG = dict(VENDOR_LAUNCH)  # TTFT_CONV1D_CONFIG\n"

OLD_IMPORT = "import triton.language as tl\n"
NEW_IMPORT = OLD_IMPORT + (
    f"from {HELPER_MODULE} import (\n"
    "    _causal_conv1d_fwd_kernel_tile2d as _ttft_conv1d_tile2d_kernel,\n"
    "    select_launch as _ttft_conv1d_select,\n"
    ")\n"
)
LAUNCH_START = "    _causal_conv1d_fwd_kernel[grid](\n"
VENDOR_META = "        BLOCK_M=8,\n        BLOCK_N=256,\n        num_stages=2,\n    )\n"
LAUNCH_END = VENDOR_META + "    return out\n"
TUNED_META = (
    '        BLOCK_M=_ttft_cfg["BLOCK_M"],\n'
    '        BLOCK_N=_ttft_cfg["BLOCK_N"],\n'
    '        num_warps=_ttft_cfg["num_warps"],\n'
    '        num_stages=_ttft_cfg["num_stages"],\n'
    "    )\n"
)
SELECT = (
    "    _ttft_cfg = _ttft_conv1d_select(\n"
    "        x, weight, bias, conv_states, cache_indices, has_initial_state, activation\n"
    "    )\n"
)


def _indent(text):
    return "".join("    " + line if line.strip() else line for line in text.splitlines(keepends=True))


def transform(source):
    if hashlib.sha256(source.encode()).hexdigest() != ORIGINAL_SHA256:
        raise RuntimeError("causal_conv1d_triton.py does not match the audited vendor SHA256")
    if source.count(OLD_IMPORT) != 1 or source.count(LAUNCH_START) != 1:
        raise RuntimeError("Unexpected import/launch anchors")
    fn_start = source.index("def causal_conv1d_fn(")
    start = source.index(LAUNCH_START)
    if start < fn_start:
        raise RuntimeError("Launch is not inside causal_conv1d_fn")
    end = source.index(LAUNCH_END, start) + len(VENDOR_META)
    vendor_launch = source[start:end]
    if vendor_launch.count(VENDOR_META) != 1 or not vendor_launch.endswith(VENDOR_META):
        raise RuntimeError("Unexpected vendor launch meta block")
    tuned_launch = vendor_launch.replace(
        LAUNCH_START, "    _ttft_kernel[grid](\n"
    ).replace(VENDOR_META, TUNED_META)
    replacement = (
        SELECT
        + "    if _ttft_cfg is None:\n"
        + _indent(vendor_launch)
        + "    else:\n"
        + "        _ttft_kernel = (\n"
        + '            _causal_conv1d_fwd_kernel if _ttft_cfg["kernel"] == "orig"\n'
        + "            else _ttft_conv1d_tile2d_kernel\n"
        + "        )\n"
        + _indent(tuned_launch)
    )
    updated = source[:start] + replacement + source[end:]
    updated = updated.replace(OLD_IMPORT, NEW_IMPORT, 1)
    compile(updated, "causal_conv1d_triton.py", "exec")
    return updated


def load_config(path, allow_placeholder=False):
    raw = json.loads(Path(path).read_text())
    if raw.get("status") != "selected" and not allow_placeholder:
        raise RuntimeError(
            f"{path}: status={raw.get('status')!r}; run bench_conv1d.py, copy the recommended "
            "exact config and set status to 'selected' (or pass --allow-placeholder)"
        )
    return raw


_KERNELS = ("orig", "tile2d")


def validate_config(cfg):
    """Return the normalized launch dict or raise ValueError."""
    if not isinstance(cfg, dict):
        raise ValueError("config must be a dict")
    out = {k: cfg.get(k) for k in ("kernel", "BLOCK_M", "BLOCK_N", "num_warps", "num_stages")}
    if out["kernel"] not in _KERNELS:
        raise ValueError(f"kernel must be one of {_KERNELS}: {out['kernel']!r}")
    # BLOCK_M >= 4 keeps every chunk>0 prior-token read inside its sequence
    # (state_len == 3 for the width-4 kernel); the vendor value is 8.
    for key, lo, hi in (("BLOCK_M", 4, 256), ("BLOCK_N", 32, 2048), ("num_warps", 1, 16)):
        value = out[key]
        if not isinstance(value, int) or isinstance(value, bool) or not lo <= value <= hi or value & (value - 1):
            raise ValueError(f"{key} must be a power of two in [{lo}, {hi}]: {value!r}")
    if out["num_stages"] not in (1, 2, 3, 4):
        raise ValueError(f"num_stages must be 1..4: {out['num_stages']!r}")
    return out


def render_helper(helper_source, cfg):
    launch = validate_config(cfg)
    if helper_source.count(CONFIG_MARKER) != 1:
        raise RuntimeError("Helper config marker missing")
    rendered = helper_source.replace(
        CONFIG_MARKER, f"CONFIG = {launch!r}  # TTFT_CONV1D_CONFIG (installed)\n"
    )
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
        (args.out_dir / "causal_conv1d_triton.py").write_text(transform(args.source.read_text()))
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
    print(f"Installed tuned causal_conv1d prefill launch {launch}; "
          f"disable with SGLANG_QWEN27B_CONV1D_TUNE=0; backup {backup.name}")


if __name__ == "__main__":
    main()
