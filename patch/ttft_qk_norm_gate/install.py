"""Apply a narrow source insertion while preserving all existing parent patches."""
import argparse
import hashlib
import json
from pathlib import Path


def transform(source):
    old_import = "from sglang.srt.models.utils import fused_qk_gemma_rmsnorm\n"
    new_import = old_import + "from sglang.srt.models.qk_norm_gate import try_qk_norm_gate\n"
    if source.count(old_import) != 1:
        raise RuntimeError("Expected exactly one QK norm import")
    class_start = source.index("class Qwen3_5AttentionDecoderLayer(")
    function_start = source.index("    def self_attention(", class_start)
    branch_start = source.index("        if self.attn_output_gate:\n", function_start)
    branch_end = source.index("        attn_output = self.attn(q, k, v, forward_batch)\n", branch_start)
    original = source[branch_start:branch_end]
    if "qkv, _ = self.qkv_proj(hidden_states)" not in source[function_start:branch_start]:
        raise RuntimeError("Missing expected QKV projection boundary")
    if original.count("q, k = self.rotary_emb(positions, q, k)") != 1:
        raise RuntimeError("Unexpected original rotary branch")
    prefix = (
        "        prepared_qkv_gate = try_qk_norm_gate(\n"
        "            self, qkv, forward_batch,\n"
        "            rotary_fusion_enabled=_use_fused_qwen_bailing_rotary,\n"
        "        )\n"
        "        if prepared_qkv_gate is not None:\n"
        "            q, k, v, gate = prepared_qkv_gate\n"
        "            q, k = self.rotary_emb(positions, q, k)\n"
        "        else:\n"
    )
    # Original fallback text is retained verbatim apart from indentation.
    fallback = "".join("    " + line if line.strip() else line for line in original.splitlines(keepends=True))
    updated = source[:branch_start] + prefix + fallback + source[branch_end:]
    updated = updated.replace(old_import, new_import)
    compile(updated, "qwen3_5.py", "exec")
    return updated


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("/usr/local/lib/python3.10/dist-packages/sglang/srt"))
    args = p.parse_args()
    here = Path(__file__).resolve().parent
    manifest = json.loads((here / "source-manifest.json").read_text())
    for relative, expected in manifest["vendor_files"].items():
        actual = hashlib.sha256((args.root/relative).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Audited parent source changed: {relative}: {actual}")
    target = args.root/"models/qwen3_5.py"
    # models/*.py is scanned by ModelRegistry: a Python backup would register
    # every Qwen implementation twice. Keep the rollback copy non-importable.
    backup = target.with_name("qwen3_5.py.ttft_qk_original")
    helper = target.with_name("qk_norm_gate.py")
    if backup.exists() or helper.exists():
        raise RuntimeError("Refusing to overwrite a prior installation")
    original = target.read_text()
    updated = transform(original)
    helper_source = (here/"qk_norm_gate.py").read_text()
    compile(helper_source, str(helper), "exec")
    backup.write_bytes(target.read_bytes())
    helper.write_text(helper_source)
    target.write_text(updated)
    print("Installed guarded QK norm/gate path; preserved GDN and MLP parent hooks")


if __name__ == "__main__":
    main()
