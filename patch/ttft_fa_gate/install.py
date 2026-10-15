"""Install only the validated attention gate hook on the audited parent."""
import argparse
import hashlib
import json
from pathlib import Path

OLD = """        if self.attn_output_gate:
            gate = torch.sigmoid(gate)
            attn_output = attn_output * gate
"""
NEW = """        if self.attn_output_gate:
            fused_gate_output = try_fa_gate(self, attn_output, gate, forward_batch)
            if fused_gate_output is None:
                gate = torch.sigmoid(gate)
                attn_output = attn_output * gate
            else:
                attn_output = fused_gate_output
"""
IMPORT = "from sglang.srt.models.qk_norm_gate import try_qk_norm_gate\n"


def transform(original):
    if original.count(OLD) != 1 or original.count(IMPORT) != 1:
        raise RuntimeError("Expected the audited QK parent import and one FA gate block")
    attention = original.index("class Qwen3_5AttentionDecoderLayer(")
    method = original.index("    def self_attention(", attention)
    if original.index(OLD) < method:
        raise RuntimeError("Gate block is outside the expected attention method")
    updated = original.replace(OLD, NEW).replace(
        IMPORT, IMPORT + "from sglang.srt.models.fa_gate import try_fa_gate\n")
    compile(updated, "qwen3_5.py", "exec")
    return updated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/usr/local/lib/python3.10/dist-packages/sglang/srt"))
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    manifest = json.loads((here/"source-manifest.json").read_text())
    for relative, expected in manifest["vendor_files"].items():
        actual = hashlib.sha256((args.root/relative).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Audited parent source changed: {relative}: {actual}")
    target = args.root/"models/qwen3_5.py"
    backup = target.with_name("qwen3_5.py.ttft_fa_gate_original")
    helper = target.with_name("fa_gate.py")
    if backup.exists() or helper.exists():
        raise RuntimeError("Refusing to overwrite an earlier candidate or backup")
    updated = transform(target.read_text())
    helper_text = (here/"fa_gate.py").read_text()
    compile(helper_text, str(helper), "exec")
    backup.write_bytes(target.read_bytes())
    helper.write_text(helper_text)
    target.write_text(updated)
    print("Installed exact BF16 FA gate hook; all parent QK/MLP/GDN code retained")


if __name__ == "__main__":
    main()
