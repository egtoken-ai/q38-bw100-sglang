"""Install the opt-in fusion on the exact audited vendor source version."""
import argparse
import hashlib
from pathlib import Path
import py_compile

EXPECTED = {
    "models/qwen3_5.py": "d8eb6d29defd58cb8300bf30950c3edf22d0470a23fd9f6ea71845ee77315066",
    "models/qwen2_moe.py": "015fc17155a8e4a56d6d2283c932aa0e4bfe6221d5fcc569a9874db358bef3f3",
    "layers/linear.py": "7806a8f6483deedc592fb5e97de4a211512e0a469eceb7e38a0826b14d563dd8",
    "layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8.py":
        "f8dd55d9a7cabc807c314db3c1ef4a95047da868005d674932db97d43bb4399b",
}
OLD_IMPORT = "from sglang.srt.models.qwen2_moe import Qwen2MoeMLP, Qwen2MoeSparseMoeBlock\n"
NEW_IMPORT = (
    "from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock\n"
    "from sglang.srt.models.qwen27b_int8_mlp import Qwen27BInt8MLP as Qwen2MoeMLP\n"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(
        "/usr/local/lib/python3.10/dist-packages/sglang/srt"))
    args = parser.parse_args()
    for relative, expected in EXPECTED.items():
        path = args.root / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Unexpected vendor source: {path}")
    target = args.root / "models/qwen3_5.py"
    backup = target.with_name("qwen3_5_before_ttft_silu.py")
    helper = target.with_name("qwen27b_int8_mlp.py")
    if backup.exists() or helper.exists():
        raise RuntimeError("Refusing to overwrite an installed candidate or backup")
    original = target.read_text()
    if original.count(OLD_IMPORT) != 1:
        raise RuntimeError("Expected exactly one Qwen dense MLP import")
    replacement = original.replace(OLD_IMPORT, NEW_IMPORT)
    compile(replacement, str(target), "exec")
    source = Path(__file__).with_name(helper.name)
    py_compile.compile(str(source), doraise=True)
    backup.write_bytes(target.read_bytes())
    helper.write_bytes(source.read_bytes())
    target.write_text(replacement)
    print("Installed opt-in Qwen27B SiLU/INT8 quantization fusion")


if __name__ == "__main__":
    main()
