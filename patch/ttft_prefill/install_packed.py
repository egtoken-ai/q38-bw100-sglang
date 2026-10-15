"""Enable only the verified Qwen27B TP2 Prefill packing route."""
import hashlib
from pathlib import Path
import py_compile

root = Path('/usr/local/lib/python3.10/dist-packages/sglang/srt/layers/attention')
target = root / 'pack_paged_kv_to_varlen.py'
expected = 'f54a7202a7507d571f596f7c2d740e38fd0280de766efc2a9872bb8cd56318f4'
assert hashlib.sha256(target.read_bytes()).hexdigest() == expected, 'Unexpected vendor source'
source = Path(__file__).with_name(target.name)
py_compile.compile(str(source), doraise=True)
backup = root / 'pack_paged_kv_to_varlen_original.py'
assert not backup.exists(), 'Refusing to overwrite baseline backup'
backup.write_bytes(target.read_bytes())
target.write_bytes(source.read_bytes())
print('Installed guarded Qwen27B packed Prefill path')
