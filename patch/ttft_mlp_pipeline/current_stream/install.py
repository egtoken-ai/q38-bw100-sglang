"""Install the current-stream repair, preserving the exact failed helper."""
import hashlib
from pathlib import Path
import py_compile
root=Path('/usr/local/lib/python3.10/dist-packages/sglang/srt/layers')
source=Path(__file__).resolve().parent
target=root/'qwen27b_mlp_pipeline.py'
expected='d4f525cf184bf9472f3f54a750d2e5bc5616a8423da3f2bfe4ed35a5d1b1658a'
if hashlib.sha256(target.read_bytes()).hexdigest()!=expected:
    raise RuntimeError('Failed-parent helper drift; review before installing')
backup=root/'qwen27b_mlp_pipeline_failed.py'
probe=root/'qwen27b_mlp_probe.py'
if backup.exists() or probe.exists():
    raise RuntimeError('Unexpected existing repair/probe')
for name in ('mlp_pipeline.py','pipeline_probe.py'):
    py_compile.compile(str(source/name),doraise=True)
backup.write_bytes(target.read_bytes())
probe.write_bytes((source/'pipeline_probe.py').read_bytes())
target.write_bytes((source/'mlp_pipeline.py').read_bytes())
print('Installed current-stream repair; failed helper preserved')
