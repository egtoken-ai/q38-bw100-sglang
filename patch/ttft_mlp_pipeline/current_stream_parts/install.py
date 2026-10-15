"""Install only the 4/8 selector over the audited current-stream+GEMM parent."""
import hashlib
from pathlib import Path
import py_compile
root=Path('/usr/local/lib/python3.10/dist-packages/sglang/srt/layers')
source=Path(__file__).resolve().parent
helper=root/'qwen27b_mlp_pipeline.py'
expected='b066831d162186cd635197703be8293f6c527e5bb0eafc5d3f0f95c1e100a68f'
if hashlib.sha256(helper.read_bytes()).hexdigest()!=expected:
    raise RuntimeError('Current-stream helper drift: review the parent before installing')
backup=root/'qwen27b_mlp_pipeline_before_parts.py'
if backup.exists():
    raise RuntimeError('Already installed or unexpected existing backup')
py_compile.compile(str(source/'mlp_pipeline.py'),doraise=True)
backup.write_bytes(helper.read_bytes())
helper.write_bytes((source/'mlp_pipeline.py').read_bytes())
py_compile.compile(str(helper),doraise=True)
print('Installed current-stream parts selector; default4, only4/8 allowed')
