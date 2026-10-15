"""Fail closed on source drift; install bounded Qwen27B pipeline candidate."""
import hashlib
from pathlib import Path
import py_compile

ROOT = Path('/usr/local/lib/python3.10/dist-packages/sglang/srt')
SOURCE = Path(__file__).resolve().parent
TARGET = ROOT / 'models/qwen3_5.py'
EXPECTED = '11621e079bce20628ed1015a9e72128e58710831992dcae12e3db4a55dbf0e75'
BACKUP = TARGET.with_suffix('.py.ttft_mlp_pipeline_original')
HELPER = ROOT / 'layers/qwen27b_mlp_pipeline.py'
DEPENDENCIES = {
    'models/qwen2_moe.py': '015fc17155a8e4a56d6d2283c932aa0e4bfe6221d5fcc569a9874db358bef3f3',
    'layers/linear.py': '7806a8f6483deedc592fb5e97de4a211512e0a469eceb7e38a0826b14d563dd8',
    'layers/layernorm.py': '5e8ef141d575d694103333708e08c50475dde47d318321a6dab9006d2c881e83',
    'layers/communicator.py': '1472b131cb6b5fe11ebe1772dc0d51a3dc65c7128f27547b7b0b0ff81e27179d',
    'layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8.py': 'f8dd55d9a7cabc807c314db3c1ef4a95047da868005d674932db97d43bb4399b',
}
for relative, digest in DEPENDENCIES.items():
    if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != digest:
        raise RuntimeError(f'Pipeline dependency differs from audited source: {relative}')
if hashlib.sha256(TARGET.read_bytes()).hexdigest() != EXPECTED:
    raise RuntimeError('Qwen3_5 source differs from audited views/prepare parent')
if BACKUP.exists() or HELPER.exists():
    raise RuntimeError('Unexpected existing pipeline installation')
for name in ('qwen3_5.py','mlp_pipeline.py'):
    py_compile.compile(str(SOURCE / name),doraise=True)
BACKUP.write_bytes(TARGET.read_bytes())
HELPER.write_bytes((SOURCE / 'mlp_pipeline.py').read_bytes())
TARGET.write_bytes((SOURCE / 'qwen3_5.py').read_bytes())
py_compile.compile(str(HELPER),doraise=True)
py_compile.compile(str(TARGET),doraise=True)
print('Installed bounded dense Qwen27B MLP pipeline')
