"""Install only against the audited vendor files, preserving rollback copies."""
import hashlib
from pathlib import Path
import py_compile

ROOT = Path('/usr/local/lib/python3.10/dist-packages/sglang/srt')
SOURCE = Path(__file__).resolve().parent
FILES = {
    'layers/attention/linear/gdn_backend.py': ('gdn_backend.py', '3cc1566a032cb6633096b1a6586191f7a9afbd9f18d0997d4cb23cd3e9395de2'),
    'layers/attention/linear/kernels/gdn_triton.py': ('gdn_triton.py', '31952f087a70ec57212dc9ba0204dab84a711f64ad3b9fd24a9481537386c4e5'),
}
for relative, (name, expected) in FILES.items():
    target = ROOT / relative
    if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
        raise RuntimeError(f'Vendor source drift: {target}')
    py_compile.compile(str(SOURCE / name), doraise=True)
helper = ROOT / 'layers/attention/fla/gdn_prefill_prepare.py'
if helper.exists():
    raise RuntimeError(f'Unexpected existing helper: {helper}')
py_compile.compile(str(SOURCE / helper.name), doraise=True)
helper.write_bytes((SOURCE / helper.name).read_bytes())
for relative, (name, _) in FILES.items():
    target = ROOT / relative
    target.with_suffix('.py.ttft_prepare_original').write_bytes(target.read_bytes())
    target.write_bytes((SOURCE / name).read_bytes())
    py_compile.compile(str(target), doraise=True)
    print('Installed', relative)
