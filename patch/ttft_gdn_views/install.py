"""Install only against the audited vendor source; never silently replace drift."""
import hashlib
import pathlib
import py_compile

ROOT = pathlib.Path('/usr/local/lib/python3.10/dist-packages/sglang/srt')
SOURCE = pathlib.Path(__file__).resolve().parent
FILES = {
    'models/qwen3_5.py': ('qwen3_5.py', 'd8eb6d29defd58cb8300bf30950c3edf22d0470a23fd9f6ea71845ee77315066'),
    'layers/attention/fla/layernorm_gated.py': ('layernorm_gated.py', '3721e41d880a3d1cc39438cd1655807f80ee16d333a5a6339fc25486373363dd'),
}
for relative, (name, expected) in FILES.items():
    target = ROOT / relative
    if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
        raise RuntimeError(f'Vendor source drift: {target}')
    py_compile.compile(str(SOURCE / name), doraise=True)
for relative, (name, _) in FILES.items():
    target = ROOT / relative
    target.with_suffix('.py.ttft_views_original').write_bytes(target.read_bytes())
    target.write_bytes((SOURCE / name).read_bytes())
    py_compile.compile(str(target), doraise=True)
    print('Installed', relative)
