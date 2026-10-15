"""Real temporary installation and allowlist checks; no torch import or GPU."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from install import ADDITION, EXPECTED_SHA256, RELATIVE, extend


def load(source):
    namespace = {}
    exec(compile(source, '<gemm-helper-cpu>', 'exec'), namespace)
    return namespace


def main():
    here = Path(__file__).resolve().parent
    project = here.parents[2]
    original = (project / 'patch/ttft_gemm_dispatch/gemm_dispatch.py').read_bytes()
    assert hashlib.sha256(original).hexdigest() == EXPECTED_SHA256
    candidate = extend(original)
    assert candidate.replace(ADDITION.encode(), b'', 1) == original
    old, new = load(original), load(candidate)
    assert new['SHAPES'] == old['SHAPES'] | {(65536, 5120, 3072)}
    # All definitions, formulas and API signatures remain byte-for-byte intact.
    for node in ast.parse(original).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            other = next(x for x in ast.parse(candidate).body if isinstance(x, ast.FunctionDef) and x.name == node.name)
            assert ast.dump(node) == ast.dump(other)
    old_shapes = json.loads((project / 'patch/ttft_gemm_dispatch/approved-shapes.json').read_text())['shapes']
    expected = {tuple(x) for x in old_shapes} | {(65536, 5120, 3072)}
    dockerfile = (here/'Dockerfile').read_text()
    value = next(line.split('=',1)[1] for line in dockerfile.splitlines() if line.startswith('ENV Q38_HIPBLASLT_SHAPES='))
    assert new['approved_shapes'](value) == expected and len(expected) == 10
    assert new['approved_shapes']('') == set()
    original_env = ','.join('x'.join(map(str,x)) for x in old_shapes)
    assert new['approved_shapes'](original_env) == {tuple(x) for x in old_shapes}
    for forbidden in ('65536x17408x5120', '65536x8192x5120', '65536x5120x8704'):
        try:
            new['approved_shapes'](forbidden)
        except ValueError:
            pass
        else:
            raise AssertionError('Unapproved shape accepted: '+forbidden)
    def run(root, *args):
        return subprocess.run([sys.executable, str(here/'install.py'), '--root', str(root), *args], capture_output=True,text=True)
    with tempfile.TemporaryDirectory(prefix='q38-gemm64k-cpu-') as temp:
        root = Path(temp)
        target = root/RELATIVE
        target.parent.mkdir(parents=True)
        target.write_bytes(original)
        backup = target.with_suffix('.py.before_64k_out')
        untouched = target.with_name('compressed_tensors_w8a8_int8.py')
        untouched.write_text('unrelated existing source marker\n')
        result=run(root,'--check');assert result.returncode==0,result.stderr
        assert target.read_bytes()==original and not backup.exists()
        result=run(root);assert result.returncode==0,result.stderr
        assert target.read_bytes()==candidate and backup.read_bytes()==original
        assert backup not in list(target.parent.glob('*.py'))
        assert untouched.read_text()=='unrelated existing source marker\n'
        result=run(root);assert result.returncode!=0
        assert target.read_bytes()==candidate and backup.read_bytes()==original
    with tempfile.TemporaryDirectory(prefix='q38-gemm64k-drift-') as temp:
        root=Path(temp);target=root/RELATIVE;target.parent.mkdir(parents=True)
        target.write_bytes(original+b'\n# simulated other patch\n')
        previous=target.read_bytes();result=run(root)
        assert result.returncode!=0 and target.read_bytes()==previous
        assert not target.with_suffix('.py.before_64k_out').exists()
    print('PASS: exact one-tuple change; functions unchanged; 10-shape ENV; rejected other M65536 shapes; check/install/reinstall/drift/backup CPU tests')


if __name__ == '__main__':
    main()
