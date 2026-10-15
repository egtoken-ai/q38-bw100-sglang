"""Extend the audited deployed helper by one exact M65536 out-projection shape."""
import argparse
import hashlib
from pathlib import Path

EXPECTED_SHA256 = '302417813ae667e02ed8c7a48784482043049d91b68e0b157f4302596928d428'
RELATIVE = Path('layers/quantization/compressed_tensors/schemes/q38_bw100_legacy_gemm_dispatch.py')
ANCHOR = ('SHAPES = {(m, n, k) for m in (4096, 8192, 16384, 32768)\n'
          '          for n, k in ((17408, 5120), (8192, 5120), (5120, 3072))}\n')
ADDITION = 'SHAPES.add((65536, 5120, 3072))\n'


def extend(original):
    if hashlib.sha256(original).hexdigest() != EXPECTED_SHA256:
        raise RuntimeError('Deployed GEMM helper SHA mismatch; refusing to overwrite another patch')
    source = original.decode()
    if source.count(ANCHOR) != 1:
        raise RuntimeError('SHAPES anchor mismatch')
    candidate = source.replace(ANCHOR, ANCHOR + ADDITION)
    compile(candidate, str(RELATIVE), 'exec')
    return candidate.encode()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('/usr/local/lib/python3.10/dist-packages/sglang/srt'))
    p.add_argument('--check', action='store_true')
    args = p.parse_args()
    target = args.root / RELATIVE
    backup = target.with_suffix('.py.before_64k_out')
    if backup.exists():
        raise RuntimeError('Existing non-py backup; refusing duplicate installation')
    original = target.read_bytes()
    candidate = extend(original)
    if not args.check:
        backup.write_bytes(original)
        target.write_bytes(candidate)
    print('PASS: ' + ('check only; no writes; ' if args.check else 'installed; ') +
          'only added SHAPES tuple (65536,5120,3072); candidate SHA256=' +
          hashlib.sha256(candidate).hexdigest())


if __name__ == '__main__':
    main()
