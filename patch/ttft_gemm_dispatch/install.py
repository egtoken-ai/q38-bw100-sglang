#!/usr/bin/env python3
"""Install only on the audited original W8A8 source; --check never writes."""
import argparse
import hashlib
import json
from pathlib import Path

from gemm_dispatch import HELPER_MODULE, patch_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='/usr/local/lib/python3.10/dist-packages/sglang/srt')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    source_dir = Path(__file__).resolve().parent
    root = Path(args.root)
    manifest = json.loads((source_dir / 'source-manifest.json').read_text())
    relative = Path('layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8.py')
    target = root / relative
    original = target.read_bytes()
    if hashlib.sha256(original).hexdigest() != manifest['sha256']:
        raise RuntimeError(f'Audited W8A8 source SHA mismatch: {target}')
    vendor = source_dir / 'vendor/compressed_tensors_w8a8_int8.py'
    if hashlib.sha256(vendor.read_bytes()).hexdigest() != manifest['sha256']:
        raise RuntimeError('Bundled vendor source SHA mismatch')
    helper = target.with_name(HELPER_MODULE.rsplit('.', 1)[1] + '.py')
    backup = target.with_suffix('.py.ttft_gemm_dispatch_original')
    if helper.exists() or backup.exists():
        raise RuntimeError('Existing helper or backup; do not overwrite a prior installation')
    patched = patch_source(original.decode())
    helper_text = (source_dir / 'gemm_dispatch.py').read_text()
    compile(patched, str(target), 'exec')
    compile(helper_text, str(helper), 'exec')
    if args.check:
        print('PASS: original/vendor SHA, unique anchor, helper/backup absence, syntax; no writes')
        return
    backup.write_bytes(original)
    helper.write_text(helper_text)
    target.write_text(patched)
    print('Installed guarded native hipBLASLt candidate; Q38_HIPBLASLT_SHAPES defaults empty')


if __name__ == '__main__':
    main()
