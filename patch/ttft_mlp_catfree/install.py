"""Install the cat-free MLP pipeline helper over the audited 4/8-parts helper.

Usage: python3 install.py [--layers-dir DIR]   (DIR defaults to the image's sglang/srt/layers)
The backup uses a NON-.py suffix so nothing scans it as an importable module.
"""
import argparse
import hashlib
from pathlib import Path
import py_compile

PARENT_SHA256 = 'a946a802abb6a6faeb84108e4ccb10db957bc553bfc573333d026cc681febbad'
CANDIDATE_SHA256 = '1796bea1dbde29d135a6f013f63a39331ca8f38583ae09c4b983144687e217a7'
BACKUP_SUFFIX = '.ttft_catfree_original'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--layers-dir', default='/usr/local/lib/python3.10/dist-packages/sglang/srt/layers')
    args = p.parse_args()
    root = Path(args.layers_dir)
    source = Path(__file__).resolve().parent / 'mlp_pipeline.py'
    helper = root / 'qwen27b_mlp_pipeline.py'
    backup = root / ('qwen27b_mlp_pipeline.py' + BACKUP_SUFFIX)
    if sha256(source) != CANDIDATE_SHA256:
        raise RuntimeError('Candidate helper drift: re-review mlp_pipeline.py and update CANDIDATE_SHA256')
    if sha256(helper) != PARENT_SHA256:
        raise RuntimeError('Installed helper drift: expected the audited 4/8-parts current-stream helper')
    if backup.exists():
        raise RuntimeError('Already installed or unexpected existing backup: ' + str(backup))
    if not (root / 'qwen27b_mlp_probe.py').exists():
        raise RuntimeError('qwen27b_mlp_probe.py missing; parent image is not the expected lineage')
    py_compile.compile(str(source), doraise=True)
    backup.write_bytes(helper.read_bytes())
    helper.write_bytes(source.read_bytes())
    py_compile.compile(str(helper), doraise=True)
    if sha256(helper) != CANDIDATE_SHA256 or sha256(backup) != PARENT_SHA256:
        raise RuntimeError('Post-install hash check failed')
    print('Installed cat-free MLP pipeline helper; SGLANG_QWEN27B_MLP_CATFREE=0 restores per-part AR + cat')
    print('backup:', backup)


if __name__ == '__main__':
    main()
