"""CPU-only guard for the exact baseline launcher and bounded NUMA changes."""
import argparse
import hashlib
from pathlib import Path
import subprocess

EXPECTED_SHA256 = 'fa4ca0aa1fd7e62e43e0c567ad4d152092e180efad193ed3d9dc2e80418063a2'
INSERTION = ('# Explicit local NUMA mapping for HIP_VISIBLE_DEVICES=4,5 (logical TP ranks 0,1).\n'
             'if [ "$HIP_VISIBLE_DEVICES" != "4,5" ] || [ "$TP_SIZE" != "2" ]; then\n'
             '  echo "NUMA candidate requires HIP_VISIBLE_DEVICES=4,5 and TP_SIZE=2" >&2\n'
             '  exit 2\n'
             'fi\n'
             'LAUNCH_CMD+=(--numa-node 4 4)\n\n')
ANCHOR = 'if [ "${RUN_IN_BACKGROUND}" = true ]; then'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reference', type=Path, help='Optional mounted original launcher, checked before service start')
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    original = (root / 'start_server.original.sh').read_bytes()
    if hashlib.sha256(original).hexdigest() != EXPECTED_SHA256:
        raise RuntimeError('Original launcher SHA mismatch')
    if args.reference is not None and args.reference.read_bytes() != original:
        raise RuntimeError('Mounted launcher differs from guarded baseline; re-review required')
    lines = original.decode().splitlines(keepends=True)
    positions = [i for i, line in enumerate(lines) if line.startswith('PROJECT_DIR=')]
    if len(positions) != 1:
        raise RuntimeError('Expected one PROJECT_DIR definition')
    lines[positions[0]] = 'PROJECT_DIR=/bench\n'
    expected = ''.join(lines)
    old_visible = 'export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-"0,1"}'
    new_visible = 'export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-"4,5"}'
    if expected.count(old_visible) != 1:
        raise RuntimeError('Expected original HIP device default exactly once')
    expected = expected.replace(old_visible, new_visible)
    if expected.count(ANCHOR) != 1:
        raise RuntimeError('Expected one daemon branch')
    expected = expected.replace(ANCHOR, INSERTION + ANCHOR)
    candidate = root / 'start_server.sh'
    if candidate.read_bytes() != expected.encode():
        raise RuntimeError('Candidate changes exceed NUMA insertion/scope guard, GPU4,5 default and /bench path adaptation')
    subprocess.run(['bash', '-n', str(candidate)], check=True)
    print('PASS: baseline SHA; NUMA scope guard + GPU4,5 default + PROJECT_DIR=/bench; bash syntax')


if __name__ == '__main__':
    main()
