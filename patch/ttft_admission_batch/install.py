"""SHA-guarded additive scheduler hook, without touching other model patches."""
import argparse
import hashlib
import json
from pathlib import Path

EXPECTED = '921491a9a881ce3a863bc951763aec7c59023c4d71fb4917066414df8b4cbd65'
IMPORT_ANCHOR = 'from sglang.srt.environ import envs\n'
IMPORT = ('from sglang.srt.managers.qwen27b_admission_batch import (\n'
          '    collect_idle_generate_requests,\n)\n')
HOOK_ANCHOR = '                    recv_reqs.append(recv_rpc)\n            else:\n'
HOOK = ('                    recv_reqs.append(recv_rpc)\n'
        '                recv_reqs = collect_idle_generate_requests(\n'
        '                    self, recv_reqs, TokenizedGenerateReqInput\n'
        '                )\n            else:\n')


def transformed(source):
    if source.count(IMPORT_ANCHOR) != 1 or source.count(HOOK_ANCHOR) != 1:
        raise RuntimeError('Scheduler hook anchors drifted')
    return source.replace(IMPORT_ANCHOR, IMPORT_ANCHOR + IMPORT).replace(HOOK_ANCHOR, HOOK)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('/usr/local/lib/python3.10/dist-packages/sglang/srt'))
    p.add_argument('--check-only', action='store_true')
    args = p.parse_args()
    bundle = Path(__file__).resolve().parent
    manifest = json.loads((bundle / 'source-manifest.json').read_text())
    helper_bytes = (bundle / 'admission_batch.py').read_bytes()
    if hashlib.sha256(helper_bytes).hexdigest() != manifest['helper_sha256']:
        raise RuntimeError('Admission helper SHA mismatch')
    path = args.root / 'managers/scheduler.py'
    backup = path.with_name('scheduler_before_qwen27b_admission.py')
    helper = path.with_name('qwen27b_admission_batch.py')
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != EXPECTED:
        raise RuntimeError('Parent scheduler SHA mismatch; preserve existing patches and re-review')
    if backup.exists() or helper.exists():
        raise RuntimeError('Admission backup/helper already exists; refusing overwrite')
    candidate = transformed(source.decode()).encode()
    compile(candidate, str(path), 'exec')
    compile(helper_bytes, str(helper), 'exec')
    if not args.check_only:
        backup.write_bytes(source)
        helper.write_bytes(helper_bytes)
        path.write_bytes(candidate)
    print(json.dumps({'check_only': args.check_only, 'parent_scheduler_sha256': EXPECTED,
                      'candidate_scheduler_sha256': hashlib.sha256(candidate).hexdigest(),
                      'helper_sha256': manifest['helper_sha256']}))


if __name__ == '__main__':
    main()
