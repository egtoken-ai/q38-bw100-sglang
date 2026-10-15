"""SHA-guarded install on the audited prepare-w4 image; --check is read-only."""
import argparse
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--root', default='/usr/local/lib/python3.10/dist-packages/sglang/srt')
p.add_argument('--check', action='store_true')
a=p.parse_args()
root=Path(a.root); source=Path(__file__).resolve().parent
manifest=json.loads((source/'vendor-sha256.json').read_text())
for relative,expected in manifest.items():
    target=root/relative
    if hashlib.sha256(target.read_bytes()).hexdigest()!=expected:
        raise RuntimeError(f'Audited vendor source drift: {target}')
patched=[x for x in (source/'patched').rglob('*.py')]
helper=root/'layers/attention/fla/gdn_cpu_metadata.py'
if helper.exists():raise RuntimeError(f'Unexpected existing helper: {helper}')
for patch in patched:
    relative=patch.relative_to(source/'patched')
    if (root/relative).with_suffix('.py.ttft_metadata_original').exists():
        raise RuntimeError(f'Existing backup: {relative}')
    compile(patch.read_text(),str(patch),'exec')
compile((source/'gdn_cpu_metadata.py').read_text(),str(source/'gdn_cpu_metadata.py'),'exec')
if a.check:
    print(f'SHA guard valid: {len(manifest)} files; {len(patched)} replacements')
else:
    for patch in patched:
        target=root/patch.relative_to(source/'patched')
        target.with_suffix('.py.ttft_metadata_original').write_bytes(target.read_bytes())
        target.write_bytes(patch.read_bytes())
    helper.write_bytes((source/'gdn_cpu_metadata.py').read_bytes())
    print('Installed GDN CPU metadata; use SGLANG_QWEN27B_GDN_CPU_METADATA=0 for fallback')
