"""Record and verify the actual two scheduler processes on the Docker host."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--container', required=True)
p.add_argument('--output', required=True, type=Path)
a = p.parse_args()
if a.output.exists():
    raise SystemExit('Refusing to overwrite binding evidence')
top = subprocess.check_output(['docker', 'top', a.container, '-eo', 'pid,comm'], text=True)
rows = []
for line in top.splitlines()[1:]:
    pid, comm = line.split(maxsplit=1)
    if not comm.startswith('sglang::schedul'):
        continue
    status = dict(line.split(':', 1) for line in Path(f'/proc/{pid}/status').read_text().splitlines() if ':' in line)
    policies = Counter(line.split()[1] for line in Path(f'/proc/{pid}/numa_maps').read_text().splitlines())
    rows.append(dict(pid=int(pid), cpus_allowed=status['Cpus_allowed_list'].strip(), policies=dict(policies)))
report = dict(container=a.container, scheduler_processes=rows,
              expected_cpus='64-79,192-207', expected_policy='prefer:4')
report['verified'] = len(rows) == 2 and all(r['cpus_allowed'] == report['expected_cpus'] and r['policies'].get('prefer:4', 0) > 0 for r in rows)
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
if not report['verified']:
    raise SystemExit('NUMA binding did not match the intended configuration')
