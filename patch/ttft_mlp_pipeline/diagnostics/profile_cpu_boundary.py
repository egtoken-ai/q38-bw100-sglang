"""CPU-only diagnostic for pipeline host stalls, never an acceptance workload."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import sys
import time
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
from benchmark import make_prompt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output-dir', required=True, type=Path)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    base = 'http://127.0.0.1:18580'
    session = requests.Session()
    session.trust_env = False

    def control(endpoint, **body):
        r = session.post(base + endpoint, json=body, timeout=60)
        r.raise_for_status()

    def wave(name):
        def request(i):
            s = requests.Session()
            s.trust_env = False
            start = time.monotonic()
            r = s.post(base + '/v1/completions', json={
                'model': 'Qwen3.8-27B-Channel-INT8-w8a8',
                'prompt': make_prompt(4096, 20260912, i),
                'temperature': 0, 'max_tokens': 1, 'ignore_eos': True,
            }, timeout=60)
            r.raise_for_status()
            usage = r.json()['usage']
            assert usage['prompt_tokens'] == 4096 and usage['completion_tokens'] == 1
            return {'elapsed_s': time.monotonic()-start, 'usage': usage}
        with concurrent.futures.ThreadPoolExecutor(9) as pool:
            rows = list(pool.map(request, range(9)))
        (args.output_dir / f'{name}.json').write_text(json.dumps(rows, indent=2)+'\n')

    wave('warmup')
    control('/start_profile', output_dir=str(args.output_dir), activities=['CPU'],
            with_stack=False, record_shapes=False, merge_profiles=False)
    try:
        wave('profiled')
    finally:
        control('/stop_profile')
    print('CPU-only diagnostic complete; not TTFT acceptance')


if __name__ == '__main__':
    main()
