#!/usr/bin/env python3
"""Profile Prefill with one output token; diagnostic data, not acceptance scores."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import time

import requests

from benchmark import make_prompt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:18580')
    parser.add_argument('--model', default='Qwen3.8-27B-Channel-INT8-w8a8')
    parser.add_argument('--input-len', type=int, required=True)
    parser.add_argument('--concurrency', type=int, default=1, choices=[1],
                        help='Single-request profiling only: C64 stalled this vendor profiler')
    parser.add_argument('--output-dir', required=True,
                        help='Shared path visible to both client and server')
    args = parser.parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=False)
    session = requests.Session()
    session.trust_env = False

    def control(endpoint, **body):
        r = session.post(args.base_url + endpoint, json=body, timeout=180)
        r.raise_for_status()
        return r.text

    def wave(phase):
        bodies = [dict(model=args.model,
                       prompt=make_prompt(args.input_len, 20260912, i),
                       max_tokens=1, temperature=0, ignore_eos=True,
                       cache_salt=f'ttft-profile-{phase}-{i}')
                  for i in range(args.concurrency)]

        def request(body):
            with requests.Session() as client:
                client.trust_env = False
                started = time.monotonic()
                r = client.post(args.base_url + '/v1/completions', json=body,
                                timeout=180)
                r.raise_for_status()
                elapsed = time.monotonic() - started
                usage = r.json()['usage']
                if (usage['prompt_tokens'] != args.input_len
                        or usage['completion_tokens'] != 1):
                    raise RuntimeError(f'Unexpected token lengths: {usage}')
                return dict(elapsed_s=elapsed, usage=usage)

        started = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(args.concurrency) as pool:
            rows = list(pool.map(request, bodies))
        data = dict(phase=phase, wall_s=time.monotonic() - started, requests=rows)
        (root / f'{phase}.json').write_text(json.dumps(data, indent=2) + '\n')
        print(json.dumps(dict(phase=phase, wall_s=data['wall_s'])), flush=True)

    (root / 'config.json').write_text(json.dumps(vars(args), indent=2) + '\n')
    wave('warmup')
    control('/start_profile', output_dir=str(root), activities=['CPU', 'GPU'],
            with_stack=False, record_shapes=False, merge_profiles=False)
    try:
        wave('profiled')
    finally:
        control('/stop_profile')
    print('Trace collection complete:', root, flush=True)


if __name__ == '__main__':
    main()
