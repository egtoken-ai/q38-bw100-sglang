#!/usr/bin/env python3
"""Compare logprobs of identical continuations; this is not an accuracy benchmark.

Run once on native and again on a candidate with --reference native-tf.json.
Both use --snapshot native-model-outputs.json. No greedy candidate continuation
is substituted. Input-logprob prefill has a different execution shape from
ordinary decode; compare native teacher forcing against candidate teacher forcing.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import requests

from benchmark import make_prompt


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()


def aligned_logprobs(meta, input_ids, prompt_length):
    """SGLang returns a leading None for logprob_start_len itself."""
    start = prompt_length - 1
    rows = meta.get('input_token_logprobs')
    expected = input_ids[start:]
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError(f'Input-logprob count mismatch: expected {len(expected)}, '
                         f'got {len(rows) if isinstance(rows, list) else None}')
    if [row[1] for row in rows] != expected:
        raise ValueError('Input-logprob token IDs do not align with requested suffix')
    if rows[0][0] is not None:
        raise ValueError('Expected leading None at logprob_start_len; API semantics differ')
    values = [row[0] for row in rows[1:]]
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
        raise ValueError('Missing or nonfinite continuation logprobs')
    if meta.get('prompt_tokens') != len(input_ids):
        raise ValueError('Server prompt length differs from submitted IDs')
    if meta.get('completion_tokens') != 1:
        raise ValueError('Expected exactly one ignored completion token')
    return values


def comparison(rows, reference):
    old_rows = {r['name']: r for r in reference['cases']}
    if set(old_rows) != {r['name'] for r in rows}:
        raise ValueError('Reference case set differs')
    result = []
    for row in rows:
        old = old_rows[row['name']]
        for key in ('input_ids_sha256', 'continuation_ids', 'prompt_length', 'logprob_start_len'):
            if old[key] != row[key]:
                raise ValueError(f"{row['name']}: reference mismatch in {key}")
        current, baseline = row['continuation_logprobs'], old['continuation_logprobs']
        if len(current) != len(baseline) or not current:
            raise ValueError('Invalid reference logprob count')
        delta = [x - y for x, y in zip(current, baseline)]
        result.append(dict(name=row['name'], tokens=len(delta),
                           max_abs_logprob_difference=max(abs(v) for v in delta),
                           mean_abs_logprob_difference=sum(abs(v) for v in delta) / len(delta),
                           mean_signed_logprob_difference=sum(delta) / len(delta),
                           token_logprob_differences=delta))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url', default='http://127.0.0.1:18580')
    p.add_argument('--snapshot', required=True, help='Original native greedy snapshot')
    p.add_argument('--reference', help='Native teacher-forcing output, not greedy snapshot')
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=20260912)
    p.add_argument('--timeout', type=float, default=240)
    p.add_argument('--dry-run', action='store_true', help='Validate plan without HTTP requests')
    args = p.parse_args()
    snapshot = json.loads(Path(args.snapshot).read_text())
    rows = []
    client = requests.Session()
    client.trust_env = False
    for length in (4096, 65536):
        name = f'random-{length}'
        matches = [r for r in snapshot['cases'] if r['name'] == name]
        if len(matches) != 1:
            raise ValueError(f'Missing or repeated snapshot case {name}')
        original = matches[0]
        continuation = original['output_ids']
        if original['prompt_tokens'] != length or len(continuation) != 16:
            raise ValueError('Expected original 16-token continuation at fixed prompt length')
        input_ids = make_prompt(length, args.seed, 0) + continuation
        row = dict(name=name, prompt_length=length, continuation_ids=continuation,
                   submitted_input_tokens=len(input_ids), logprob_start_len=length - 1,
                   input_ids_sha256=fingerprint(input_ids))
        if not args.dry_run:
            response = client.post(args.base_url.rstrip('/') + '/generate', json=dict(
                input_ids=input_ids, return_logprob=True, logprob_start_len=length - 1,
                sampling_params=dict(temperature=0, max_new_tokens=1, ignore_eos=True)),
                timeout=args.timeout)
            response.raise_for_status()
            data = response.json()
            row['continuation_logprobs'] = aligned_logprobs(data['meta_info'], input_ids, length)
            row['input_token_logprobs'] = data['meta_info']['input_token_logprobs']
            row['server_meta'] = {k: data['meta_info'].get(k) for k in
                                  ('prompt_tokens', 'completion_tokens', 'cached_tokens', 'id')}
        rows.append(row)
    result = dict(scope='Same-history numerical diagnostic only; no accuracy pass threshold',
                  mode='dry-run' if args.dry_run else 'teacher-forcing', seed=args.seed,
                  base_url=args.base_url, snapshot_sha256=fingerprint(snapshot),
                  prompt_generator_sha256=hashlib.sha256(
                      Path(__file__).with_name('benchmark.py').read_bytes()).hexdigest(), cases=rows)
    if args.reference and not args.dry_run:
        reference = json.loads(Path(args.reference).read_text())
        if reference.get('mode') != 'teacher-forcing':
            raise ValueError('Reference must be teacher-forcing output')
        result['comparison'] = comparison(rows, reference)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(output=str(target), mode=result['mode'],
                          comparison=result.get('comparison'),
                          cases=[dict(name=r['name'], input_tokens=r['submitted_input_tokens'],
                                      scored_tokens=len(r['continuation_ids'])) for r in rows]),
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
