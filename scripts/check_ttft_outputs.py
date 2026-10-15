#!/usr/bin/env python3
"""Small deterministic model regression snapshots, separate from performance."""
import argparse
import json
from pathlib import Path

import requests

from benchmark import make_prompt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url', default='http://127.0.0.1:18580')
    p.add_argument('--output', required=True)
    p.add_argument('--reference')
    args = p.parse_args()
    client = requests.Session()
    client.trust_env = False
    rows = []
    questions = [
        ('add', '只回答数字：17+25等于多少？', '42'),
        ('multiply', '只回答数字：12乘以13等于多少？', '156'),
        ('subtract', '只回答数字：100减去37等于多少？', '63'),
        ('sort', '把3、1、2从小到大排序，只输出逗号分隔的三个数字。', '1,2,3'),
    ]
    for name, question, expected in questions:
        body = dict(model='Qwen3.8-27B-Channel-INT8-w8a8',
                    messages=[dict(role='user', content=question)],
                    temperature=0, max_tokens=64,
                    chat_template_kwargs=dict(enable_thinking=False))
        r = client.post(args.base_url + '/v1/chat/completions', json=body, timeout=120)
        r.raise_for_status()
        data = r.json()
        answer = data['choices'][0]['message']['content'] or ''
        normalized = answer.strip().replace('，', ',').replace(' ', '')
        rows.append(dict(name=name, output=answer, expected=expected,
                         pass_expected=normalized==expected, usage=data['usage']))
    for length in [4096, 65536]:
        body = dict(input_ids=make_prompt(length, 20260912, 0),
                    sampling_params=dict(temperature=0, max_new_tokens=16, ignore_eos=True),
                    return_logprob=True, logprob_start_len=-1, top_logprobs_num=5)
        r = client.post(args.base_url + '/generate', json=body, timeout=180)
        r.raise_for_status()
        data = r.json()
        meta = data['meta_info']
        output_ids = data.get('output_ids') or [x[1] for x in meta['output_token_logprobs']]
        if meta['prompt_tokens'] != length or len(output_ids) != 16:
            raise RuntimeError('Unexpected token lengths in model regression')
        rows.append(dict(name=f'random-{length}', output_ids=output_ids,
                         output_logprobs=meta.get('output_token_logprobs'),
                         prompt_tokens=meta['prompt_tokens'],
                         completion_tokens=meta['completion_tokens']))
    result = dict(scope='Smoke and deterministic token regression, not full accuracy evaluation',
                  cases=rows, all_expected_pass=all(r.get('pass_expected', True) for r in rows))
    if args.reference:
        reference = {r['name']: r for r in json.loads(Path(args.reference).read_text())['cases']}
        result['comparison'] = []
        for row in rows:
            old = reference[row['name']]
            key = 'output_ids' if 'output_ids' in row else 'output'
            result['comparison'].append(dict(name=row['name'], exact_output=row[key]==old[key]))
        result['all_outputs_match'] = all(r['exact_output'] for r in result['comparison'])
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k!='cases'}, ensure_ascii=False))
    if not result['all_expected_pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
