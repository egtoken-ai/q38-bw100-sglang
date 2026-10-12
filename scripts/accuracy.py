#!/usr/bin/env python3
"""OpenCompass / VLMEvalKit accuracy suite for an OpenAI-compatible SGLang API.

prepare/report need only the Python standard library. run uses the toolkits'
Python APIs (official prompts and scorers), with shared HTTP/cache/report glue.
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import csv
import fcntl
import hashlib
import importlib.metadata
import io
import json
import math
import mimetypes
import os
from pathlib import Path
import pickle
import re
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OC_REV = '1543bd270b6fbabb1a118307f78d4144e8296467'
VLM_REV = '21e52b8df8ce1f628e93c82de547edb2c985f6ac'
SPECS = {
    'gsm8k': dict(name='GSM8K', baseline=97.0, threshold=95.0, count=1319,
                  protocol='OpenCompass gsm8k_gen_1d7fe4: 4-shot CoT, exact numeric match'),
    'livecodebench': dict(name='LiveCodeBench v6', baseline=90.3, threshold=88.3, count=175,
                         protocol='test6.jsonl only; Python; zero-shot; n=1; pass@1'),
    'realworldqa': dict(name='RealWorldQA', baseline=85.9, threshold=83.9, count=765,
                       protocol='VLMEvalKit RealWorldQA.tsv; zero-shot; exact_matching'),
    'mathvision': dict(name='MathVision (Without CI)', baseline=90.0, threshold=88.0, count=304,
                      protocol='MathVision_MINI / testmini; zero-shot; no code interpreter'),
}
SOURCES = {
    'gsm8k': [('gsm8k/test.jsonl', 'https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl', None)],
    'livecodebench': [('livecodebench/test6.jsonl', 'https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/0fe84c3912ea0c4d4a78037083943e8f0c4dd505/test6.jsonl',
                      ('sha256', 'bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5'))],
    'realworldqa': [('vlm/RealWorldQA.tsv', 'https://opencompass.openxlab.space/utils/VLMEval/RealWorldQA.tsv',
                    ('md5', '4de008f55dc4fd008ca9e15321dc44b7'))],
    'mathvision': [('vlm/MathVision_MINI.tsv', 'https://opencompass.openxlab.space/utils/VLMEval/MathVision_MINI.tsv',
                   ('md5', '060fe4fa5d868987ce179307bd5f8a33'))],
}


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, 'item'):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def table_rows(path):
    csv.field_size_limit(sys.maxsize)
    with Path(path).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def validate_data(key, path):
    rows = table_rows(path) if '.tsv' in path.suffixes else read_jsonl(path)
    expected = SPECS[key]['count']
    if len(rows) != expected:
        raise ValueError(f'{path}: expected {expected} rows, found {len(rows)}')
    fields = {
        'gsm8k': {'question', 'answer'},
        'livecodebench': {'question_id', 'question_content', 'starter_code', 'public_test_cases', 'private_test_cases', 'metadata'},
        'realworldqa': {'index', 'question', 'answer', 'image', 'A', 'B'},
        'mathvision': {'index', 'question', 'answer', 'image', 'choices', 'category'},
    }[key]
    for i, row in enumerate(rows):
        if not fields <= row.keys() or any(row[f] is None for f in fields):
            raise ValueError(f'{path}: invalid schema at row {i}')
    id_field = 'question_id' if key == 'livecodebench' else 'index'
    if key != 'gsm8k' and len({str(r[id_field]) for r in rows}) != expected:
        raise ValueError(f'{path}: duplicate IDs')
    return len(rows)


def open_url(request, *, timeout, insecure=False):
    context = ssl.create_default_context()
    if insecure:
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
    url = request.full_url if isinstance(request, urllib.request.Request) else request
    host = urllib.parse.urlsplit(url).hostname
    proxy = urllib.request.ProxyHandler({}) if host in ('localhost', '127.0.0.1', '::1') else urllib.request.ProxyHandler()
    opener = urllib.request.build_opener(proxy, urllib.request.HTTPSHandler(context=context))
    return opener.open(request, timeout=timeout)


def prepare(args):
    manifest = {}
    for key in args.benchmarks:
        for relative, source, checksum in SOURCES[key]:
            if key in ('realworldqa', 'mathvision'):
                source = args.vlm_data_base_url.rstrip('/') + '/' + Path(relative).name
            target = args.data_dir / relative
            if not target.exists():
                if args.offline:
                    raise FileNotFoundError(f'offline: missing {target}')
                target.parent.mkdir(parents=True, exist_ok=True)
                partial = target.with_suffix(target.suffix + '.part')
                print(f'[download] {source} -> {target}', flush=True)
                request = urllib.request.Request(source, headers={'User-Agent': 'q38-opencompass-accuracy/1'})
                insecure = args.insecure_dataset_download and key in ('realworldqa', 'mathvision')
                with open_url(request, timeout=120, insecure=insecure) as response, partial.open('wb') as out:
                    while chunk := response.read(1024 * 1024):
                        out.write(chunk)
                if checksum and digest(partial, checksum[0]) != checksum[1]:
                    raise ValueError(f'checksum mismatch: {partial}; final cache was not replaced')
                # Validate before publishing a complete cache entry.
                validate_data(key, partial)
                partial.replace(target)
            else:
                print(f'[cache] {target}', flush=True)
            if checksum and digest(target, checksum[0]) != checksum[1]:
                raise ValueError(f'checksum mismatch: {target}; move the invalid file aside before retrying')
            count = validate_data(key, target)
            manifest[key] = dict(path=str(target), source=source, sha256=digest(target), count=count)
    previous = args.data_dir / 'manifest.json'
    combined = json.loads(previous.read_text()) if previous.exists() else {}
    combined.update(manifest)
    write_json(previous, combined)
    return manifest


def final_text(text):
    """Only score the final answer; never substitute reasoning_content for it."""
    return text.rsplit('</think>', 1)[-1].strip()


def extract_math_answer(text):
    text = final_text(text)
    # Balanced braces are necessary for e.g. \\boxed{\\frac{1}{2}}.
    start = text.rfind('\\boxed{')
    if start >= 0:
        pos, depth = start + len('\\boxed{'), 1
        for end in range(pos, len(text)):
            if text[end] == '{':
                depth += 1
            elif text[end] == '}':
                depth -= 1
            if depth == 0:
                return text[pos:end].strip()
    matches = re.findall(r'(?:final answer|the answer)\s*(?:is\s*[:=]?|:|=)\s*(.+)', text, re.I)
    candidate = matches[-1] if matches else text.splitlines()[-1] if text else ''
    return candidate.strip().strip('$').rstrip('.').strip()


def request_json(url, payload=None, key='EMPTY', timeout=60):
    body = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=body, headers={
        'Content-Type': 'application/json', 'Authorization': f'Bearer {key}',
    })
    # VLMEvalKit changes the global SSL default; explicitly keep verification.
    with open_url(req, timeout=timeout) as response:
        return json.load(response)


def generate(args, messages, *, model=None, base_url=None, api_key=None, max_tokens=None, temperature=None):
    payload = dict(model=model or args.model, messages=messages, temperature=args.temperature if temperature is None else temperature,
                   top_p=1.0, max_tokens=max_tokens or args.max_tokens, stream=False, seed=args.seed)
    if model is None:  # Judge endpoints need not support the target's SGLang extensions.
        payload.update(args.extra_body)
    last_error = None
    for attempt in range(args.retries + 1):
        try:
            raw = request_json((base_url or args.base_url) + '/chat/completions', payload,
                               api_key or os.environ.get('ACCURACY_API_KEY', 'EMPTY'), args.timeout)
            choice = raw['choices'][0]
            content = choice['message'].get('content')
            if not isinstance(content, str) or not final_text(content):
                raise ValueError('API returned no final text content')
            if choice.get('finish_reason') != 'stop':
                return dict(status='invalid', error=f"finish_reason={choice.get('finish_reason')}", response=raw)
            return dict(status='ok', prediction=final_text(content), response=raw)
        except urllib.error.HTTPError as exc:
            last_error = f'HTTP {exc.code}: {exc.reason}'
            exc.close()
            if exc.code not in (408, 429, 500, 502, 503, 504):
                break
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError, IndexError) as exc:
            last_error = str(exc)
        if attempt < args.retries:
            time.sleep(min(2 ** attempt, 10))
    return dict(status='error', error=last_error)


def toolkit_versions(keys):
    result = {}
    packages = []
    if set(keys) & {'gsm8k', 'livecodebench'}:
        packages.append(('opencompass', OC_REV))
    if set(keys) & {'realworldqa', 'mathvision'}:
        packages.append(('vlmeval', VLM_REV))
    for name, revision in packages:
        dist = importlib.metadata.distribution(name)
        direct = json.loads(dist.read_text('direct_url.json') or '{}')
        actual = direct.get('vcs_info', {}).get('commit_id')
        if actual != revision:
            raise RuntimeError(f'{name}: install requirements-accuracy.txt; expected commit {revision}, found {actual}')
        result[name] = dict(version=dist.version, commit=actual)
    return result


def load_benchmark(key, args):
    source = args.data_dir / SOURCES[key][0][0]
    if key == 'gsm8k':
        from opencompass.configs.datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_infer_cfg
        template = gsm8k_infer_cfg['prompt_template']['template']['round']
        assert len(template) == 9, 'Expected the pinned 4-shot GSM8K prompt'
        rows = read_jsonl(source)

        def prompt(i):
            return [{'role': {'HUMAN': 'user', 'BOT': 'assistant'}[m['role']],
                     'content': m['prompt'].replace('{question}', rows[i]['question'])} for m in template]
        return rows, prompt, None
    if key == 'livecodebench':
        if sys.platform != 'linux':
            raise RuntimeError('LiveCodeBench code execution requires a Linux evaluation client')
        from opencompass.datasets.livecodebench.testing_util import RuntimeModule
        if RuntimeModule is None:
            raise RuntimeError('LiveCodeBench requires pyext==0.7 (use Python 3.10)')
        from opencompass.datasets.livecodebench.prompts import CodeGenerationPromptConstants as C
        rows = read_jsonl(source)

        def prompt(i):
            row = rows[i]
            if row['starter_code']:
                formatting = f'### Format: {C.FORMATTING_MESSAGE_WITH_STARTER_CODE}\n```python\n{row["starter_code"]}\n```\n\n'
            else:
                formatting = f'### Format: {C.FORMATTING_WITHOUT_STARTER_CODE}\n```python\n# YOUR CODE HERE\n```\n\n'
            return [dict(role='user', content=f'### Question:\n{row["question_content"]}\n\n{formatting}### Answer: (use the provided format with backticks)\n\n')]
        return rows, prompt, None
    os.environ['LMUData'] = str(args.data_dir / 'vlm')
    import pandas as pd
    from vlmeval.dataset.image_mcq import ImageMCQDataset
    from vlmeval.dataset.image_vqa import MathVision
    if key == 'mathvision':
        from vlmeval.dataset.utils.mathv import post_check  # verify math parser dependencies before inference
    cls = ImageMCQDataset if key == 'realworldqa' else MathVision

    class CachedDataset(cls):
        def load_data(self, dataset):
            # Already validated by prepare(); never trigger an implicit download.
            return pd.read_csv(source, sep='\t', keep_default_na=False, na_values=[''])

    dataset = CachedDataset(dataset='RealWorldQA' if key == 'realworldqa' else 'MathVision_MINI')
    if len(dataset) != SPECS[key]['count']:
        raise ValueError('VLMEvalKit dropped rows (possibly missing images)')
    rows = dataset.data.to_dict('records')

    def prompt(i):
        content = []
        for part in dataset.build_prompt(i):
            if part['type'] == 'image':
                path = Path(part['value'])
                mime = mimetypes.guess_type(path.name)[0] or 'image/jpeg'
                data = base64.b64encode(path.read_bytes()).decode()
                content.append(dict(type='image_url', image_url={'url': f'data:{mime};base64,{data}'}))
            elif part['type'] == 'text':
                text = part['value']
                if key == 'mathvision':
                    text += '\nReason step by step. Put the final answer in \\boxed{}. For multiple-choice questions, box only the option letter.'
                content.append(dict(type='text', text=text))
            else:
                raise ValueError(f'Unsupported prompt part: {part["type"]}')
        if not any(p['type'] == 'image_url' for p in content):
            raise ValueError('Vision benchmark has no image input')
        return [dict(role='user', content=content)]
    return rows, prompt, dataset


class StringOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError('LCB private test payload must not reference globals')


def lcb_sample(row):
    public = json.loads(row['public_test_cases'])
    try:
        private = json.loads(row['private_test_cases'])
    except json.JSONDecodeError:
        packed = zlib.decompress(base64.b64decode(row['private_test_cases']))
        private = json.loads(StringOnlyUnpickler(io.BytesIO(packed)).load())
    cases = public + private
    if not public or not private:
        raise ValueError(f'{row["question_id"]}: missing public/private tests')
    return {'input_output': json.dumps(dict(inputs=[c['input'] for c in cases],
                                            outputs=[c['output'] for c in cases],
                                            fn_name=json.loads(row['metadata']).get('func_name')))}


def score_benchmark(key, rows, predictions, dataset, args, out):
    if key == 'gsm8k':
        from opencompass.datasets.gsm8k import Gsm8kEvaluator, gsm8k_postprocess, gsm8k_dataset_postprocess
        result = Gsm8kEvaluator().score([gsm8k_postprocess(p) for p in predictions],
                                      [gsm8k_dataset_postprocess(r['answer']) for r in rows])
        write_json(out / 'evaluation.json', result)
        return float(result['accuracy'])
    if key == 'livecodebench':
        if sys.platform != 'linux':
            raise RuntimeError('LiveCodeBench code execution requires the Linux evaluation client')
        import multiprocessing
        # Upstream evaluator uses nested process targets and therefore needs fork.
        multiprocessing.set_start_method('fork', force=True)
        from opencompass.datasets.livecodebench.evaluator import codegen_metrics
        from opencompass.datasets.livecodebench.extract_utils import extract_code_generation_v2
        from opencompass.datasets.livecodebench.testing_util import RuntimeModule  # fail before evaluation if pyext is missing
        assert RuntimeModule is not None
        metrics, results, metadata = codegen_metrics(
            [lcb_sample(r) for r in rows], [[extract_code_generation_v2(p)] for p in predictions],
            k_list=[1], num_process_evaluate=args.eval_workers, timeout=args.code_timeout)
        if len(results) != len(rows):
            raise RuntimeError('LiveCodeBench evaluator returned incomplete results')
        write_json(out / 'evaluation.json', dict(metrics=metrics, results=results, metadata=metadata))
        return float(metrics['pass@1'])  # OpenCompass already returns 0..100.
    import pandas as pd
    data = pd.DataFrame(rows).drop(columns=['image'], errors='ignore')
    data['prediction'] = predictions
    # Excel cells truncate at 32767 characters, which can remove a long-CoT
    # answer. VLMEvalKit supports TSV and preserves the entire response there.
    evaluation_file = out / 'predictions.tsv'
    data.to_csv(evaluation_file, sep='\t', index=False)
    if key == 'realworldqa':
        result = dataset.evaluate(str(evaluation_file), model='exact_matching', nproc=args.eval_workers)
        result.to_csv(out / 'evaluation.csv', index=False)
        if len(result) != 1 or 'Overall' not in result:
            raise ValueError('Unexpected RealWorldQA score schema')
        return float(result.iloc[0]['Overall']) * 100.0  # VLMEvalKit MCQ: 0..1.
    from vlmeval.dataset.utils.mathv import MATH_V_acc, MATH_V_auxeval, post_check
    records = []

    class Judge:
        def generate(self, prompt, temperature=0):
            reply = generate(args, [dict(role='user', content=prompt)], model=args.judge_model,
                             base_url=args.judge_base_url, api_key=os.environ.get('ACCURACY_JUDGE_API_KEY', 'EMPTY'),
                             max_tokens=512, temperature=temperature)
            if reply['status'] != 'ok':
                raise RuntimeError(f'MathVision judge failed: {reply.get("error")}')
            return reply['prediction']

    for row in data.to_dict('records'):
        if args.math_judge == 'api':
            graded = MATH_V_auxeval(Judge(), row)
        else:
            graded = dict(res=extract_math_answer(row['prediction']), log='local boxed/final-line extraction')
        row.update(graded)
        row['correct'] = bool(post_check(row, prefetch=False))
        records.append(row)
    # Persist answers and per-question decisions, without serializing images.
    write_json(out / 'evaluation.json', records)
    scored = out / 'mathvision_scored.tsv'
    pd.DataFrame(records).to_csv(scored, sep='\t', index=False)
    result = MATH_V_acc(str(scored))
    result.to_csv(out / 'evaluation.csv', index=False)
    overall = result[result['Subject'] == 'Overall']
    if len(overall) != 1 or int(overall.iloc[0]['tot']) != len(rows):
        raise ValueError('Unexpected MathVision score schema/count')
    return float(overall.iloc[0]['acc'])


def load_cache(path):
    if not path.exists():
        return {}
    # An interrupted append may leave a partial final line. Only that tail is
    # truncated; corruption in any completed line is an error.
    with path.open('rb+') as f:
        payload = f.read()
        end = payload.rfind(b'\n') + 1
        if end < len(payload):
            f.truncate(end)
    records = [json.loads(line) for line in payload[:end].splitlines() if line]
    return {str(r['id']): r for r in records if r['status'] == 'ok'}


def run_one(key, args, manifest, versions):
    out = args.output_dir / key
    out.mkdir(parents=True, exist_ok=True)
    config = dict(benchmark=key, data=manifest[key], model=args.model, base_url=args.base_url,
                  model_revision=args.model_revision, temperature=args.temperature, max_tokens=args.max_tokens,
                  seed=args.seed, extra_body=args.extra_body, limit=args.limit, toolkit_versions=versions,
                  math_judge=args.math_judge, judge_model=args.judge_model, judge_base_url=args.judge_base_url,
                  code_timeout=args.code_timeout, script_sha256=digest(Path(__file__)))
    config_path = out / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError(f'{out}: configuration changed; use a new --output-dir to avoid mixing predictions')
    if not config_path.exists() and (out / 'predictions.jsonl').exists():
        raise ValueError('Predictions without a matching config cannot be resumed')
    write_json(config_path, config)
    rows, prompt, dataset = load_benchmark(key, args)
    if args.limit:
        rows = rows[:args.limit]
    ids = [str(r.get('question_id', r.get('index', i))) for i, r in enumerate(rows)]
    cache_path = out / 'predictions.jsonl'
    cached = load_cache(cache_path)
    pending = [(i, sid) for i, sid in enumerate(ids) if sid not in cached]
    print(f'[{key}] total={len(rows)} cached={len(ids) - len(pending)} pending={len(pending)}', flush=True)

    def infer(item):
        i, sid = item
        started = time.monotonic()
        try:
            messages = prompt(i)
            response = generate(args, messages)
            prompt_hash = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()
        except Exception as exc:
            response, prompt_hash = dict(status='error', error=str(exc)), None
        return dict(id=sid, row=i, prompt_sha256=prompt_hash, seconds=time.monotonic() - started,
                    timestamp=utcnow(), **response)

    with cache_path.open('a', encoding='utf-8') as f, concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(infer, item) for item in pending]
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            record = future.result()
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            f.flush()
            if record['status'] == 'ok':
                cached[record['id']] = record
            print(f'[{key}] {done}/{len(pending)} id={record["id"]} {record["status"]}', flush=True)
    missing = [sid for sid in ids if sid not in cached]
    if missing:
        raise RuntimeError(f'{len(missing)}/{len(ids)} invalid or failed responses; rerun to retry. IDs: {missing[:10]}')
    score = score_benchmark(key, rows, [cached[sid]['prediction'] for sid in ids], dataset, args, out)
    if not math.isfinite(score) or not 0 <= score <= 100:
        raise ValueError(f'Invalid score {score}')
    result = dict(benchmark=key, score=score, evaluated=len(rows), status='scored',
                  smoke=bool(args.limit),
                  protocol=SPECS[key]['protocol'], math_judge=args.math_judge if key == 'mathvision' else None)
    return result


def suite_config(args):
    names = ('model', 'base_url', 'model_revision', 'temperature', 'max_tokens', 'seed',
             'extra_body', 'limit', 'math_judge', 'judge_model', 'judge_base_url', 'code_timeout')
    config = {name: getattr(args, name) for name in names}
    config.update(data_dir=str(args.data_dir), script_sha256=digest(Path(__file__)),
                  opencompass_revision=OC_REV, vlmeval_revision=VLM_REV)
    return config


def admission(key, result):
    spec = SPECS[key]
    score = result.get('score')
    status = result.get('status', 'missing')
    if status == 'scored':
        if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 100:
            status = 'ERROR'
        elif result.get('smoke') or result.get('evaluated') != spec['count']:
            status = 'SMOKE'
        else:
            status = 'PASS' if score >= spec['threshold'] and spec['baseline'] - score <= 2.0 + 1e-12 else 'FAIL'
    else:
        status = status.upper()
    return dict(benchmark=spec['name'], baseline=spec['baseline'], threshold=spec['threshold'],
                score=score, drop_pp=None if score is None else spec['baseline'] - score,
                evaluated=result.get('evaluated', 0), expected=spec['count'], status=status,
                error=result.get('error', ''), protocol=result.get('protocol', spec['protocol']),
                math_judge=result.get('math_judge'))


def report(output_dir, results):
    rows = [admission(key, results.get(key, {})) for key in SPECS]
    passed = all(r['status'] == 'PASS' for r in rows)
    overall = 'PASS' if passed else 'FAIL' if all(r['status'] in ('PASS', 'FAIL') for r in rows) else 'INCOMPLETE'
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / 'summary.json', dict(created_at=utcnow(), overall=overall, results=rows))
    with (output_dir / 'summary.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['| Benchmark | 基线 % | 阈值 % | 实测 % | 降幅 pp | 完成题数 | 判定 |',
             '|---|---:|---:|---:|---:|---:|---|']
    for row in rows:
        score = '—' if row['score'] is None else f'{row["score"]:.4f}'
        drop = '—' if row['drop_pp'] is None else f'{row["drop_pp"]:.4f}'
        lines.append(f'| {row["benchmark"]} | {row["baseline"]:.1f} | {row["threshold"]:.1f} | {score} | {drop} | {row["evaluated"]}/{row["expected"]} | {row["status"]} |')
    lines += ['', f'整体判定：**{overall}**', '',
              '分数以百分数记录，降幅单位为百分点；使用未舍入分数判定。负降幅表示提升。',
              '本报告验证所选协议下的数值阈值；官方基线的完整生成/判分配置尚未提供，不能证明与官方分数完全同口径。',
              'MathVision local 使用规则提取 + VLMEvalKit 数学匹配；api 使用 VLMEvalKit LLM 答案提取。二者须分别比较。', '']
    for row in rows:
        if row['error']:
            lines.append(f'- {row["benchmark"]}: {row["error"]}')
    content = '\n'.join(lines) + '\n'
    (output_dir / 'summary.md').write_text(content, encoding='utf-8')
    print('\n' + content)
    return 0 if passed else 1 if overall == 'FAIL' else 2


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('command', choices=['prepare', 'run', 'report'])
    p.add_argument('--benchmarks', nargs='+', choices=list(SPECS), default=list(SPECS))
    p.add_argument('--data-dir', type=Path, default=ROOT / 'data/accuracy')
    p.add_argument('--output-dir', type=Path, default=ROOT / 'result/accuracy/default')
    p.add_argument('--offline', action='store_true', help='Require local data; never download')
    p.add_argument('--vlm-data-base-url', default='https://opencompass.openxlab.space/utils/VLMEval', help='Mirror containing the identical visual TSV files')
    p.add_argument('--insecure-dataset-download', action='store_true', help='Skip TLS verification ONLY for the two checksum-validated visual TSV downloads')
    p.add_argument('--base-url', default=os.environ.get('BASE_URL', 'http://127.0.0.1:18580/v1'))
    p.add_argument('--model', default=os.environ.get('SERVED_MODEL_NAME', 'Qwen3.8-27B-Channel-INT8-w8a8'))
    p.add_argument('--model-revision', default='', help='Checkpoint/server configuration ID for safe resume')
    p.add_argument('--concurrency', type=int, default=4)
    p.add_argument('--max-tokens', type=int, default=32768)
    p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--seed', type=int, default=20261012)
    p.add_argument('--timeout', type=int, default=3600)
    p.add_argument('--retries', type=int, default=2)
    p.add_argument('--eval-workers', type=int, default=4)
    p.add_argument('--code-timeout', type=int, default=6, help='LiveCodeBench timeout per test case, seconds')
    p.add_argument('--limit', type=int, default=0, help='Smoke test first N questions; cannot receive PASS')
    p.add_argument('--extra-body', default='{}', help='JSON, e.g. {"chat_template_kwargs":{"enable_thinking":true}}')
    p.add_argument('--math-judge', choices=['local', 'api'], default='local')
    p.add_argument('--judge-base-url', default='', help='Independent answer-extraction API, including /v1')
    p.add_argument('--judge-model', default='')
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    args.data_dir, args.output_dir = args.data_dir.resolve(), args.output_dir.resolve()
    args.base_url = args.base_url.rstrip('/')
    if not args.base_url.endswith('/v1'):
        args.base_url += '/v1'
    args.judge_base_url = args.judge_base_url.rstrip('/')
    if min(args.concurrency, args.max_tokens, args.timeout, args.eval_workers, args.code_timeout) < 1 or min(args.retries, args.limit) < 0:
        p.error('Counts/timeouts must be positive; retries/limit must be nonnegative')
    if not math.isfinite(args.temperature) or not 0 <= args.temperature <= 2:
        p.error('temperature must be finite and between 0 and 2')
    try:
        args.extra_body = json.loads(args.extra_body)
    except json.JSONDecodeError as exc:
        p.error(str(exc))
    if not isinstance(args.extra_body, dict) or set(args.extra_body) - {'chat_template_kwargs', 'top_k', 'min_p', 'repetition_penalty'}:
        p.error('extra-body supports only chat_template_kwargs, top_k, min_p, repetition_penalty; tools are disabled')
    if args.math_judge == 'api' and 'mathvision' in args.benchmarks and args.command == 'run' and not (args.judge_base_url and args.judge_model):
        p.error('--math-judge api requires --judge-base-url and --judge-model')
    if args.command == 'prepare':
        prepare(args)
        return 0
    if args.command == 'report':
        results = {}
        for key in SPECS:
            file = args.output_dir / key / 'result.json'
            if file.exists():
                results[key] = json.loads(file.read_text())
        return report(args.output_dir, results)
    results = {}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # The open file must live until main returns. flock is released on crashes.
    run_lock = (args.output_dir / '.run.lock').open('a')
    try:
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError(f'Another evaluation is using {args.output_dir}')
    suite_path = args.output_dir / 'suite_config.json'
    config = suite_config(args)
    if suite_path.exists() and json.loads(suite_path.read_text()) != config:
        raise ValueError('Suite configuration changed; select a new --output-dir')
    write_json(suite_path, config)
    for key in SPECS:
        file = args.output_dir / key / 'result.json'
        if file.exists() and key not in args.benchmarks:
            results[key] = json.loads(file.read_text())
    for key in args.benchmarks:
        write_json(args.output_dir / key / 'result.json', dict(status='running'))
    # Never let the old report survive an interrupted new run as a current PASS.
    report(args.output_dir, results)
    write_json(args.output_dir / 'run.json', dict(started_at=utcnow(), command=sys.argv,
                                                model=args.model, base_url=args.base_url))
    environment = sorted(f'{d.metadata["Name"]}=={d.version}' for d in importlib.metadata.distributions())
    (args.output_dir / 'environment.txt').write_text('\n'.join(environment) + '\n', encoding='utf-8')
    try:
        models = request_json(args.base_url + '/models', key=os.environ.get('ACCURACY_API_KEY', 'EMPTY'))
        if args.model not in {m['id'] for m in models['data']}:
            raise ValueError(f'Server does not list model {args.model}')
        for key in args.benchmarks:
            try:
                selected = argparse.Namespace(**{**vars(args), 'benchmarks': [key]})
                manifest = prepare(selected)
                versions = toolkit_versions([key])
                results[key] = run_one(key, args, manifest, versions)
                write_json(args.output_dir / key / 'result.json', results[key])
            except Exception as exc:
                results[key] = dict(status='error', error=f'{type(exc).__name__}: {exc}')
                write_json(args.output_dir / key / 'result.json', results[key])
                print(f'[{key}] ERROR: {exc}', file=sys.stderr, flush=True)
            report(args.output_dir, results)
    except Exception as exc:
        for key in args.benchmarks:
            results.setdefault(key, dict(status='error', error=f'{type(exc).__name__}: {exc}'))
            write_json(args.output_dir / key / 'result.json', results[key])
    return report(args.output_dir, results)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('Interrupted; valid per-question predictions are retained.', file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}', file=sys.stderr)
        sys.exit(2)
