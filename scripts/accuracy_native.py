#!/usr/bin/env python3
"""Native CLI orchestration: data/config preparation, subprocesses, result audit.

Model requests, prompt rendering, retries, inference resumption and all scoring
are performed by upstream OpenCompass / VLMEvalKit commands.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import importlib.metadata
import json
import math
import os
from pathlib import Path
import pprint
import shlex
import shutil
import signal
import subprocess
import sys
import urllib.request

# Reuse only download/checksum/report utilities from the earlier API runner.
import accuracy as shared

ROOT = shared.ROOT
TEMPLATES = ROOT / 'config/accuracy_native'
VLM_RUN_SHA = '9ddff184e640dd3d66411d83d8cc1a811d294a6918182832c5132486a30d4586'
ALIASES = {'gsm8k': 'gsm8k', 'livecodebench': 'livecodebench_v6',
           'realworldqa': 'RealWorldQA', 'mathvision': 'MathVision_MINI'}


def fetch(url, path, offline, expected_sha=None):
    if not path.exists():
        if offline:
            raise FileNotFoundError(f'offline: missing {path}')
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(path.suffix + '.part')
        print(f'[download] {url}', flush=True)
        with shared.open_url(urllib.request.Request(url), timeout=120) as response, partial.open('wb') as output:
            shutil.copyfileobj(response, output)
        if expected_sha and shared.digest(partial) != expected_sha:
            raise ValueError(f'Unexpected SHA256 for {partial}')
        partial.replace(path)
    if expected_sha and shared.digest(path) != expected_sha:
        raise ValueError(f'Unexpected SHA256 for {path}')


def prepare(args):
    manifest = shared.prepare(args)
    if 'gsm8k' in args.benchmarks:
        train = args.data_dir / 'gsm8k/train.jsonl'
        fetch('https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl',
              train, args.offline)
        rows = shared.read_jsonl(train)
        if len(rows) != 7473 or any(not {'question', 'answer'} <= row.keys() for row in rows):
            raise ValueError('Expected the 7473-row GSM8K training file required by its native loader')
        manifest['gsm8k']['train_sha256'] = shared.digest(train)
    if 'livecodebench' in args.benchmarks:
        target = args.data_dir / 'native_oc/data/code_generation_lite/code_generation_lite.py'
        target.parent.mkdir(parents=True, exist_ok=True)
        content = (TEMPLATES / 'code_generation_lite.py').read_bytes()
        if not target.exists() or target.read_bytes() != content:
            target.write_bytes(content)
        manifest['livecodebench']['loader_sha256'] = shared.digest(target)
    if set(args.benchmarks) & {'mathvision', 'realworldqa'}:
        fetch(f'https://raw.githubusercontent.com/open-compass/VLMEvalKit/{shared.VLM_REV}/run.py',
              vlm_entrypoint(args), args.offline, VLM_RUN_SHA)
    return manifest


def vlm_entrypoint(args):
    return args.data_dir / 'native_tools' / shared.VLM_REV / 'run.py'


def settings(args):
    names = ('model', 'model_revision', 'base_url', 'temperature', 'max_tokens',
             'gsm8k_max_tokens', 'concurrency', 'eval_workers', 'code_timeout',
             'timeout', 'seed', 'tokenizer_path', 'extra_body', 'judge_model', 'judge_base_url')
    result = {name: getattr(args, name) for name in names}
    result.update(data_dir=str(args.data_dir), oc_cache=str(args.data_dir / 'native_oc'),
                  lcb_file=str(args.data_dir / 'livecodebench/test6.jsonl'),
                  gsm8k_path=str(args.data_dir / 'gsm8k'),
                  opencompass_commit=shared.OC_REV, vlmeval_commit=shared.VLM_REV,
                  implementation_sha256={str(p.relative_to(ROOT)): shared.digest(p) for p in
                                         [Path(__file__), ROOT / 'scripts/accuracy.py',
                                          TEMPLATES / 'opencompass.py.template', TEMPLATES / 'code_generation_lite.py']})
    return result


def generate_configs(args):
    out = args.output_dir / 'configs'
    out.mkdir(parents=True, exist_ok=True)
    configs = {}
    for key in args.benchmarks:
        if key in ('gsm8k', 'livecodebench'):
            cfg = {**settings(args), 'benchmark': key}
            path = out / f'{key}.py'
            text = (TEMPLATES / 'opencompass.py.template').read_text()
            path.write_text(text.replace('@@SETTINGS@@', pprint.pformat(cfg, sort_dicts=True)), encoding='utf-8')
        else:
            path = out / f'{key}.json'
            shared.write_json(path, dict(
                model={'q38': dict(
                    # Upstream generic vision API adapter, without model-specific custom prompts.
                    **args.extra_body, **{'class': 'GPT4V'}, model=args.model,
                    api_base=args.base_url + '/chat/completions',
                    temperature=args.temperature, top_p=1.0, seed=args.seed,
                    max_tokens=args.max_tokens, timeout=args.timeout,
                    retry=3, verbose=False, img_detail='high', img_size=-1,
                )},
                data={ALIASES[key]: {'class': 'ImageMCQDataset' if key == 'realworldqa' else 'MathVision',
                                     'dataset': ALIASES[key]}},
            ))
        configs[key] = path
    return configs


def oc_executable():
    sibling = Path(sys.executable).parent / 'opencompass'
    return str(sibling) if sibling.exists() else (shutil.which('opencompass') or 'opencompass')


def commands(args, key, config, mode):
    if key in ('gsm8k', 'livecodebench'):
        return [dict(stage=mode, argv=[oc_executable(), str(config), '--work-dir',
                                      str(args.output_dir / 'opencompass' / key),
                                      '--reuse', 'native', '--mode', mode])]
    base = [sys.executable, str(vlm_entrypoint(args)), '--config', str(config),
            '--work-dir', str(args.output_dir / 'vlmeval' / key),
            '--api-nproc', str(args.concurrency), '--reuse', '--reuse-aux', 'all']
    stages = ['infer', 'eval'] if mode == 'all' else [mode]
    result = []
    for stage in stages:
        cmd = base + ['--mode', stage]
        if key == 'realworldqa':
            cmd += ['--judge', 'exact_matching']
        elif stage == 'eval':
            cmd += ['--judge', args.judge_model or 'SET_JUDGE_MODEL',
                    '--judge-base-url', args.judge_base_url or 'SET_JUDGE_BASE_URL']
        result.append(dict(stage=stage, argv=cmd))
    return result


def save_commands(args, plans):
    shared.write_json(args.output_dir / 'commands.json', plans)
    exports = {
        'COMPASS_DATA_CACHE': str(args.data_dir / 'native_oc'),
        'ACCURACY_LCB_FILE': str(args.data_dir / 'livecodebench/test6.jsonl'),
        'LMUData': str(args.data_dir / 'vlm'), 'PRED_FORMAT': 'tsv', 'EVAL_FORMAT': 'csv',
    }
    lines = ['#!/usr/bin/env bash', 'set -euo pipefail', f'cd {shlex.quote(str(ROOT))}',
             'unset DATASET_SOURCE LOCAL_LLM RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE',
             'export NO_PROXY="${NO_PROXY:-},localhost,127.0.0.1,::1"',
             'export no_proxy="${no_proxy:-},localhost,127.0.0.1,::1"']
    lines += [f'export {name}={shlex.quote(value)}' for name, value in exports.items()]
    for key, plan in plans.items():
        for command in plan:
            credential = 'ACCURACY_JUDGE_API_KEY' if key == 'mathvision' and command['stage'] == 'eval' else 'ACCURACY_API_KEY'
            lines += ['', f'# {key}: {command["stage"]}',
                      f'OPENAI_API_KEY="${{{credential}:-EMPTY}}" ' + shlex.join(command['argv'])]
    (args.output_dir / 'commands.sh').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def child_env(args, stage, key):
    env = os.environ.copy()
    env['COMPASS_DATA_CACHE'] = str(args.data_dir / 'native_oc')
    env['ACCURACY_LCB_FILE'] = str(args.data_dir / 'livecodebench/test6.jsonl')
    env['LMUData'] = str(args.data_dir / 'vlm')
    env['PRED_FORMAT'] = 'tsv'
    env['EVAL_FORMAT'] = 'csv'
    env['TOKENIZERS_PARALLELISM'] = 'false'
    env['OPENAI_API_KEY'] = env.get('ACCURACY_JUDGE_API_KEY' if stage == 'eval' and key == 'mathvision'
                                    else 'ACCURACY_API_KEY', 'EMPTY')
    # Do not inherit distributed model-server settings into the CPU client.
    for name in ('DATASET_SOURCE', 'LOCAL_LLM', 'RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE'):
        env.pop(name, None)
    for name in ('NO_PROXY', 'no_proxy'):
        env[name] = ','.join(filter(None, [env.get(name, ''), 'localhost,127.0.0.1,::1']))
    return env


def execute(args, key, command):
    print('+ ' + shlex.join(command['argv']), flush=True)
    log = args.output_dir / 'logs' / f'{key}-{command["stage"]}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    env = child_env(args, command['stage'], key)
    secrets = [v for k, v in env.items() if k in ('ACCURACY_API_KEY', 'ACCURACY_JUDGE_API_KEY', 'OPENAI_API_KEY') and v != 'EMPTY' and v]
    def redact(line):
        for secret in secrets:
            line = line.replace(secret, '[REDACTED]')
        return line
    with log.open('a', encoding='utf-8') as output:
        process = subprocess.Popen(command['argv'], cwd=ROOT, env=env, stdout=subprocess.PIPE, start_new_session=True,
                                   stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
        try:
            for line in process.stdout:
                output.write(redact(line))
                output.flush()
                print(redact(line), end='', flush=True)
            code = process.wait()
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
        finally:
            process.stdout.close()
    # The pinned native GPT4V adapter logs its key at INFO. Redact its own logs,
    # too; keys are never put into config files or command-line arguments here.
    for logfile in (args.output_dir / 'vlmeval' / key / 'logs').glob('*.log'):
        logfile.write_text(redact(logfile.read_text(encoding='utf-8')), encoding='utf-8')
    if code:
        raise RuntimeError(f'Native command exited {code}; see {log}')


def check_predictions(predictions):
    for text in predictions:
        if not isinstance(text, str) or not text.strip() or 'Failed to obtain answer via API' in text:
            raise ValueError('Native predictions contain an empty answer or an API failure marker')


def audit_oc(args, key):
    run = args.output_dir / 'opencompass' / key / 'native'
    alias = ALIASES[key]
    predictions = json.loads((run / 'predictions/q38' / f'{alias}.json').read_text())
    n = shared.SPECS[key]['count']
    if set(predictions) != {str(i) for i in range(n)}:
        raise ValueError(f'{key}: expected all {n} native prediction indices')
    check_predictions([predictions[str(i)].get('prediction') for i in range(n)])
    result_file = run / 'results/q38' / f'{alias}.json'
    result = json.loads(result_file.read_text())
    if result.get('error'):
        raise ValueError(result['error'])
    metric = 'accuracy' if key == 'gsm8k' else 'pass@1'
    score = float(result[metric])
    return dict(status='scored', score=score, evaluated=n, source=str(result_file),
                protocol=f'OpenCompass native CLI; {alias}; {metric}; n=1')


def latest_vlm_run(args, key):
    root = args.output_dir / 'vlmeval' / key / 'q38'
    runs = sorted(p for p in root.glob('T*') if p.is_dir() and (p / 'status.json').exists())
    if not runs:
        raise FileNotFoundError(f'No native status.json under {root}')
    return runs[-1]


def audit_vlm(args, key):
    run = latest_vlm_run(args, key)
    status = json.loads((run / 'status.json').read_text())['datasets'][ALIASES[key]]
    if status.get('status') != 'done' or status.get('error_message') or status.get('skip_reason'):
        raise ValueError(f'Native evaluation not complete: {status.get("status")}; {status.get("error_message") or status.get("skip_reason")}')
    pred = Path(status['prediction_file'])
    pred = pred if pred.is_absolute() else run / pred
    if pred.suffix != '.tsv':
        raise ValueError('Expected native TSV predictions (PRED_FORMAT=tsv)')
    rows = shared.table_rows(pred)
    data = shared.table_rows(args.data_dir / shared.SOURCES[key][0][0])
    ids = [r['index'] for r in rows]
    if len(ids) != len(data) or set(ids) != {r['index'] for r in data}:
        raise ValueError(f'{key}: incomplete/duplicate native prediction IDs')
    check_predictions([r.get('prediction') for r in rows])
    metrics = status['metrics']
    if key == 'realworldqa':
        score = float(metrics['split=none|Overall']) * 100.0
    else:
        score = float(metrics['Subject=Overall|acc'])
        judge_file = pred.with_name(f'{pred.stem}_{args.judge_model}.tsv')
        # Upstream uses the configured judge name for intermediate file names.
        graded = shared.table_rows(judge_file)
        if len(graded) != len(data):
            raise ValueError('MathVision judge output is incomplete')
        for row in graded:
            if 'All 5 retries failed' in row.get('log', '') or not row.get('res', '').strip():
                raise ValueError('MathVision native judge failed to extract one or more answers')
    return dict(status='scored', score=score, evaluated=len(rows), source=str(run / 'status.json'),
                protocol=f'VLMEvalKit native CLI; {ALIASES[key]}',
                math_judge='api' if key == 'mathvision' else None)


def audit(args, key):
    result = audit_oc(args, key) if key in ('gsm8k', 'livecodebench') else audit_vlm(args, key)
    if not math.isfinite(result['score']) or not 0 <= result['score'] <= 100:
        raise ValueError('Invalid native score scale/value')
    return result


def report(output_dir, results):
    rows = []
    for key in shared.SPECS:
        row = shared.admission(key, results.get(key, {}))
        row['source'] = results.get(key, {}).get('source', '')
        rows.append(row)
    overall = ('PASS' if all(r['status'] == 'PASS' for r in rows) else
               'FAIL' if all(r['status'] in ('PASS', 'FAIL') for r in rows) else 'INCOMPLETE')
    shared.write_json(output_dir / 'summary.json', dict(created_at=shared.utcnow(), engine='native-cli', overall=overall, results=rows))
    with (output_dir / 'summary.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['| Benchmark | 基线 % | 阈值 % | 实测 % | 降幅 pp | 完成题数 | 判定 |',
             '|---|---:|---:|---:|---:|---:|---|']
    for row in rows:
        score = '—' if row['score'] is None else f'{row["score"]:.4f}'
        drop = '—' if row['drop_pp'] is None else f'{row["drop_pp"]:.4f}'
        lines.append(f'| {row["benchmark"]} | {row["baseline"]:.1f} | {row["threshold"]:.1f} | {score} | {drop} | {row["evaluated"]}/{row["expected"]} | {row["status"]} |')
    lines += ['', f'整体判定：**{overall}**', '',
              '分数由 OpenCompass / VLMEvalKit 原生命令生成；本报告仅校验完整性、转换单位和判断阈值。',
              '准入使用未舍入的分数。MathVision 使用原生 LLM Judge，无 Code Interpreter。',
              '官方基线的完整参数尚未提供；原生命令不等同于已经复现官方基线。', '']
    lines.extend(f'- {r["benchmark"]}: {r["error"]}' for r in rows if r['error'])
    content = '\n'.join(lines) + '\n'
    (output_dir / 'summary.md').write_text(content, encoding='utf-8')
    print('\n' + content, flush=True)
    return 0 if overall == 'PASS' else 1 if overall == 'FAIL' else 2


def check_suite(previous, current):
    if previous == current:
        return
    # Allow inference without a judge followed by evaluation with a judge.
    # Changing an already chosen judge requires a new suite directory.
    judge_keys = ('judge_model', 'judge_base_url')
    if (not any(previous.get(k) for k in judge_keys)
            and all(current.get(k) for k in judge_keys)
            and {k: v for k, v in previous.items() if k not in judge_keys}
            == {k: v for k, v in current.items() if k not in judge_keys}):
        return
    raise ValueError('Native suite configuration changed; use a different --output-dir')


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('command', choices=['prepare', 'config', 'run', 'infer', 'eval', 'report'])
    p.add_argument('--benchmarks', nargs='+', choices=list(shared.SPECS), default=list(shared.SPECS))
    p.add_argument('--data-dir', type=Path, default=ROOT / 'data/accuracy')
    p.add_argument('--output-dir', type=Path, default=ROOT / 'result/accuracy-native/default')
    p.add_argument('--offline', action='store_true')
    p.add_argument('--insecure-dataset-download', action='store_true')
    p.add_argument('--vlm-data-base-url', default='https://opencompass.openxlab.space/utils/VLMEval')
    p.add_argument('--base-url', default='http://127.0.0.1:18580/v1')
    p.add_argument('--model', default='Qwen3.8-27B-Channel-INT8-w8a8')
    p.add_argument('--model-revision', default='')
    p.add_argument('--max-tokens', type=int, default=32768)
    p.add_argument('--gsm8k-max-tokens', type=int, default=512, help='Preserve upstream GSM8K default unless explicitly overridden')
    p.add_argument('--concurrency', type=int, default=4)
    p.add_argument('--eval-workers', type=int, default=4)
    p.add_argument('--code-timeout', type=int, default=6)
    p.add_argument('--timeout', type=int, default=3600)
    p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--seed', type=int, default=20261012)
    p.add_argument('--tokenizer-path', default='gpt-4', help='Native API adapter token-counting tokenizer; may be a local tokenizer directory')
    p.add_argument('--extra-body', default='{}')
    p.add_argument('--judge-base-url', default=os.environ.get('ACCURACY_JUDGE_BASE_URL', ''))
    p.add_argument('--judge-model', default=os.environ.get('ACCURACY_JUDGE_MODEL', ''))
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    args.data_dir, args.output_dir = args.data_dir.resolve(), args.output_dir.resolve()
    for name in ('base_url', 'judge_base_url'):
        url = getattr(args, name).rstrip('/')
        if url and not url.endswith('/v1'):
            url += '/v1'
        setattr(args, name, url)
    if min(args.max_tokens, args.gsm8k_max_tokens, args.concurrency, args.eval_workers, args.code_timeout, args.timeout) <= 0:
        p.error('Counts and timeouts must be positive')
    if not math.isfinite(args.temperature) or not 0 <= args.temperature <= 2:
        p.error('temperature must be finite and between 0 and 2')
    args.extra_body = json.loads(args.extra_body)
    if not isinstance(args.extra_body, dict) or set(args.extra_body) - {'chat_template_kwargs', 'top_k', 'min_p', 'repetition_penalty'}:
        p.error('Only chat_template_kwargs, top_k, min_p, repetition_penalty are supported; no tools')
    if args.judge_model and any(c in args.judge_model for c in '/\\'):
        p.error('Use a slash-free served judge model alias (VLMEvalKit embeds it in filenames)')
    if args.command == 'prepare':
        shared.write_json(args.data_dir / 'native_manifest.json', prepare(args))
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / '.native.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.command == 'report':
            results = {}
            suite_file = args.output_dir / 'suite_config.json'
            if suite_file.exists():
                for name, value in json.loads(suite_file.read_text()).items():
                    if hasattr(args, name):
                        setattr(args, name, value)
                args.data_dir = Path(args.data_dir)
            for key in shared.SPECS:
                file = args.output_dir / key / 'result.json'
                if file.exists():
                    results[key] = json.loads(file.read_text())
                root = args.output_dir / ('opencompass' if key in ('gsm8k', 'livecodebench') else 'vlmeval') / key
                if root.exists() and results.get(key, {}).get('status') not in ('error', 'running'):
                    try:
                        results[key] = audit(args, key)
                    except Exception as exc:
                        results[key] = dict(status='error', error=str(exc))
            return report(args.output_dir, results)
        config = settings(args)
        suite = args.output_dir / 'suite_config.json'
        if suite.exists():
            check_suite(json.loads(suite.read_text()), config)
        shared.write_json(suite, config)
        configs = generate_configs(args)
        mode = {'run': 'all', 'config': 'all'}.get(args.command, args.command)
        plans = {key: commands(args, key, configs[key], mode) for key in args.benchmarks}
        save_commands(args, plans)
        if args.command == 'config':
            for key, plan in plans.items():
                for command in plan:
                    print(f'[{key}] ' + shlex.join(command['argv']))
            return 0
        results = {}
        shared.write_json(args.output_dir / 'run.json', dict(started_at=shared.utcnow(), command=argv or sys.argv[1:]))
        environment = sorted(f'{d.metadata["Name"]}=={d.version}' for d in importlib.metadata.distributions())
        (args.output_dir / 'environment.txt').write_text('\n'.join(environment) + '\n', encoding='utf-8')
        for key in shared.SPECS:
            file = args.output_dir / key / 'result.json'
            if key not in args.benchmarks and file.exists():
                results[key] = json.loads(file.read_text())
        for key in args.benchmarks:
            shared.write_json(args.output_dir / key / 'result.json', dict(status='running'))
        report(args.output_dir, results)
        for key in args.benchmarks:
            try:
                if key == 'mathvision' and mode != 'infer' and not (args.judge_model and args.judge_base_url):
                    raise ValueError('MathVision native scoring needs --judge-model and --judge-base-url; infer mode is available without a judge')
                versions = shared.toolkit_versions([key])
                shared.write_json(args.output_dir / key / 'versions.json', versions)
                selected = argparse.Namespace(**{**vars(args), 'benchmarks': [key]})
                manifest = prepare(selected)
                data_file = args.output_dir / key / 'data.json'
                if data_file.exists() and json.loads(data_file.read_text()) != manifest:
                    raise ValueError('Cached dataset content changed; choose a new output directory')
                shared.write_json(data_file, manifest)
                for command in plans[key]:
                    execute(args, key, command)
                results[key] = dict(status='inferred', evaluated=0) if mode == 'infer' else audit(args, key)
            except Exception as exc:
                results[key] = dict(status='error', error=f'{type(exc).__name__}: {exc}')
                print(f'[{key}] ERROR: {exc}', file=sys.stderr, flush=True)
            shared.write_json(args.output_dir / key / 'result.json', results[key])
            report(args.output_dir, results)
        code = report(args.output_dir, results)
        if mode == 'infer':
            return 0 if all(results[k]['status'] == 'inferred' for k in args.benchmarks) else 2
        return code


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('Interrupted; native work directories are retained for --reuse.', file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}', file=sys.stderr)
        sys.exit(2)
