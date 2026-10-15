"""Compare the frozen two-case quick workload; never declare final Goal success."""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path

CASES = ((4096, 64), (65536, 1))
MATCH_FIELDS = ('kind', 'model', 'warmup_rounds', 'measure_rounds',
                'requests_per_round', 'output_len', 'seed', 'cases', 'script_sha256')


def read_run(root):
    root = Path(root)
    meta = json.loads((root / 'metadata.json').read_text())
    summary = json.loads((root / 'summary.json').read_text())
    if (meta['kind'] != 'optimization-quick' or meta['warmup_rounds'] != 1
            or meta['measure_rounds'] != 1 or meta['output_len'] != 1024
            or meta['requests_per_round'] != 'concurrency'
            or meta['cases'] != [dict(input_len=n, concurrency=c) for n, c in CASES]):
        raise ValueError(f'{root}: workload differs from the frozen quick protocol')
    if (root / 'exit_code').read_text().strip() != '0' or summary['status'] != 'PASS':
        raise ValueError(f'{root}: incomplete or failed run')
    started = datetime.fromisoformat((root / 'started_at').read_text().strip().replace('Z', '+00:00'))
    finished = datetime.fromisoformat((root / 'finished_at').read_text().strip().replace('Z', '+00:00'))
    elapsed = (finished - started).total_seconds()
    if not 0 < elapsed <= 480:
        raise ValueError(f'{root}: quick suite duration {elapsed}s is outside (0,480]')
    rows = summary['cases']
    if len(rows) != 2 or {(r['input_len'], r['concurrency']) for r in rows} != set(CASES):
        raise ValueError(f'{root}: missing, duplicate or unexpected case')
    by_case = {(r['input_len'], r['concurrency']): r for r in rows}
    for n, c in CASES:
        row = by_case[n, c]
        detail = json.loads((root / f'{n}in_1024out_c{c}' / 'summary.json').read_text())
        rounds = detail['rounds']
        if (row['status'] != 'PASS' or row['output_len'] != 1024
                or detail['status'] != 'PASS' or detail['successful_rounds'] != 1
                or len(rounds) != 1 or rounds[0]['phase'] != 'measure'):
            raise ValueError(f'{root}: invalid measured round {n}/{c}')
        wave = rounds[0]
        if (wave['status'] != 'PASS' or wave['requested'] != c
                or wave['successful'] != c or wave['exact_length'] != c
                or wave['input_len'] != n or wave['output_len'] != 1024
                or wave['concurrency'] != c):
            raise ValueError(f'{root}: request or token integrity failure {n}/{c}')
        for field, detail_field in [('ttft_p50_s', 'ttft_p50_s'),
                                    ('tpot_p50_ms', 'tpot_p50_ms'),
                                    ('output_tps', 'output_throughput_tps')]:
            value = row[field]
            if not math.isfinite(value) or value <= 0 or value != detail[detail_field]['mean']:
                raise ValueError(f'{root}: invalid or inconsistent metric {field}')
    deployment = json.loads((root / 'deployment.json').read_text())
    if not deployment.get('image_id', '').startswith('sha256:'):
        raise ValueError(f'{root}: missing immutable image ID')
    return meta, by_case, elapsed, deployment


def compare(baseline, candidate):
    bm, br, bt, bd = read_run(baseline)
    cm, cr, ct, cd = read_run(candidate)
    for field in MATCH_FIELDS:
        if bm[field] != cm[field]:
            raise ValueError(f'Unmatched protocol field: {field}')
    rows = []
    for key in CASES:
        b, c = br[key], cr[key]
        improvement = (1 - c['ttft_p50_s'] / b['ttft_p50_s']) * 100
        rows.append(dict(input_len=key[0], output_len=1024, concurrency=key[1],
                         baseline=b, candidate=c, ttft_improvement_pct=improvement,
                         tps_improvement_pct=(c['output_tps']/b['output_tps']-1)*100,
                         tpot_improvement_pct=(1-c['tpot_p50_ms']/b['tpot_p50_ms'])*100,
                         ttft_20pct_met=c['ttft_p50_s'] <= b['ttft_p50_s'] * 0.8))
    return dict(scope='one matched quick pair; repeated A/B and model accuracy are separate gates',
                baseline=str(baseline), candidate=str(candidate),
                baseline_image_id=bd['image_id'], candidate_image_id=cd['image_id'],
                baseline_elapsed_s=bt, candidate_elapsed_s=ct,
                all_cases_ttft_20pct_met=all(r['ttft_20pct_met'] for r in rows),
                final_goal_verified=False, cases=rows)


def markdown(report):
    lines = ['| 场景 | 并发 | 指标 | 安擎基线 | 优化结果 | 提升 |',
             '| --- | ---: | --- | ---: | ---: | ---: |']
    for row in report['cases']:
        for field, label, gain in [('output_tps', 'TPS(token/s)', 'tps_improvement_pct'),
                                   ('ttft_p50_s', 'TTFT P50(s)', 'ttft_improvement_pct'),
                                   ('tpot_p50_ms', 'TPOT P50(ms/token)', 'tpot_improvement_pct')]:
            lines.append(f"| {row['input_len']//1024}K输入/1K输出 | {row['concurrency']} | {label} | "
                         f"{row['baseline'][field]:.3f} | {row['candidate'][field]:.3f} | {row[gain]:.2f}% |")
    met = '满足' if report['all_cases_ttft_20pct_met'] else '未满足'
    lines += ['', f'本轮两档逐项 TTFT 提升≥20%：{met}。单对快测不代表最终 Goal 验收；重复 A/B 和模型精度另验。',
              f"套件耗时：安擎基线 {report['baseline_elapsed_s']:.0f}s，优化结果 {report['candidate_elapsed_s']:.0f}s。"]
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--output-prefix', type=Path)
    args = parser.parse_args()
    report = compare(args.baseline, args.candidate)
    rendered = markdown(report)
    if args.output_prefix:
        args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
        args.output_prefix.with_suffix('.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        args.output_prefix.with_suffix('.md').write_text(rendered)
    print(rendered, end='')


if __name__ == '__main__':
    main()
