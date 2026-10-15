#!/usr/bin/env bash
# 优化迭代专用：2 个 case，每档 1 轮 warmup + 1 轮正式测试，各发一波请求。
# 用法：bash scripts/run_bench_quick.sh [base|优化版本标签] [--dry-run]
# 原生/优化服务使用同一套 quick 配置比较；完整验收仍使用 run_bench_fixed.sh。
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_URL=${BASE_URL:-"http://127.0.0.1:${PORT:-18580}"}
MODEL_NAME=${MODEL_NAME:-"Qwen3.8-27B-Channel-INT8-w8a8"}
RESULT_ROOT=${RESULT_ROOT:-"${PROJECT_DIR}/result/quick"}
RUN_LABEL=opt
DRY_RUN=0
LABEL_SET=0

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help)
      echo "用法: bash scripts/run_bench_quick.sh [标签] [--dry-run]"
      echo "4K/1K: C64；64K/1K: C1。每档预热 1 轮、正式 1 轮，请求数=并发数。"
      echo "环境变量: BASE_URL、MODEL_NAME、RESULT_ROOT；结果写入独立的标签/时间戳目录。"
      exit 0 ;;
    *)
      if [[ "$arg" == -* || "$LABEL_SET" == 1 || ! "$arg" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
        echo "无效参数或标签: $arg" >&2
        exit 2
      fi
      RUN_LABEL=$arg
      LABEL_SET=1
      ;;
  esac
done

if [[ "$DRY_RUN" == 1 ]]; then
  RUN_DIR="${RESULT_ROOT}/${RUN_LABEL}/<timestamp>"
else
  # 仅连接用户已启动的服务，避免自动启动时改变待测镜像/GPU/服务参数。
  python3 - "$BASE_URL" "$MODEL_NAME" <<'PY'
import sys
import requests

base, model = sys.argv[1:]
s = requests.Session()
s.trust_env = False
s.get(base.rstrip('/') + '/health', timeout=10).raise_for_status()
r = s.get(base.rstrip('/') + '/v1/models', timeout=10)
r.raise_for_status()
if model not in {item['id'] for item in r.json()['data']}:
    raise SystemExit('模型名称与运行中的服务不匹配: ' + model)
PY
  mkdir -p "${RESULT_ROOT}/${RUN_LABEL}"
  RUN_DIR=$(mktemp -d "${RESULT_ROOT}/${RUN_LABEL}/$(date +%Y%m%d-%H%M%S)-XXXXXX")
  trap 'rc=$?; printf "%s\n" "$rc" > "${RUN_DIR}/exit_code"; date -u +%FT%TZ > "${RUN_DIR}/finished_at"' EXIT
  date -u +%FT%TZ > "${RUN_DIR}/started_at"
  echo "$$" > "${RUN_DIR}/runner.pid"
  python3 - "$RUN_DIR" "$RUN_LABEL" "$BASE_URL" "$MODEL_NAME" "$PROJECT_DIR" <<'PY'
import hashlib
import json
import pathlib
import sys

out, label, base, model, project = sys.argv[1:]
scripts = pathlib.Path(project) / 'scripts'
metadata = {
    'kind': 'optimization-quick', 'label': label, 'base_url': base, 'model': model,
    'warmup_rounds': 1, 'measure_rounds': 1, 'requests_per_round': 'concurrency',
    'output_len': 1024, 'seed': 20260912,
    'cases': [{'input_len': n, 'concurrency': c}
              for n, c in [(4096, 64), (65536, 1)]],
    'script_sha256': {name: hashlib.sha256((scripts / name).read_bytes()).hexdigest()
                      for name in ['benchmark.py', 'run_bench_quick.sh']},
    'scope': 'Iteration screening; compare against the same quick workload, not full benchmark acceptance.',
}
pathlib.Path(out, 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
PY
fi

echo "快速优化测试: ${RUN_LABEL}；结果目录: ${RUN_DIR}"
for spec in '4096 64' '65536 1'; do
  read -r input_len concurrency <<< "$spec"
  scene="${input_len}in_1024out_c${concurrency}"
  cmd=(python3 -u "${PROJECT_DIR}/scripts/benchmark.py"
    --base-url "$BASE_URL" --model "$MODEL_NAME"
    --output-dir "${RUN_DIR}/${scene}"
    --input-len "$input_len" --output-len 1024 --concurrency "$concurrency"
    --warmup-rounds 1 --measure-rounds 1 --requests-per-round "$concurrency"
    --seed 20260912 --timeout 3600)
  echo "=== ${scene}：预热 1 轮 + 正式 1 轮，每轮 ${concurrency} 个请求 ==="
  if [[ "$DRY_RUN" == 1 ]]; then
    printf '%q ' "${cmd[@]}"
    printf '\n'
  else
    "${cmd[@]}" 2>&1 | tee "${RUN_DIR}/${scene}.log"
  fi
done

if [[ "$DRY_RUN" == 1 ]]; then
  exit 0
fi

python3 - "$RUN_DIR" <<'PY'
import csv
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
rows = []
for n, c in [(4096, 64), (65536, 1)]:
    d = json.loads((root / f'{n}in_1024out_c{c}' / 'summary.json').read_text())
    if d['status'] != 'PASS' or d['successful_rounds'] != 1:
        raise SystemExit(f'档位未通过: {n} / {c}')
    rows.append({'input_len': n, 'output_len': 1024, 'concurrency': c,
                 'status': d['status'], 'output_tps': d['output_throughput_tps']['mean'],
                 'ttft_p50_s': d['ttft_p50_s']['mean'],
                 'tpot_p50_ms': d['tpot_p50_ms']['mean']})
(root / 'summary.json').write_text(json.dumps({'status': 'PASS', 'cases': rows}, indent=2) + '\n')
with (root / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print('\n场景\t并发\tTPS(token/s)\tTTFT P50(s)\tTPOT P50(ms/token)')
for r in rows:
    print(f"{r['input_len']//1024}K/1K\t{r['concurrency']}\t{r['output_tps']:.3f}\t"
          f"{r['ttft_p50_s']:.3f}\t{r['tpot_p50_ms']:.3f}")
PY
echo "快速测试完成: ${RUN_DIR}/summary.csv"
