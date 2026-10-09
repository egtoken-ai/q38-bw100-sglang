#!/usr/bin/env bash
# ==============================================================================
# 脚本名称: run_bench_fixed.sh
# 脚本功能: 自动化执行固定长度 (4K/1K 与 64K/1K) 基准性能压测
# 测试规范:
#   - 4K/1K 档位: C1, C8, C16, C32, C64, C128
#   - 64K/1K 档位: C1, C8, C16, C32
#   - 预热 2 轮，正式测试 3 轮，每轮请求数 max(8, 2×并发)
# ==============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_PY="${PROJECT_DIR}/scripts/benchmark.py"
RESULT_BASE_DIR="${PROJECT_DIR}/result/fixed"

BASE_URL=${BASE_URL:-"http://127.0.0.1:18580"}
MODEL_NAME=${MODEL_NAME:-"Qwen3.8-27B-Channel-INT8-w8a8"}
SCENARIO=${1:-"all"} # 可选: all, 4k, 64k

mkdir -p "${RESULT_BASE_DIR}"

run_single_fixed() {
  local input_len=$1
  local output_len=$2
  local concurrency=$3
  local scene_tag="${input_len}in_${output_len}out_c${concurrency}"
  local output_dir="${RESULT_BASE_DIR}/${scene_tag}"
  local reqs_per_round=$((concurrency * 2))
  if [ "${reqs_per_round}" -lt 8 ]; then
    reqs_per_round=8
  fi

  echo "----------------------------------------------------------------"
  echo ">>> 开始执行测试: 输入=${input_len}, 输出=${output_len}, 并发=${concurrency}"
  echo "    输出目录: ${output_dir}"
  echo "    轮次配置: 预热 2 轮, 测量 3 轮, 每轮 ${reqs_per_round} 请求"
  echo "----------------------------------------------------------------"

  python3 -u "${BENCH_PY}" \
    --base-url "${BASE_URL}" \
    --model "${MODEL_NAME}" \
    --output-dir "${output_dir}" \
    --input-len "${input_len}" \
    --output-len "${output_len}" \
    --concurrency "${concurrency}" \
    --warmup-rounds 2 \
    --measure-rounds 3 \
    --requests-per-round "${reqs_per_round}" \
    --seed 20260912 \
    --timeout 3600
}

# 1. 4K 输入 / 1K 输出测试集
if [ "${SCENARIO}" = "all" ] || [ "${SCENARIO}" = "4k" ]; then
  echo "=== [阶段 1] 启动 4K 输入 / 1K 输出 固定长度压测 ==="
  CONCS_4K=(1 8 16 32 64 128)
  for c in "${CONCS_4K[@]}"; do
    run_single_fixed 4096 1024 "${c}"
  done
fi

# 2. 64K 输入 / 1K 输出测试集
if [ "${SCENARIO}" = "all" ] || [ "${SCENARIO}" = "64k" ]; then
  echo "=== [阶段 2] 启动 64K 输入 / 1K 输出 固定长度压测 ==="
  CONCS_64K=(1 8 16 32)
  for c in "${CONCS_64K[@]}"; do
    run_single_fixed 65536 1024 "${c}"
  done
fi

echo "================================================================"
echo "固定长度压测全部完成！测试结果保存在: ${RESULT_BASE_DIR}"
echo "================================================================"
