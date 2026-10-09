#!/usr/bin/env bash
# ==============================================================================
# 脚本名称: run_bench_sharegpt.sh
# 脚本功能: 自动化执行 ShareGPT 真实对话数据集基准测试
# 测试规范:
#   - 并发档位: C1, C4, C16, C32, C64
#   - 每档执行 3 轮正式测试，每轮 1000 个请求，2×并发进行预热
# ==============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULT_BASE_DIR="${PROJECT_DIR}/result/sharegpt"

BASE_URL=${BASE_URL:-"http://127.0.0.1:18580"}
MODEL_PATH=${MODEL_PATH:-"/model"}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"Qwen3.8-27B-Channel-INT8-w8a8"}
DATASET_PATH=${DATASET_PATH:-"/dataset/ShareGPT_V3_unfiltered_cleaned_split.json"}

if [ ! -f "${DATASET_PATH}" ]; then
  echo "错误: 未找到 ShareGPT 数据集文件: ${DATASET_PATH}"
  echo "请检查挂载或设置环境变量 DATASET_PATH"
  exit 1
fi

mkdir -p "${RESULT_BASE_DIR}"

CONCURRENCIES=(1 4 16 32 64)
NUM_PROMPTS=1000
ROUNDS=3

echo "================================================================"
echo "开始 ShareGPT 真实数据集基准测试"
echo "服务地址: ${BASE_URL}"
echo "模型名称: ${SERVED_MODEL_NAME}"
echo "数据集路径: ${DATASET_PATH}"
echo "并发档位: ${CONCURRENCIES[*]}"
echo "每档轮次: ${ROUNDS} 轮 (每轮 ${NUM_PROMPTS} 请求)"
echo "结果目录: ${RESULT_BASE_DIR}"
echo "================================================================"

for c in "${CONCURRENCIES[@]}"; do
  c_dir="${RESULT_BASE_DIR}/sharegpt-c${c}"
  mkdir -p "${c_dir}"
  warmup_reqs=$((2 * c))

  echo "----------------------------------------------------------------"
  echo ">>> 开始执行并发档位: C${c} (预热请求数: ${warmup_reqs})"
  echo "----------------------------------------------------------------"

  for r in $(seq 1 "${ROUNDS}"); do
    round_str=$(printf "%02d" "${r}")
    output_file="${c_dir}/round-${round_str}.jsonl"

    echo "[C${c}] 开始第 ${r}/${ROUNDS} 轮测试 -> ${output_file}"

    python3 -u -m sglang.bench_serving \
      --backend sglang-oai \
      --base-url "${BASE_URL}" \
      --dataset-name sharegpt \
      --dataset-path "${DATASET_PATH}" \
      --model "${MODEL_PATH}" \
      --served-model-name "${SERVED_MODEL_NAME}" \
      --sharegpt-context-len 131072 \
      --num-prompts "${NUM_PROMPTS}" \
      --max-concurrency "${c}" \
      --request-rate inf \
      --warmup-requests "${warmup_reqs}" \
      --seed 20260912 \
      --disable-tqdm \
      --output-details \
      --output-file "${output_file}"

    echo "[C${c}] 第 ${r}/${ROUNDS} 轮完成"
    sleep 5
  done
done

echo "================================================================"
echo "ShareGPT 评测全部完成！测试结果保存在: ${RESULT_BASE_DIR}"
echo "================================================================"
