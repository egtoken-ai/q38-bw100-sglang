#!/usr/bin/env bash
# ==============================================================================
# 脚本名称: start_server.sh
# 脚本功能: 启动 SGLang 推理服务 (适用于海光 BW100 双卡平台 + INT8-W8A8 模型)
# 默认端口: 18580 (NCCL 端口: 19580)
# ==============================================================================

set -euo pipefail

# 基础目录
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# 环境变量设置 (海光 BW100 官方推荐环境参数)
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-"0,1"}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export HSA_FORCE_FINE_GRAIN_PCIE=1
export SGLANG_ROCM_USE_AITER_MOE=false
export SGLANG_KV_LAYOUT_HCU_FA=${SGLANG_KV_LAYOUT_HCU_FA:-false}

# 服务配置参数
MODEL_PATH=${MODEL_PATH:-"/model"}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"Qwen3.8-27B-Channel-INT8-w8a8"}
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-18580}
NCCL_PORT=${NCCL_PORT:-19580}
TP_SIZE=${TP_SIZE:-2}
DTYPE=${DTYPE:-"bfloat16"}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-"bfloat16"}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-131072}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.85}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-128}
CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-8192}
MAX_PREFILL_TOKENS=${MAX_PREFILL_TOKENS:-16384}
PAGE_SIZE=${PAGE_SIZE:-64}
CUDA_GRAPH_MAX_BS=${CUDA_GRAPH_MAX_BS:-128}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-triton}

LOG_FILE="${LOG_DIR}/server.log"

echo "================================================================"
echo "准备启动 SGLang 推理服务"
echo "硬件设备: HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES} (海光 BW100)"
echo "模型路径: ${MODEL_PATH}"
echo "模型名称: ${SERVED_MODEL_NAME}"
echo "服务地址: http://${HOST}:${PORT}"
echo "并行规模: TP=${TP_SIZE}"
echo "上下文长: ${CONTEXT_LENGTH}"
echo "日志输出: ${LOG_FILE}"
echo "================================================================"

# 检查模型路径是否存在
if [ ! -d "${MODEL_PATH}" ] && [ ! -f "${MODEL_PATH}/config.json" ]; then
  echo "警告: 模型路径 ${MODEL_PATH} 不存在或未挂载，请确保挂载正确！"
fi

RUN_IN_BACKGROUND=false
if [ "${1:-}" = "--daemon" ] || [ "${1:-}" = "-d" ]; then
  RUN_IN_BACKGROUND=true
fi

LAUNCH_CMD=(
  python3 -u -m sglang.launch_server
  --model-path "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host "${HOST}"
  --port "${PORT}"
  --nccl-port "${NCCL_PORT}"
  --tp-size "${TP_SIZE}"
  --dtype "${DTYPE}"
  --kv-cache-dtype "${KV_CACHE_DTYPE}"
  --context-length "${CONTEXT_LENGTH}"
  --mem-fraction-static "${MEM_FRACTION_STATIC}"
  --max-running-requests "${MAX_RUNNING_REQUESTS}"
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}"
  --max-prefill-tokens "${MAX_PREFILL_TOKENS}"
  --disable-radix-cache
  --attention-backend "${ATTENTION_BACKEND}"
  --page-size "${PAGE_SIZE}"
  --mamba-scheduler-strategy no_buffer
  --cuda-graph-max-bs "${CUDA_GRAPH_MAX_BS}"
  --disable-custom-all-reduce
  --enable-metrics
)

if [ -n "${PACK_PAGED_KV_TO_VARLEN:-}" ]; then
  LAUNCH_CMD+=(--pack-paged-kv-to-varlen "${PACK_PAGED_KV_TO_VARLEN}")
fi

# Explicit local NUMA mapping for TP_SIZE=2
if [ "$TP_SIZE" = "2" ]; then
  case "${HIP_VISIBLE_DEVICES// /}" in
    "2,3")
      LAUNCH_CMD+=(--numa-node 0 0)
      ;;
    "4,5")
      LAUNCH_CMD+=(--numa-node 4 4)
      ;;
    "6,7")
      LAUNCH_CMD+=(--numa-node 4 4)
      ;;
    "0,1")
      LAUNCH_CMD+=(--numa-node 0 0)
      ;;
    *)
      echo "提示: 未预设的 HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}，跳过显式 --numa-node 参数"
      ;;
  esac
fi

if [ "${RUN_IN_BACKGROUND}" = true ]; then
  echo "正在后台启动服务，日志记录至: ${LOG_FILE}"
  nohup "${LAUNCH_CMD[@]}" > "${LOG_FILE}" 2>&1 &
  SERVER_PID=$!
  echo "服务 PID: ${SERVER_PID}"

  echo "等待服务就绪 (轮询 http://127.0.0.1:${PORT}/health)..."
  MAX_RETRIES=120
  COUNT=0
  until curl -fsS --max-time 5 "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1 || [ ${COUNT} -eq ${MAX_RETRIES} ]; do
    sleep 3
    COUNT=$((COUNT + 1))
    echo -n "."
  done
  echo ""

  if curl -fsS --max-time 5 "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
    echo "SGLang 推理服务已就绪！"
  else
    echo "警告: 等待超时，服务可能仍在初始化或发生错误，请查看日志: ${LOG_FILE}"
  fi
else
  # 前台执行
  exec "${LAUNCH_CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
fi
