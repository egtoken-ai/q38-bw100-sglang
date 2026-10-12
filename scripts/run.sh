#!/usr/bin/env bash
# ==============================================================================
# 脚本名称: run.sh
# 脚本功能: 竞赛指定的一键式评测入口脚本
# 用法:
#   bash scripts/run.sh start-server   # 启动推理服务 (后台运行并等待就绪)
#   bash scripts/run.sh bench-fixed    # 执行固定长度 (4K / 64K) 测试
#   bash scripts/run.sh bench-sharegpt # 执行 ShareGPT 真实数据集测试
#   bash scripts/run.sh all            # 完整流水线 (启动服务 -> 固定长度测试 -> ShareGPT 测试)
#   bash scripts/run.sh stop-server    # 停止后台推理服务
# ==============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS_DIR="${PROJECT_DIR}/scripts"

COMMAND=${1:-"all"}

PORT=${PORT:-18580}
BASE_URL="http://127.0.0.1:${PORT}"

check_server_ready() {
  curl -s "${BASE_URL}/health" > /dev/null 2>&1
}

case "${COMMAND}" in
  start-server)
    echo "=== 启动 SGLang 推理服务 ==="
    bash "${SCRIPTS_DIR}/start_server.sh" --daemon
    ;;

  bench-fixed)
    if ! check_server_ready; then
      echo "提示: 推理服务尚未运行，正在自动启动服务..."
      bash "${SCRIPTS_DIR}/start_server.sh" --daemon
    fi
    echo "=== 开始执行固定长度测试 ==="
    bash "${SCRIPTS_DIR}/run_bench_fixed.sh" "${2:-all}"
    ;;

  bench-sharegpt)
    if ! check_server_ready; then
      echo "提示: 推理服务尚未运行，正在自动启动服务..."
      bash "${SCRIPTS_DIR}/start_server.sh" --daemon
    fi
    echo "=== 开始执行 ShareGPT 测试 ==="
    bash "${SCRIPTS_DIR}/run_bench_sharegpt.sh"
    ;;

  accuracy)
    shift
    bash "${SCRIPTS_DIR}/run_accuracy.sh" "$@"
    ;;

  accuracy-native)
    shift
    bash "${SCRIPTS_DIR}/run_accuracy_native.sh" "$@"
    ;;

  all)
    echo "=== 启动全量评测任务 ==="
    if ! check_server_ready; then
      echo "[1/3] 启动 SGLang 推理服务..."
      bash "${SCRIPTS_DIR}/start_server.sh" --daemon
    else
      echo "[1/3] 检测到 SGLang 服务已经在运行，直接进入评测..."
    fi

    echo "[2/3] 执行固定长度测试 (4K 与 64K)..."
    bash "${SCRIPTS_DIR}/run_bench_fixed.sh" all

    echo "[3/3] 执行 ShareGPT 测试..."
    bash "${SCRIPTS_DIR}/run_bench_sharegpt.sh"

    echo "=== 全部评测流水线已成功执行完毕！==="
    ;;

  stop-server)
    echo "=== 正在停止 SGLang 服务 ==="
    pkill -f "sglang.launch_server" || echo "未找到运行中的 SGLang 进程"
    ;;

  *)
    echo "未知命令: ${COMMAND}"
    echo "可用命令: start-server | bench-fixed | bench-sharegpt | accuracy {prepare|run|report} | accuracy-native {prepare|config|run|infer|eval|report} | all | stop-server"
    exit 1
    ;;
esac
