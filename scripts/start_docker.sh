#!/usr/bin/env bash
# ==============================================================================
# 脚本名称: start_docker.sh
# 脚本功能: 启动用于海光 BW100 的 SGLang 官方推荐 Docker 容器
# 镜像标签: harbor.sourcefind.cn:5443/dcu/admin/base/sglang:0.5.12-ubuntu22.04-dtk2604-py3.10
# ==============================================================================

set -euo pipefail

# 容器与环境默认变量
CONTAINER_NAME=${CONTAINER_NAME:-"qwen38_bw100_sglang"}
IMAGE_NAME=${IMAGE_NAME:-"harbor.sourcefind.cn:5443/dcu/admin/base/sglang:0.5.12-ubuntu22.04-dtk2604-py3.10"}
MODEL_PATH=${MODEL_PATH:-"/workspace/work/Qwen3.8-27B-Channel-INT8-w8a8"}
DATASET_PATH=${DATASET_PATH:-"/workspace/dataset"}
BENCH_DIR=${BENCH_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
SHM_SIZE=${SHM_SIZE:-"128g"}

echo "=== [1/2] 准备启动 DCU 容器 ==="
echo "容器名称: ${CONTAINER_NAME}"
echo "镜像名称: ${IMAGE_NAME}"
echo "模型挂载: ${MODEL_PATH} -> /model:ro"
echo "项目挂载: ${BENCH_DIR} -> /bench"
if [ -d "${DATASET_PATH}" ]; then
  echo "数据挂载: ${DATASET_PATH} -> /dataset:ro"
fi

# 检查是否已存在同名容器
if docker ps -a --format '{{.Names}}' | grep -Eq "^${CONTAINER_NAME}\$"; then
  echo "警告: 容器 ${CONTAINER_NAME} 已存在，正在停止并移除旧容器..."
  docker rm -f "${CONTAINER_NAME}" || true
fi

# 组装挂载命令
MOUNT_ARGS=(
  -v /opt/hyhal:/opt/hyhal:ro
  -v "${MODEL_PATH}:/model:ro"
  -v "${BENCH_DIR}:/bench"
)

if [ -d "${DATASET_PATH}" ]; then
  MOUNT_ARGS+=(-v "${DATASET_PATH}:/dataset:ro")
fi

# 启动容器
echo "=== [2/2] 执行 docker run ==="
docker run -d \
  --name "${CONTAINER_NAME}" \
  --network host \
  --ipc host \
  --shm-size "${SHM_SIZE}" \
  --privileged \
  --device=/dev/kfd \
  --device=/dev/dri \
  --device=/dev/mkfd \
  --group-add video \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -u root \
  "${MOUNT_ARGS[@]}" \
  --entrypoint /bin/bash \
  "${IMAGE_NAME}" \
  -lc 'sleep infinity'

echo "容器 ${CONTAINER_NAME} 启动成功！"
echo "进入容器命令: docker exec -it ${CONTAINER_NAME} bash"
