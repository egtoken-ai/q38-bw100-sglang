#!/usr/bin/env bash
# Repeated matched A/B for the two fixed TTFT cases (4K/1K C64, 64K/1K C1).
# Runs on the BW100 host from the synced project directory. Every arm gets a
# fresh service start; arm order alternates per round to cancel drift.
#
#   bash scripts/run_ttft_ab_repeat.sh --rounds 3 --tag <tag> [--allow-busy-gpus]
#
# Arms (fixed):
#   native    container qwen38_bw100_sglang      (native image, scripts/start_server.sh)
#   candidate container ${CANDIDATE_CONTAINER}   (/opt/ttft-numa/start_server.sh)
# Results: result/ab/<tag>/round-<i>-<arm>.path holds each quick-suite directory.
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/data/tools/q38-bw100-sglang}
NATIVE_CONTAINER=${NATIVE_CONTAINER:-qwen38_bw100_sglang}
CANDIDATE_CONTAINER=${CANDIDATE_CONTAINER:-qwen38_bw100_ttft_rccl8}
ROUNDS=3
TAG=""
ALLOW_BUSY=false
ARM_TIMEOUT_S=${ARM_TIMEOUT_S:-1500}

while [ $# -gt 0 ]; do
  case "$1" in
    --rounds) ROUNDS=$2; shift 2 ;;
    --tag) TAG=$2; shift 2 ;;
    --allow-busy-gpus) ALLOW_BUSY=true; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[ -n "$TAG" ] || { echo "--tag is required" >&2; exit 2; }

OUT_DIR="${PROJECT_DIR}/result/ab/${TAG}"
mkdir -p "${OUT_DIR}"
cd "${PROJECT_DIR}"

# HCU busy check: prints VRAM% of the given HCU indices.
vram_of() {
  bash -lc "hy-smi" | awk -v want="$1" '$1==want {print $6}'
}

check_gpus() {
  local stage=$1 busy=""
  for g in 4 5 6 7; do
    local v; v=$(vram_of "$g")
    echo "${stage} HCU${g} VRAM=${v}" >> "${OUT_DIR}/gpu-state.log"
    if [ "$v" != "0%" ]; then busy="${busy} ${g}"; fi
  done
  if [ -n "$busy" ] && [ "$ALLOW_BUSY" != true ]; then
    echo "HCU busy before ${stage}:${busy}; refuse matched A/B (use --allow-busy-gpus for screening)" >&2
    exit 3
  fi
}

run_arm() {
  local round=$1 arm=$2 container start label exit_file
  label="ab-${TAG}-r${round}-${arm}"
  exit_file="result/ab/${TAG}/round-${round}-${arm}.exit_code"
  rm -f "${exit_file}"
  check_gpus "round${round}-${arm}-start"
  if [ "$arm" = native ]; then
    container=$NATIVE_CONTAINER
    start=(docker exec -d -w /bench -e HIP_VISIBLE_DEVICES=4,5 -e MODEL_PATH=/model
      -e ATTENTION_BACKEND=triton -e SGLANG_KV_LAYOUT_HCU_FA=false
      -e CHUNKED_PREFILL_SIZE=8192 -e MAX_PREFILL_TOKENS=16384 "$container"
      bash -lc "bash scripts/start_server.sh --daemon > logs/startup-${label}.log 2>&1 && bash scripts/run_bench_quick.sh ${label} > logs/quick-${label}.log 2>&1; echo \$? > ${exit_file}")
  else
    container=$CANDIDATE_CONTAINER
    start=(docker exec -d -w /bench -e HIP_VISIBLE_DEVICES=4,5 -e MODEL_PATH=/model
      -e CHUNKED_PREFILL_SIZE=65536 -e MAX_PREFILL_TOKENS=65536 -e MEM_FRACTION_STATIC=0.80
      -e SGLANG_QWEN27B_MLP_PIPELINE_PARTS=8 -e TORCH_NCCL_BLOCKING_WAIT=0 "$container"
      bash -lc "python3 /opt/ttft-numa/check_startup.py --reference /bench/scripts/start_server.sh && bash /opt/ttft-numa/start_server.sh --daemon > logs/startup-${label}.log 2>&1 && bash scripts/run_bench_quick.sh ${label} > logs/quick-${label}.log 2>&1; echo \$? > ${exit_file}")
  fi
  docker start "$container" > /dev/null
  docker inspect "$container" --format '{{.Image}} {{.Config.Image}}' > "${OUT_DIR}/round-${round}-${arm}.image"
  "${start[@]}"
  local waited=0
  until [ -f "${exit_file}" ]; do
    sleep 15; waited=$((waited + 15))
    if [ "$waited" -ge "$ARM_TIMEOUT_S" ]; then
      echo "timeout round${round} ${arm}" >&2
      docker stop --timeout 20 "$container" > /dev/null || true
      exit 4
    fi
  done
  grep -o '结果目录: [^ ]*' "logs/quick-${label}.log" | head -1 | sed 's#结果目录: /bench/##' \
    > "${OUT_DIR}/round-${round}-${arm}.path"
  docker stop --timeout 20 "$container" > /dev/null
  sleep 10
  local rc; rc=$(cat "${exit_file}")
  echo "round${round} ${arm} rc=${rc} $(cat "${OUT_DIR}/round-${round}-${arm}.path")" | tee -a "${OUT_DIR}/progress.log"
  [ "$rc" = 0 ] || exit 5
}

date -Is > "${OUT_DIR}/started_at"
for ((i = 1; i <= ROUNDS; i++)); do
  if [ $((i % 2)) -eq 1 ]; then order=(native candidate); else order=(candidate native); fi
  for arm in "${order[@]}"; do run_arm "$i" "$arm"; done
done
date -Is > "${OUT_DIR}/finished_at"
echo "A/B finished: ${OUT_DIR}"
