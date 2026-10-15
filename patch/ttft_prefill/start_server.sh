#!/usr/bin/env bash
# Use the target-vendor FA implementation verified by bench_attention.py.
set -euo pipefail
export ATTENTION_BACKEND=${ATTENTION_BACKEND:-fa3}
export SGLANG_KV_LAYOUT_HCU_FA=${SGLANG_KV_LAYOUT_HCU_FA:-true}
exec bash /bench/scripts/start_server.sh "$@"
