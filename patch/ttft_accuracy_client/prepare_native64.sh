#!/usr/bin/env bash
# Generate only native configs; no model requests or dataset preparation.
set -euo pipefail
q38_config_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${q38_config_dir}/client_env.sh"
q38_project_dir=$(cd -- "${q38_config_dir}/../.." && pwd)
q38_accuracy_venv=${ACCURACY_VENV:-"${q38_project_dir}/.venv-accuracy"}
export ACCURACY_PYTHON="${q38_accuracy_venv}/bin/python"
q38_output_root=${ACCURACY_OUTPUT_ROOT:-"${q38_project_dir}/result/accuracy-native"}
for q38_label in native optimized; do
  q38_output="${q38_output_root}/gsm8k64-${q38_label}"
  if [[ -d "${q38_output}/opencompass" ]]; then
    printf 'Existing run directory; choose a fresh ACCURACY_OUTPUT_ROOT: %s\n' "$q38_output" >&2
    exit 1
  fi
  bash "${q38_project_dir}/scripts/run_accuracy_native.sh" config \
    --benchmarks gsm8k --offline \
    --data-dir "${q38_project_dir}/data/accuracy" \
    --output-dir "$q38_output" \
    --base-url "${ACCURACY_BASE_URL:-http://127.0.0.1:18580/v1}" \
    --model Qwen3.8-27B-Channel-INT8-w8a8 \
    --tokenizer-path "${ACCURACY_TOKENIZER_PATH:-/model}" \
    --gsm8k-max-tokens 512 --temperature 0 --seed 20261012 --concurrency 32 \
    --extra-body '{"chat_template_kwargs":{"enable_thinking":false}}'
done
"${ACCURACY_PYTHON}" - "$q38_output_root" <<'PY'
from pathlib import Path
import sys
root=Path(sys.argv[1])
for label in ('native','optimized'):
    p=root/f'gsm8k64-{label}/configs/gsm8k.py'
    text=p.read_text()
    marker="datasets[0]['reader_cfg']['test_range'] = '[:64]'"
    if marker not in text:
        p.write_text(text+'\n# Explicit identical subset; not full GSM8K acceptance.\n'+marker+'\n')
    compile(p.read_text(),str(p),'exec')
    print('Prepared native config:',p)
PY
