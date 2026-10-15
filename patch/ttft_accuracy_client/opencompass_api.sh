#!/usr/bin/env bash
# Native OpenCompass entrypoint with scoped optional-backend isolation.
set -euo pipefail
q38_launcher_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${q38_launcher_dir}/client_env.sh"
q38_project_dir=$(cd -- "${q38_launcher_dir}/../.." && pwd)
q38_accuracy_venv=${ACCURACY_VENV:-"${q38_project_dir}/.venv-accuracy"}
[[ -x "${q38_accuracy_venv}/bin/opencompass" ]] || { printf 'Missing native OpenCompass: %s\n' "${q38_accuracy_venv}/bin/opencompass" >&2; exit 1; }
exec taskset -c "${ACCURACY_CPU_AFFINITY:-0-3}" nice -n 19 "${q38_accuracy_venv}/bin/opencompass" "$@"
