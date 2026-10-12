#!/usr/bin/env bash
# Native OpenCompass / VLMEvalKit CLI orchestrator; no custom inference engine.
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ACCURACY_PYTHON:-${PROJECT_DIR}/.venv-accuracy/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=python3
fi
exec "${PYTHON}" -u "${PROJECT_DIR}/scripts/accuracy_native.py" "$@"
