#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/../.venv/bin/python}"
exec "${SCRIPT_DIR}/../../scripts/run_pipeline.sh" llava15_earthvqa 1 "$@"
