#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 ]]; then
    echo "Usage: $0 JOB DEFAULT_CUDA_VISIBLE_DEVICES [detect-sdc run options]" >&2
    exit 2
fi

JOB="$1"
DEFAULT_CUDA_VISIBLE_DEVICES="$2"
shift 2

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPOSITORY_ROOT}/../.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python}"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${DEFAULT_CUDA_VISIBLE_DEVICES}}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONPATH="${REPOSITORY_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

exec "${PYTHON_BIN}" -m detect_sdc.cli run \
    --job "${JOB}" \
    --stage profile \
    --stage collect_mapping \
    --stage train_mapping \
    --stage inject \
    --repository-root "${REPOSITORY_ROOT}" \
    --device "${DEVICE:-cuda:0}" \
    "$@"
