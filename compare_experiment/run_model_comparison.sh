#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "usage: $0 {qwen25_vl|internvl3|llava15} PHYSICAL_GPU [JOB]" >&2
    exit 2
fi

MODEL_KEY="$1"
PHYSICAL_GPU="$2"
REQUESTED_JOB="${3:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_ROOT="${ROOT}/logs/comparison_iclr_v2"

case "${MODEL_KEY}" in
    qwen25_vl)
        PYTHON="${ROOT}/Qwen2.5-VL-7B/.venv/bin/python"
        JOBS=(qwen25_vl_earthvqa qwen25_vl_lingoqa qwen25_vl_vqav2)
        ;;
    internvl3)
        PYTHON="${ROOT}/InternVL3-8B/.venv/bin/python"
        JOBS=(internvl3_earthvqa internvl3_lingoqa internvl3_vqav2)
        ;;
    llava15)
        PYTHON="${ROOT}/llava-v1.5-7B/.venv/bin/python"
        JOBS=(llava15_earthvqa llava15_lingoqa llava15_vqav2)
        ;;
    *)
        echo "unknown model key: ${MODEL_KEY}" >&2
        exit 2
        ;;
esac

if [[ -n "${REQUESTED_JOB}" ]]; then
    JOB_FOUND=false
    for JOB in "${JOBS[@]}"; do
        if [[ "${JOB}" == "${REQUESTED_JOB}" ]]; then
            JOB_FOUND=true
            break
        fi
    done
    if [[ "${JOB_FOUND}" != true ]]; then
        echo "job ${REQUESTED_JOB} does not belong to model ${MODEL_KEY}" >&2
        exit 2
    fi
    JOBS=("${REQUESTED_JOB}")
fi

mkdir -p "${LOG_ROOT}"
LOG_KEY="${REQUESTED_JOB:-${MODEL_KEY}}"
LOG="${LOG_ROOT}/${LOG_KEY}.log"
exec > >(tee -a "${LOG}") 2>&1

cd "${ROOT}"
export PYTHONPATH="src:."
export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export TOKENIZERS_PARALLELISM="false"
export PYTHONUNBUFFERED="1"
export HF_HUB_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"

echo "[comparison] model=${MODEL_KEY} physical_gpu=${PHYSICAL_GPU}"
echo "[comparison] started=$(date --iso-8601=seconds)"

for JOB in "${JOBS[@]}"; do
    ARTIFACT_ROOT="${ROOT}/artifacts/iclr_v2/${JOB}"
    RESULT_ROOT="${ROOT}/compare_experiment/results_v2/${JOB}"
    mkdir -p "${RESULT_ROOT}"

    if [[ ! -f "${ARTIFACT_ROOT}/json/profile.json" ]]; then
        "${PYTHON}" -m detect_sdc.cli run --job "${JOB}" \
            --stage profile --device cuda:0
    fi
    if [[ ! -f "${ARTIFACT_ROOT}/json/mapping.jsonl" ]]; then
        "${PYTHON}" -m detect_sdc.cli run --job "${JOB}" \
            --stage collect_mapping --device cuda:0
    fi
    if [[ ! -f "${ARTIFACT_ROOT}/model/mapping_model.pt" ]]; then
        "${PYTHON}" -m detect_sdc.cli run --job "${JOB}" \
            --stage train_mapping --device cuda:0
    fi
    if [[ ! -f "${RESULT_ROOT}/profiles.json" ]]; then
        "${PYTHON}" -m compare_experiment.profile_baselines \
            --job "${JOB}" --device cuda:0
    fi
    INJECTION_PATH="${ARTIFACT_ROOT}/json/injection.jsonl"
    if [[ ! -f "${INJECTION_PATH}" ]]; then
        "${PYTHON}" -m compare_experiment.collect_detection_data \
            --job "${JOB}" --device cuda:0
    elif ! "${PYTHON}" -c '
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    record = next(json.loads(line) for line in stream if line.strip())
required = {"ranger_score", "drdna_score", "has_non_finite"}
raise SystemExit(0 if required.issubset(record) else 1)
' "${INJECTION_PATH}"; then
        echo "[comparison] existing injection lacks unified scores: ${INJECTION_PATH}" >&2
        echo "[comparison] archive/remove it, then rerun this launcher" >&2
        exit 1
    fi
    if [[ ! -f "${ARTIFACT_ROOT}/json/labels.jsonl" ]]; then
        "${PYTHON}" -m detect_sdc.cli run --job "${JOB}" \
            --stage label --device cuda:0
    fi
    if [[ ! -f "${ARTIFACT_ROOT}/train_data/test.csv" ]]; then
        "${PYTHON}" -m detect_sdc.cli run --job "${JOB}" --stage featurize
    fi
    if [[ ! -f "${ARTIFACT_ROOT}/output/metrics_summary.json" ]]; then
        "${PYTHON}" -m detect_sdc.cli run --job "${JOB}" \
            --stage train_detector
    fi
    "${PYTHON}" -m compare_experiment.evaluate_results --job "${JOB}"
done

echo "[comparison] completed=$(date --iso-8601=seconds)"
