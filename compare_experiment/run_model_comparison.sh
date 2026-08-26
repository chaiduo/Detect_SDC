#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 {qwen25_vl|internvl3|llava15} PHYSICAL_GPU" >&2
    exit 2
fi

MODEL_KEY="$1"
PHYSICAL_GPU="$2"
ROOT="/data01/cd_workspace/Detect_SDC"
LOG_ROOT="${ROOT}/logs/comparison_20260825"

case "${MODEL_KEY}" in
    qwen25_vl)
        PYTHON="${ROOT}/Qwen2.5-VL-7B/.venv/bin/python"
        JOBS=(
            qwen25_vl_earthvqa
            qwen25_vl_lingoqa
            qwen25_vl_vqav2
        )
        ;;
    internvl3)
        PYTHON="${ROOT}/InternVL3-8B/.venv/bin/python"
        JOBS=(
            internvl3_earthvqa
            internvl3_lingoqa
            internvl3_vqav2
        )
        ;;
    llava15)
        PYTHON="${ROOT}/llava-v1.5-7B/.venv/bin/python"
        JOBS=(
            llava15_earthvqa
            llava15_lingoqa
            llava15_vqav2
        )
        ;;
    *)
        echo "unknown model key: ${MODEL_KEY}" >&2
        exit 2
        ;;
esac

mkdir -p "${LOG_ROOT}"
LOG="${LOG_ROOT}/${MODEL_KEY}.log"
exec > >(tee -a "${LOG}") 2>&1

cd "${ROOT}"
export PYTHONPATH="src:."
export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export TOKENIZERS_PARALLELISM="false"
export PYTHONUNBUFFERED="1"
export HF_HUB_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"

echo "[comparison] model=${MODEL_KEY} physical_gpu=${PHYSICAL_GPU}"
echo "[comparison] started=$(date --iso-8601=seconds)"

for JOB in "${JOBS[@]}"; do
    RESULT_ROOT="${ROOT}/compare_experiment/results/${JOB}"
    mkdir -p "${RESULT_ROOT}"
    echo "[comparison] job=${JOB} stage=profile"
    if [[ ! -f "${RESULT_ROOT}/profiles.json" ]]; then
        "${PYTHON}" -m compare_experiment.profile_baselines \
            --job "${JOB}" \
            --device cuda:0
    else
        echo "[comparison] skip existing ${RESULT_ROOT}/profiles.json"
    fi

    echo "[comparison] job=${JOB} stage=manifest"
    if [[ ! -f "${RESULT_ROOT}/replay_manifest.jsonl" ]]; then
        "${PYTHON}" -m compare_experiment.build_replay_manifest \
            --job "${JOB}"
    else
        echo "[comparison] skip existing replay manifest"
    fi

    echo "[comparison] job=${JOB} stage=replay"
    "${PYTHON}" -m compare_experiment.replay_detection \
        --job "${JOB}" \
        --device cuda:0

    echo "[comparison] job=${JOB} stage=evaluate"
    "${PYTHON}" -m compare_experiment.evaluate_results \
        --job "${JOB}"
done

echo "[comparison] completed=$(date --iso-8601=seconds)"
