#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU="$1"
PYTHON="$2"
JOB="$3"
OFFLINE="${4:-0}"

OUT="$ROOT/analysis/telemetry_50/$JOB"
INJECTION="$OUT/injection.jsonl"
LABELS="$OUT/labels.jsonl"
LOG="$OUT/job.log"
mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1

if [[ "$OFFLINE" == "1" ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$ROOT/src"
export TOKENIZERS_PARALLELISM=false

cd "$ROOT"
echo "[$(date -Is)] start job=$JOB gpu=$GPU telemetry_max_steps=50"

"$PYTHON" -m detect_sdc.cli run \
  --job "$JOB" \
  --stage inject \
  --device cuda:0 \
  --telemetry-max-steps 50 \
  --injection-output "$INJECTION"

"$PYTHON" -m detect_sdc.cli label \
  --job "$JOB" \
  --input "$INJECTION" \
  --output "$LABELS" \
  --device cuda:0 \
  --batch-size 64 \
  --overwrite

echo "[$(date -Is)] completed job=$JOB"
