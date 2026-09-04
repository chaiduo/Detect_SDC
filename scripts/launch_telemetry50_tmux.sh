#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="${1:-sieve_telemetry50}"
JOB_SCRIPT="$ROOT/scripts/run_telemetry50_job.sh"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi

tmux new-session -d -s "$SESSION" -n qwen_gpu0
tmux send-keys -t "$SESSION:qwen_gpu0" \
  "bash '$JOB_SCRIPT' 0 '$ROOT/Qwen2.5-VL-7B/.venv/bin/python' qwen25_vl_earthvqa 0 && bash '$JOB_SCRIPT' 0 '$ROOT/Qwen2.5-VL-7B/.venv/bin/python' qwen25_vl_lingoqa 0" C-m

declare -a JOBS=(
  "qwen_gpu1|1|$ROOT/Qwen2.5-VL-7B/.venv/bin/python|qwen25_vl_vqav2|0"
  "llava_gpu2|2|$ROOT/llava-v1.5-7B/.venv/bin/python|llava15_earthvqa|1"
  "llava_gpu3|3|$ROOT/llava-v1.5-7B/.venv/bin/python|llava15_lingoqa|1"
  "llava_gpu4|4|$ROOT/llava-v1.5-7B/.venv/bin/python|llava15_vqav2|1"
  "internvl_gpu5|5|$ROOT/InternVL3-8B/.venv/bin/python|internvl3_earthvqa|0"
  "internvl_gpu6|6|$ROOT/InternVL3-8B/.venv/bin/python|internvl3_lingoqa|0"
  "internvl_gpu7|7|$ROOT/InternVL3-8B/.venv/bin/python|internvl3_vqav2|0"
)

for item in "${JOBS[@]}"; do
  IFS='|' read -r window gpu python job offline <<<"$item"
  tmux new-window -t "$SESSION" -n "$window"
  tmux send-keys -t "$SESSION:$window" \
    "bash '$JOB_SCRIPT' '$gpu' '$python' '$job' '$offline'" C-m
done

tmux select-window -t "$SESSION:qwen_gpu0"
echo "Started tmux session: $SESSION"
echo "Attach with: tmux attach -t $SESSION"
echo "Windows: $(tmux list-windows -t "$SESSION" -F '#{window_name}' | paste -sd ',')"
