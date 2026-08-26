# Detection Method Comparison

This directory implements detection-only comparisons for:

- Ranger-style numerical range checking
- Dr.DNA-style activation distribution checking
- SIEVE semantic discrepancy detection

The implementations are mechanism-matched PyTorch/VLM adaptations. They are
not claimed to reproduce the original Ranger or Dr.DNA systems.

## Protocol

- Monitor the eight decoder layers used by SIEVE.
- Observe the first two decoding steps.
- Replay the same saved activation faults once for all methods.
- Calibrate method thresholds on Non-SDC records at a 1% FPR budget.
- Report SDC recall, Significant-SDC recall, Non-SDC FPR, precision, and F1.
- Report both the full cohort and the finite-only cohort.

Configuration is stored in
`configs/detection_comparison.yaml`.

## Per-job execution

Run modules from the repository root with the virtual environment associated
with the selected model:

```bash
export PYTHONPATH=src:.

python -m compare_experiment.profile_baselines \
  --job qwen25_vl_lingoqa \
  --device cuda:0

python -m compare_experiment.build_replay_manifest \
  --job qwen25_vl_lingoqa

python -m compare_experiment.replay_detection \
  --job qwen25_vl_lingoqa \
  --device cuda:0

python -m compare_experiment.evaluate_results \
  --job qwen25_vl_lingoqa
```

`replay_detection` appends one completed record at a time and resumes by
`sample_uid` after interruption. Use `--overwrite` only when intentionally
restarting a job.

After all nine jobs complete:

```bash
python -m compare_experiment.summarize_results
```

Artifacts are written below `compare_experiment/results/<job>/`.
