# Detection Method Comparison

This directory evaluates three detectors on the same fault executions:

- Ranger-style activation range checking;
- Dr.DNA-style activation distribution checking;
- SIEVE nominal inter-layer discrepancy detection.

Ranger-style and Dr.DNA-style are mechanism-matched PyTorch/VLM adaptations,
not reproductions of the original systems.

## Protocol

- Use the frozen dataset-level Fit/Calibration/Final-test manifests.
- Build Ranger and Dr.DNA profiles from clean Fit samples only.
- Inject every fault once and retain every run.
- Attach Ranger and Dr.DNA scores during the canonical fault inference.
- Train SIEVE on Fit features.
- Calibrate every method independently by maximizing Significant-SDC F1.
- Report Full and Finite metrics on untouched Final test groups; Finite excludes
  only rows whose 72 SIEVE features are all NaN.
- Bootstrap confidence intervals by `semantic_group_id`.

## Execution

Run one physical GPU per model family:

```bash
./compare_experiment/run_model_comparison.sh qwen25_vl 2
./compare_experiment/run_model_comparison.sh internvl3 3
./compare_experiment/run_model_comparison.sh llava15 4
```

The launcher executes the complete v2 pipeline in dependency order. The key
comparison-specific stages are:

```bash
PYTHONPATH=src:. python -m compare_experiment.profile_baselines \
  --job qwen25_vl_lingoqa --device cuda:0

PYTHONPATH=src:. python -m compare_experiment.collect_detection_data \
  --job qwen25_vl_lingoqa --device cuda:0

PYTHONPATH=src:. python -m compare_experiment.evaluate_results \
  --job qwen25_vl_lingoqa
```

`collect_detection_data` writes the canonical injection JSONL. It runs each
fault once while SIEVE telemetry, Ranger scores, and Dr.DNA scores observe the
same execution. There is no replay manifest or answer-mismatch filtering.

After all nine jobs:

```bash
PYTHONPATH=src:. python -m compare_experiment.summarize_results
```

Canonical outputs are stored below
`compare_experiment/results_v2/<job>/`; model artifacts and labeled records are
stored below `artifacts/iclr_v2/<job>/`.
