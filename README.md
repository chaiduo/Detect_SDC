# SIEVE: Significant Silent Data Corruption Detection

This repository contains the code and experiment protocol for **SIEVE**, a
semantic-aware detector for significant silent data corruptions (SDCs) in
multimodal language models.

SIEVE monitors discrepancies between adjacent transformer layers, predicts the
fault-free representation with a lightweight mapping model, and uses an
XGBoost detector to identify corruptions that are likely to cause a significant
semantic failure. The repository is organized for reproducible evaluation
across three vision-language models and three multimodal benchmarks.

> The public repository contains source code, configuration, tests, analysis
> reports, and compact figure data. Model checkpoints, datasets, and large
> JSONL artifacts are intentionally kept outside Git.

## Highlights

- Significant-SDC detection rather than generic output-change detection.
- Group-aware `Fit` / `Calibration` / `Final Test` isolation.
- Prefill activation fault injection with configurable bit policies.
- Layer-aware residual mapping model for fault-free feature prediction.
- Compact 48D monitoring candidate selected using validation-only evidence.
- Complete 9-task evaluation matrix:
  Qwen2.5-VL, LLaVA-1.5, and InternVL3 on EarthVQA, LingoQA, and VQAv2.
- Online monitoring and cross-domain detector-transfer experiments.

## Repository Status

The repository contains two related experiment tracks:

| Track | Representation | Purpose |
| --- | --- | --- |
| v2 reference track | 6 layer pairs, `K=2`, 72D | Reproducible baseline artifact and historical comparison |
| compact track | 4 layer pairs, `K=2`, 48D | Validation-selected compact detector candidate |
| telemetry extension | Prefix telemetry up to `K=50` | Offline step-window ablations for `K=2,4,8,16` |

The compact layer pairs are:

```text
(6, 7), (22, 23), (25, 26), (26, 27)
```

The raw telemetry extension is written to `analysis/telemetry_50/` and is
separate from the reference artifacts under `artifacts/iclr_v2/`. Large local
outputs are not versioned.

## Method

The canonical experiment evaluates:

- **Models:** Qwen2.5-VL-7B, LLaVA-1.5-7B, and InternVL3-8B.
- **Datasets:** EarthVQA, LingoQA, and VQAv2.
- **Fault:** one Prefill activation double-bit flip on an eligible linear
  operation.
- **Fault policy:** `random` for the main campaign.
- **Runs:** one clean execution and ten fault runs per input.
- **Monitoring:** projected inter-layer discrepancies over decoding steps.
- **Features:** cosine similarity, mean difference, standard deviation
  difference, and L2 distance, aggregated with mean/max/min.
- **Target:** `significant_sdc_target`, assigned by the Prometheus judge when a
  corruption produces a severity-2 semantic failure.

The evaluation protocol is strictly separated:

```text
Fit            train Mapping and Detector
Calibration    freeze the operating threshold
Final Test     report metrics only
```

Splits are made by `semantic_group_id`, so related frames or questions cannot
cross the outer partition boundary:

- EarthVQA: image group
- LingoQA: question group
- VQAv2: image group

## Reported Results

The validation-selected 48D analysis reports the following nine-task
Final-Test macro averages:

| Cohort | Precision | Recall | F1 | FPR |
| --- | ---: | ---: | ---: | ---: |
| Full | 93.74% | 91.23% | 92.03% | 0.210% |
| Finite | 88.90% | 83.45% | 84.75% | 0.210% |

The 48D representation removes one third of the 72D features while retaining
comparable Fit-holdout performance in a 20-seed paired validation experiment.
The detailed reports are available in:

```text
analysis/layer_pair_48d_stability/
analysis/layer_pair_subset_search/
analysis/detector_transfer_48d/
```

The LLaVA prefix-step study is available in:

```text
analysis/step_ablation_telemetry50/
```

It compares `K=2,4,8,16` and larger reference windows using the same
telemetry campaign.

## Installation

The pipeline requires Python 3.10 or newer. Install the package and the
CPU-side training dependencies with:

```bash
python -m pip install -e '.[train,dev]'
```

Model-specific environments are documented by the frozen dependency files:

```text
reproducibility/environments/qwen25_vl.freeze.txt
reproducibility/environments/internvl3.freeze.txt
reproducibility/environments/llava15.freeze.txt
```

The reference setup uses separate environments because the three model
adapters require different Transformers and model-runtime versions.

## Data and Model Preparation

The repository does not redistribute model checkpoints, datasets, or judge
weights. Set local paths in:

```text
configs/models/qwen25_vl.yaml
configs/models/llava15.yaml
configs/models/internvl3.yaml
configs/datasets/earthvqa.yaml
configs/datasets/lingoqa.yaml
configs/datasets/vqav2.yaml
```

The expected external inputs are:

```text
Qwen2.5-VL-7B-Instruct
InternVL3-8B
llava-v1.5-7b
LLaVA source tree
Prometheus judge checkpoint
EarthVQA, LingoQA, and VQAv2 data
```

Do not commit these inputs or generated JSONL artifacts.

## Quickstart

Validate the experiment configuration without loading model weights:

```bash
PYTHONPATH=src python -m detect_sdc.cli config validate \
  configs/experiments/current.yaml
```

Create or validate the shared semantic-group manifests:

```bash
PYTHONPATH=src python -m detect_sdc.cli split --dataset earthvqa
PYTHONPATH=src python -m detect_sdc.cli split --dataset lingoqa
PYTHONPATH=src python -m detect_sdc.cli split --dataset vqav2
```

Run a dry-run for one configured job:

```bash
PYTHONPATH=src python -m detect_sdc.cli run \
  --job qwen25_vl_earthvqa \
  --stage inject \
  --dry-run
```

## Full Pipeline

For one model/dataset job, the stages are executed in this order:

```text
profile
collect_mapping
train_mapping
inject
label
featurize
train_detector
report
```

Example:

```bash
export PYTHONPATH=src:.
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

PYTHON=Qwen2.5-VL-7B/.venv/bin/python
JOB=qwen25_vl_earthvqa

for STAGE in profile collect_mapping train_mapping inject label \
    featurize train_detector report; do
  "$PYTHON" -m detect_sdc.cli run \
    --job "$JOB" \
    --stage "$STAGE" \
    --device cuda:0
done
```

Before a long GPU run, use `--dry-run` and verify the configured output paths.
Injection outputs are written atomically and can be resumed from a completed
fault-run boundary.

## Extended Telemetry and Step Ablation

To retain a longer prefix window for offline step studies, use:

```bash
bash scripts/run_telemetry50_job.sh \
  0 \
  Qwen2.5-VL-7B/.venv/bin/python \
  qwen25_vl_earthvqa
```

The script writes to:

```text
analysis/telemetry_50/<job>/
```

`--telemetry-max-steps 50` changes the retained telemetry prefix. It does not
force the model to generate 50 tokens; generation remains controlled by
`max_new_tokens` in the model configuration.

After a complete telemetry campaign, the 48D step runner evaluates:

```bash
PYTHONPATH=src Qwen2.5-VL-7B/.venv/bin/python \
  scripts/run_48d_step_ablation.py \
  --input-root analysis/telemetry_50 \
  --output-dir analysis/step_ablation_telemetry50 \
  --overwrite
```

## Baseline Comparison

`compare_experiment/` evaluates Ranger-style, Dr.DNA-style, and SIEVE signals
on the same fault executions. This avoids comparing methods that observed
different injected samples.

```bash
./compare_experiment/run_model_comparison.sh qwen25_vl 2
./compare_experiment/run_model_comparison.sh internvl3 3
./compare_experiment/run_model_comparison.sh llava15 4
```

## Reproducibility

The detailed protocol and environment assumptions are documented in:

- [`docs/reproducibility.md`](docs/reproducibility.md)
- [`docs/sieve_iclr_revision_plan.md`](docs/sieve_iclr_revision_plan.md)
- [`docs/xgboost_current_methods_and_results.md`](docs/xgboost_current_methods_and_results.md)
- [`reproducibility/reference_sha256.txt`](reproducibility/reference_sha256.txt)

The most important invariants are:

- zero overlap between outer Fit, Calibration, and Final-Test groups;
- Mapping and Detector training use Fit data only;
- Calibration labels are not used to train the Detector;
- Final Test is never used for feature or threshold selection;
- all clean and fault runs are retained;
- generated outputs use stable sample UIDs and atomic writes.

## Testing

Run the unit-test suite with:

```bash
python -m pytest
```

Or use the standard library test runner:

```bash
PYTHONPATH=src python -m unittest discover tests
```

## Citation

The repository name is now `SIEVE_ICLR-27`. Citation metadata will be added
when the associated paper record is public.

## License

No open-source license has been declared for this repository yet. Please
contact the authors before redistributing the code or generated artifacts.
