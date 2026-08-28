# Detect_SDC Reproducibility Guide

This document is the execution contract for reproducing the Detect_SDC
experiments. Result interpretation remains in
[`experiments_overview.md`](experiments_overview.md). Run all commands from the
repository root:

```bash
cd /data01/cd_workspace/Detect_SDC
```

## 1. Reproduction levels

There are three supported reproduction levels:

| Level | Starting artifacts | Reproduces | Main requirement |
|---|---|---|---|
| Full | Models and raw datasets | All eight pipeline stages | GPU time, judge model, about 70 GB of canonical JSONL output |
| Detector | `labels.jsonl` and Mapping checkpoints | Features, detectors, and ablations | Large JSONL archive |
| Analysis | Tracked CSV/JSON summaries | Aggregate tables and figures with tracked source data | Git checkout only |

The Git repository intentionally excludes model weights, raw datasets, Mapping
checkpoints, Detector `.ubj` files, and multi-GB JSONL files. A full or
Detector-level reproduction therefore requires the external artifact archive
described in Section 5.

The experiment code and tracked results were produced from:

```text
Git commit: 10f3d6074f10b860bd0068a24ed84d3769f39c75
Commit date: 2026-08-26
Commit subject: feat: add online detection experiments
```

Documentation added after that commit does not change the experiment code or
reported metrics.

## 2. Hardware and system software

The reference machine used:

| Component | Reference value |
|---|---|
| GPU | 8 x NVIDIA H20 |
| Memory per GPU | 97,871 MiB |
| NVIDIA driver | 580.173.02 |
| CUDA toolkit (`nvcc`) | 12.8, build 12.8.93 |
| OS kernel | Linux 5.15.152.bsk.10-amd64 |
| Architecture | x86_64 |

One model job occupies one physical GPU. In commands below,
`CUDA_VISIBLE_DEVICES=2` exposes physical GPU 2 as logical `cuda:0`; substitute
another free physical GPU when necessary. Do not colocate timing experiments
with other GPU workloads.

The three model families require separate Python environments:

| Model | Python | PyTorch | PyTorch CUDA | Transformers | XGBoost |
|---|---|---|---|---|---|
| Qwen2.5-VL-7B | 3.12.13 | 2.11.0+cu128 | 12.8 | 5.13.1 | 3.2.0 |
| InternVL3-8B | 3.12.13 | 2.11.0+cu128 | 12.8 | 4.48.3 | 3.3.0 |
| LLaVA-1.5-7B | 3.10.20 | 2.1.2+cu121 | 12.1 | 4.37.2 | 3.2.0 |

Complete snapshots, including transitive dependencies, are stored in:

```text
reproducibility/environments/qwen25_vl.freeze.txt
reproducibility/environments/internvl3.freeze.txt
reproducibility/environments/llava15.freeze.txt
```

The snapshots record provenance. The InternVL snapshot contains a local
`flash_attn` wheel path and editable project path, and the LLaVA snapshot
contains its upstream Git revision. Relocate the local paths or install the
equivalent wheel before installing this repository with `pip install -e .`.
A best-effort environment reconstruction is:

```bash
python3.12 -m venv Qwen2.5-VL-7B/.venv
Qwen2.5-VL-7B/.venv/bin/python -m pip install \
  -r reproducibility/environments/qwen25_vl.freeze.txt
Qwen2.5-VL-7B/.venv/bin/python -m pip install -e .
```

Use Python 3.12 for InternVL and Python 3.10 for LLaVA, replacing the freeze
file accordingly. Exact CUDA wheels may require the original package index or
a wheel archive.

To refresh a snapshot without changing the experiment:

```bash
Qwen2.5-VL-7B/.venv/bin/python scripts/capture_environment.py \
  --output reproducibility/environments/qwen25_vl.freeze.txt
```

## 3. External inputs

The reference configuration uses these paths:

| Input | Reference path | Approximate size |
|---|---|---:|
| Qwen2.5-VL-7B-Instruct | `/data01/cd_workspace/llm/Qwen2.5-VL-7B-Instruct` | 16 GB |
| InternVL3-8B | `/data01/cd_workspace/llm/InternVL3-8B` | 15 GB |
| LLaVA-1.5-7B weights | `/data01/cd_workspace/llm/llava-v1.5-7b` | 13 GB |
| LLaVA source | `/data01/cd_workspace/llm/LLaVA` | 35 MB |
| Prometheus-7B-v2.0 judge | `/data01/cd_workspace/llm/prometheus-7b-v2.0` | 14 GB |
| EarthVQA annotations | `/data01/cd_workspace/llm/EarthVQA/Train_QA.json` | 12 MB |
| EarthVQA images | `/data01/cd_workspace/llm/EarthVQA/Train/images_png` | 3.8 GB |
| LingoQA validation data | `/data01/cd_workspace/llm/LingoQA` | 226 MB |
| VQAv2 validation shards | `/data01/cd_workspace/llm/VQAv2` | 184 MB |

The LLaVA source revision is:

```text
c121f0432da27facab705978f83c4ada465e46fd
```

LLaVA also requires `openai/clip-vit-large-patch14-336` in the Hugging Face
cache when running offline. Model and dataset paths may be relocated, but all
corresponding YAML paths must be changed together.

Verify configuration files, model metadata, and dataset annotation files:

```bash
sha256sum -c reproducibility/reference_sha256.txt
```

This manifest hashes model configuration/index files, not every weight shard.
For archival publication, create and retain a complete manifest:

```bash
find /data01/cd_workspace/llm \
  -type f -print0 | sort -z | xargs -0 sha256sum \
  > external_artifacts_full.sha256
sha256sum external_artifacts_full.sha256
```

The reference dataset inventory is 2,522 EarthVQA image files, 501 files below
the LingoQA root, and two VQAv2 Parquet shards. File counts and checksums must
match before attributing metric changes to code.

Dataset selection is deterministic. EarthVQA keeps at most 5,000 samples of
`Comprehensive Analysis`, with at most two questions per image. LingoQA uses
all rows in `val.parquet` and five images per question. VQAv2 takes at most
5,000 rows from the two validation Parquet shards. All three preserve adapter
iteration order before grouped splitting.

## 4. Canonical configuration

The only main-experiment entry point is:

```text
configs/experiments/current.yaml
```

Validate all nine jobs before running GPU work:

```bash
PYTHONPATH=src Qwen2.5-VL-7B/.venv/bin/python \
  -m detect_sdc.cli config validate configs/experiments/current.yaml
sha256sum -c reproducibility/reference_sha256.txt
```

Global settings are:

- 10 double-bit activation fault runs;
- answer suffix `The answer must be limited to 30 words.` for every model;
- Qwen image pixel range 200,704 to 1,003,520;
- InternVL image size 448, 1 to 12 tiles, thumbnail enabled, BF16 and FlashAttention;
- Prometheus judge maximum 200 generated tokens;
- maximum 50 generated tokens and deterministic greedy decoding;
- 64-dimensional `project` projection;
- seeds 42 for projection, fault injection, Mapping, feature split, and
  Detector unless overridden below;
- `orig_id`-grouped 85/15 feature train/validation split;
- six layer pairs and 72 discrepancy features;
- binary target
  `(pred_answer != clean_answer) and (significance == 2)`;
- XGBoost `learning_rate=0.01`, `max_depth=6`,
  `n_estimators=10000`, `early_stopping_rounds=500`, and CPU execution.

Job-specific overrides that must not be silently normalized are:

| Jobs | Override |
|---|---|
| Qwen/InternVL LingoQA | Keep all records for fault runs 0 and 1; later runs retain SDC records |
| Qwen/InternVL VQAv2 Mapping | Random split, validation ratio 0.1, test ratio 0.1, cosine weight 0.1, patience 15 |
| LLaVA LingoQA projection | Seed 1234 |
| LLaVA LingoQA Mapping | Cosine weight 2.0, patience 15 |
| LLaVA LingoQA Predictor | Hidden size 1024, 8 residual blocks |

All other Qwen and InternVL Predictors use hidden size 64 and 4 residual
blocks. Other LLaVA Predictors use hidden size 256 and 4 residual blocks.
Mapping runs in FP32; no AMP or `GradScaler` is used.

## 5. Required non-Git artifacts

Preserve the following for each of the nine jobs:

```text
<model>/<dataset>/json/profile.json
<model>/<dataset>/json/mapping.jsonl
<model>/<dataset>/model/*mapping_model.pt
<model>/<dataset>/json/injection.jsonl
<model>/<dataset>/json/labels.jsonl
<model>/<dataset>/train_data/*train_set.csv
<model>/<dataset>/train_data/*valid_set.csv
<model>/<dataset>/output/train_with_nan/
```

The current injection plus labeled JSONL footprint is approximately:

| Model | EarthVQA | LingoQA | VQAv2 |
|---|---:|---:|---:|
| Qwen2.5-VL-7B | 9.7 GB | 8.5 GB | 6.8 GB |
| InternVL3-8B | 12.0 GB | 12.1 GB | 10.6 GB |
| LLaVA-1.5-7B | 1.9 GB | 2.8 GB | 1.2 GB |

Keep the files byte-for-byte with a SHA-256 manifest. `sample_uid` is the
stable record identity; fault records use `<orig_id>:fault:<run_index>`.
Never reconstruct a split from row position when its train/validation CSV or
fixed-cohort manifest is available.

Recommended archive command:

```bash
find Qwen2.5-VL-7B InternVL3-8B llava-v1.5-7B \
  -type f \( -name '*.jsonl' -o -name '*.pt' -o -name '*.csv' \
  -o -name '*.ubj' -o -name 'metrics_summary.json' \) \
  -print0 | sort -z | xargs -0 sha256sum \
  > experiment_artifacts_full.sha256
```

## 6. Full pipeline

Use the matching interpreter for each model:

```bash
QWEN_PY=Qwen2.5-VL-7B/.venv/bin/python
INTERNVL_PY=InternVL3-8B/.venv/bin/python
LLAVA_PY=llava-v1.5-7B/.venv/bin/python
export PYTHONPATH=src
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Run one fresh job, stage by stage:

```bash
export CUDA_VISIBLE_DEVICES=2
JOB=qwen25_vl_earthvqa
PYTHON="$QWEN_PY"

for STAGE in profile collect_mapping train_mapping inject label \
  featurize train_detector report; do
  "$PYTHON" -m detect_sdc.cli run \
    --config configs/experiments/current.yaml \
    --job "$JOB" \
    --stage "$STAGE" \
    --device cuda:0
done
```

Use these job/interpreter assignments:

```text
QWEN_PY:
  qwen25_vl_earthvqa
  qwen25_vl_lingoqa
  qwen25_vl_vqav2

INTERNVL_PY:
  internvl3_earthvqa
  internvl3_lingoqa
  internvl3_vqav2

LLAVA_PY:
  llava15_earthvqa
  llava15_lingoqa
  llava15_vqav2
```

`--overwrite` intentionally destroys the selected stage's prior output and
must only be used after archiving it. Interrupted injection is resumed at run
granularity:

```bash
"$PYTHON" -m detect_sdc.cli run \
  --job "$JOB" --stage inject --device cuda:0 \
  --resume-injection-from-run 4
```

Before a long run, use `--dry-run` for each stage. After feature extraction,
verify the reported `orig_id` overlap is zero. After Detector training, use
`output/train_with_nan/metrics_summary.json`; the older
`output/metrics_summary.json` is not the canonical result.

## 7. Labeling and determinism

Prometheus labeling uses greedy generation (`do_sample=False`) and accepts a
score only when the final line contains exactly one of `[RESULT] 0`,
`[RESULT] 1`, or `[RESULT] 2`. Identical answers bypass judge inference and are
marked non-significant. Parse failures are written to a separate
`*_prometheus_parse_failed.jsonl` file and must not be silently assigned a
numeric label.

The pipeline seeds Python, NumPy, PyTorch, CUDA, projection generation,
fault selection, Mapping splits, XGBoost, and bootstrap sampling. CuDNN
deterministic mode is enabled for model generation. Nevertheless, different
GPU drivers, CUDA kernels, FlashAttention builds, or model-library versions can
change generated text. Qwen showed occasional answer drift even between
nominally identical runs. Reproduction therefore requires:

1. Exact environment and model checksums.
2. Persisted `profile.json`, `labels.jsonl`, and `sample_uid` cohorts.
3. Exact agreement for deterministic downstream CSV/Detector runs.
4. Aggregate confidence-interval agreement, rather than a bitwise claim, for
   full GPU regeneration.

## 8. Split contracts

Main feature data is split with `GroupShuffleSplit`, grouping by `orig_id`,
validation ratio 0.15, and seed 42. XGBoost then takes a grouped 0.15 holdout
from the training CSV for early stopping; the original validation CSV remains
the final evaluation set. All-feature-NaN rows stay in Detector training.
`Non-all-NaN` is an evaluation slice of that same fitted model.

Mapping uses seed 42. Its default split is sequential with validation ratio
0.15, test ratio 0.1, and `test_ratio_in_train=0.15`; the two VQAv2 overrides
are listed in Section 4.

Online-step, layer-pair, and feature ablations must reuse the main experiment's
train/validation `sample_uid` sets. Their `fixed_cohort.json` files are part of
the result provenance. Re-running `GroupShuffleSplit` from reordered JSONL is
not equivalent.

The method comparison uses a separate contract:

- main training CSV IDs form the fit/calibration pool;
- 20% of those IDs become calibration IDs using SHA-256 stable ranking,
  seed 42;
- main validation CSV IDs are the test IDs;
- sets are disjoint by `orig_id`;
- at most 1,000 SDC and 1,000 Non-SDC records are sampled per job;
- every Significant SDC test record is retained;
- thresholds are selected on calibration data at a target 1% Non-SDC FPR;
- final metrics use the untouched test split;
- confidence intervals use 10,000 bootstrap replicates, seed 20260825.

## 9. Ablation commands

These experiments start from canonical `labels.jsonl` and feature CSV files.
Use the Qwen environment for CPU-only aggregation/training unless a script
loads a model.

Projection preservation:

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=2 "$QWEN_PY" \
  scripts/analyze_projection_preservation.py \
  --device cuda:0 --max-samples 128 --max-new-tokens 50 \
  --projection-dim 64 --seed 42 --bootstrap-replicates 2000
```

Leave-One-Pair-Out, layer-layout, and prefix-step ablations:

```bash
PYTHONPATH=src "$QWEN_PY" scripts/run_qwen_leave_one_pair_out.py \
  --model Qwen2.5-VL-7B --dataset EarthVQA

PYTHONPATH=src "$QWEN_PY" scripts/run_layer_pair_design_ablation.py \
  --model Qwen2.5-VL-7B --dataset EarthVQA

PYTHONPATH=src "$QWEN_PY" scripts/run_online_step_ablation.py \
  --model Qwen2.5-VL-7B --dataset EarthVQA
```

Repeat the last three commands across the model names
`Qwen2.5-VL-7B`, `InternVL3-8B`, `LLaVA-1.5-7B` and dataset names
`EarthVQA`, `LingoQA`, `VQAv2`. The prefix-step script evaluates
`K={1,2,4,8,12,16,24,32,50}` and writes its fixed K=50 cohort.

Feature ablations and XGBoost depth:

```bash
PYTHONPATH=src "$QWEN_PY" scripts/run_feature_group_ablation.py \
  --model all --dataset all
PYTHONPATH=src "$QWEN_PY" scripts/run_compact_feature_ablation.py \
  --model all --dataset all
PYTHONPATH=src "$QWEN_PY" scripts/run_online_cosine_mean_depth_ablation.py \
  --model all --dataset all
```

Do not add overwrite flags when validating existing artifacts. Add
`--overwrite`, `--overwrite-features`, or `--overwrite-detectors` only for an
intentional clean rerun.

## 10. Online overhead

The formal protocol uses LingoQA, batch size 1, 50 measured samples, all-sample
shape prewarm, five warmup samples, two repeats, K=2, and both mode orders.
No other process should use the measured GPU. GPU clocks and power policy
should remain unchanged across paired runs.

Run each model with its own interpreter and a free physical GPU:

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=2 "$QWEN_PY" scripts/benchmark_online_overhead.py \
  --job qwen25_vl_lingoqa --device cuda:0 \
  --samples 50 --warmup-samples 5 --repeats 2 --online-steps 2 \
  --feature-profile full --mode-order forward \
  --output-root analysis/online_overhead_20260814/qwen25_vl

PYTHONPATH=src CUDA_VISIBLE_DEVICES=2 "$QWEN_PY" scripts/benchmark_online_overhead.py \
  --job qwen25_vl_lingoqa --device cuda:0 \
  --samples 50 --warmup-samples 5 --repeats 2 --online-steps 2 \
  --feature-profile full --mode-order reverse \
  --output-root analysis/online_overhead_reverse_20260814/qwen25_vl
```

Replace the job, interpreter, and output leaf for InternVL and LLaVA. Summarize
the six runs with 10,000 paired bootstrap replicates:

```bash
PYTHONPATH=src "$QWEN_PY" scripts/summarize_online_overhead.py \
  --forward-root analysis/online_overhead_20260814 \
  --reverse-root analysis/online_overhead_reverse_20260814 \
  --output-dir analysis/online_overhead_20260814/combined \
  --bootstrap-replicates 10000 --seed 20260814
```

Run the paired 72-D/6-D comparison in one process:

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=2 "$QWEN_PY" \
  scripts/benchmark_feature_profile_comparison.py \
  --job qwen25_vl_lingoqa --device cuda:0 \
  --samples 50 --warmup-samples 5 --repeats 2 --online-steps 2 \
  --bootstrap-replicates 10000 --seed 20260815 \
  --output-root analysis/online_feature_profile_comparison_20260815/qwen25_vl
```

Timing results are paired by sample. Report median/mean latency, throughput,
peak GPU memory, detection-ready latency, and paired-bootstrap 95% confidence
intervals from the generated `samples.csv` and `summary.json`.

## 11. Ranger-style and Dr.DNA-style comparison

The comparison configuration is
`compare_experiment/configs/detection_comparison.yaml`. These are
mechanism-matched VLM adaptations, not claims of exact original-system
reproduction.

Run one physical GPU per model:

```bash
tmux new-session -d -s compare_qwen \
  "cd /data01/cd_workspace/Detect_SDC && \
   ./compare_experiment/run_model_comparison.sh qwen25_vl 2"
tmux new-session -d -s compare_internvl \
  "cd /data01/cd_workspace/Detect_SDC && \
   ./compare_experiment/run_model_comparison.sh internvl3 3"
tmux new-session -d -s compare_llava \
  "cd /data01/cd_workspace/Detect_SDC && \
   ./compare_experiment/run_model_comparison.sh llava15 4"
```

The launcher executes `profile_baselines`, `build_replay_manifest`,
`replay_detection`, and `evaluate_results`. Replay appends records atomically
and resumes by `sample_uid`. After all nine jobs:

```bash
PYTHONPATH=src:. "$QWEN_PY" \
  -m compare_experiment.summarize_results
```

Canonical outputs:

```text
compare_experiment/results/<job>/profiles.json
compare_experiment/results/<job>/replay_manifest.jsonl
compare_experiment/results/<job>/replay_manifest.summary.json
compare_experiment/results/<job>/detection_scores.jsonl
compare_experiment/results/<job>/evaluation/metrics.json
compare_experiment/results/summary/detailed_metrics.csv
compare_experiment/results/summary/macro_average_metrics.csv
```

## 12. Figure regeneration

Portable figure data, its SHA-256 manifest, and the no-GPU replot workflow are
documented in [`figure_data.md`](figure_data.md). On a second machine, the
complete portable workflow is:

```bash
python scripts/replot_paper_figures.py
```

After changing raw experiments, refresh the compact data once on the source
machine:

```bash
Qwen2.5-VL-7B/.venv/bin/python \
  scripts/export_portable_figure_data.py
```

The export streams raw JSONL and full feature CSV files but stores only the
minimal aggregate values required by existing figures. The nine-job quadrant
comparison removes non-finite `before` and `after` values. Do not combine it
with older outputs that classified NaN/Inf as large deviation.

## 13. Verification checklist

Before accepting a reproduction:

1. Record the Git commit and confirm `git diff --check` is clean.
2. Validate `current.yaml` and run `sha256sum -c` on the reference manifest.
3. Record GPU model, driver, CUDA toolkit, Python, PyTorch, and Transformers.
4. Confirm the nine model-dataset jobs and all external paths exist.
5. Confirm `project`, 64 dimensions, 10 fault runs, and the job overrides.
6. Confirm no train/validation/calibration/test `orig_id` overlap.
7. Preserve the exact UID lists and SHA-256 hashes of large artifacts.
8. Check parse-failure counts before feature extraction.
9. Compare generated summary JSON/CSV files, not values copied from plots.
10. For overhead experiments, retain raw per-sample CSVs and both mode orders.
11. Run the unit tests:

```bash
PYTHONPATH=src Qwen2.5-VL-7B/.venv/bin/python \
  -m unittest discover tests
```

The reference suite contains 73 passing tests. A reproduction is incomplete if
only the final table is retained without its config, split identities,
environment snapshot, and raw summary artifacts.
