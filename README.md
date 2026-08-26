# Detect SDC

Detect SDC evaluates silent data corruption in multimodal language models. All
runtime code lives under `src/detect_sdc/`. The `Qwen2.5-VL-7B/` and
`llava-v1.5-7B/` trees contain only configured inputs and experiment artifacts.

## Pipeline

The canonical pipeline stages are:

1. `profile`: generate clean baseline answers.
2. `collect_mapping`: collect inter-layer mapping samples.
3. `train_mapping`: train the residual mapping model.
4. `inject`: inject faults and collect telemetry.
5. `label`: assign Prometheus quality and significance labels.
6. `featurize`: aggregate telemetry into tabular features.
7. `train_detector`: train the significant-SDC detector.
8. `report`: load the detector metrics artifact.

## Architecture

- `adapters/datasets`: EarthVQA, LingoQA, and VQAv2 sample normalization.
- `adapters/models`: Qwen2.5-VL and LLaVA 1.5 deterministic generation.
- `pipeline`: configured profile, mapping, injection, and training dispatch.
- `fault_injector.py`: canonical activation and weight bit-flip implementation.
- `labeling.py`: streaming Prometheus labeling with atomic outputs.
- `profiler.py`: canonical model instrumentation and telemetry implementation.
- `features`: shared 72-feature extraction and stable sample identities.
- `splitting.py`: the only production `orig_id` grouped split implementation.
- `detector`: binary significant-SDC training, evaluation, and layer-pair experiments.
- `mapping`: one shared mapping architecture and trainer with configured job
  profiles.

Torch, Transformers, Qwen, and LLaVA dependencies are imported lazily by the
model and labeling adapters. Configuration inspection and dry-runs do not load
model weights.

`detect-sdc config validate configs/experiments/current.yaml` checks the full
execution contract without loading model weights: matrix/job coverage, adapter
and trainer imports, callable arguments, input paths, stage output suffixes,
mapping dimensions and layer counts, split parameters, and fault-run bounds.

## Canonical labels

- `quality_score`: `0..2`, where higher is better.
- `significance`: `0..2`, where higher is more severe.
- `significant_sdc_target`: `pred_answer != clean_answer and significance == 2`.

Parse failures are represented by an explicit status instead of the numeric
sentinel `-1`.

## Configuration

Model, dataset, and experiment configuration lives under `configs/`:

```text
configs/
├── models/
├── datasets/
└── experiments/
```

Validate the current experiment matrix without running a model:

```bash
PYTHONPATH=src python -m detect_sdc.cli config validate \
  configs/experiments/current.yaml
```

Run one stage without executing GPU work:

```bash
PYTHONPATH=src python -m detect_sdc.cli run \
  --job qwen25_vl_earthvqa \
  --stage inject \
  --dry-run
```

`--stage` is repeatable. `detect-sdc` (or `python -m detect_sdc.cli`) is the
only supported command entry point.

The GPU stages use the same orthogonal model and dataset adapters:

```bash
PYTHONPATH=src python -m detect_sdc.cli run \
  --job qwen25_vl_earthvqa \
  --stage collect_mapping \
  --overwrite

PYTHONPATH=src python -m detect_sdc.cli run \
  --job qwen25_vl_earthvqa \
  --stage inject \
  --overwrite
```

Mapping collection writes adjacent-layer supervision rows atomically. Injection
loads the clean answers and mapping checkpoint, runs one clean pass followed by
the configured fault passes, and keeps complete or SDC-only runs according to
`injection.retain_all_fault_runs`.

## Labeling

Prometheus output is accepted only when the final line contains exactly one
`[RESULT] 0`, `[RESULT] 1`, or `[RESULT] 2`. Identical clean and predicted
answers skip judge inference and receive `quality_score=2`, `significance=0`.
Large JSONL inputs are processed in bounded chunks and written atomically.

```bash
PYTHONPATH=src python -m detect_sdc.cli label \
  --job qwen25_vl_earthvqa \
  --device cuda:0 \
  --batch-size 64
```

## Feature extraction

All six model and dataset combinations use the same streaming extractor and
`orig_id`-grouped splitter. Run a configured job with:

```bash
PYTHONPATH=src python -m detect_sdc.cli featurize \
  --job qwen25_vl_earthvqa
```

The available jobs are defined under `featurization.jobs` in
`configs/experiments/current.yaml`. Use `--train-output` and `--valid-output`
to write a comparison run without replacing the configured CSV files.

The shared extractor enforces:

- 72 telemetry features from the configured layer pairs and last 50 steps;
- finite-value aggregation, while preserving all-feature-NaN rows;
- canonical ternary `label` and binary `significant_sdc_target`;
- stable `sample_uid` values based on `orig_id` and fault metadata;
- deterministic grouped splitting with zero `orig_id` overlap.

Exact duplicate source samples are collapsed by stable UID. A UID collision
with different feature content is rejected instead of being silently merged.

## Detector

All six jobs train one binary XGBoost target: `significant_sdc_target`. Rows
whose 72 features are all NaN remain in training. The same fitted model reports
`valid_full_metrics` and `valid_non_all_nan_metrics`. Train/test splitting is
deterministic and grouped by `orig_id`; overlap is rejected.

```bash
PYTHONPATH=src python -m detect_sdc.cli train \
  --job qwen25_vl_earthvqa
```

The layer-pair sweep is an experimental backend under
`detect_sdc.detector.layer_pair_sweep`; it preserves threshold and ROC analysis
but defaults to the canonical keep-all-NaN policy and significant-SDC target.

## Runtime stages

Mapping collection and fault injection are model-agnostic stages built from
ModelAdapter, DatasetAdapter, Profiler, and FaultInjector. Both stages stream to
temporary JSONL files and atomically publish successful outputs. The inject
stage loads each 7B model once for its clean run and all configured fault runs.

Every model/dataset artifact directory uses the same stage filenames under
`json/`: `profile.json`, `mapping.jsonl`, `injection.jsonl`, and `labels.jsonl`.

Mapping-model architecture, profiler projection, fault run count, retained
full runs, bit count, and random seed are explicit model/job configuration.
Mapping-model training is also a package stage and atomically publishes its
checkpoint; trainer-specific split and optimization settings remain explicit
configuration.

## Baseline

The pre-refactor inputs and metrics are frozen in:

```text
baselines/pre_refactor_20260727/baseline.yaml
```

It contains:

- the pre-refactor Git revision;
- SHA-256 checksums and sizes for active labeled JSONL and feature CSV files;
- CSV row, label, significance, and target distributions;
- complete snapshots of available `metrics_summary.json` files;
- the Python and dependency versions used during capture.

Regenerate it only when intentionally defining a new baseline:

```bash
PYTHONPATH=src python -m detect_sdc.cli baseline freeze \
  --spec configs/baseline.yaml \
  --output baselines/pre_refactor_20260727/baseline.yaml
```

## Development

Install the shared package and training dependencies:

```bash
python -m pip install -e '.[train,dev]'
```

Run the unified CLI from an environment compatible with the selected model.
The local LLaVA fork requires its pinned Transformers and SentencePiece
versions. The LLaVA vision tower
`openai/clip-vit-large-patch14-336` must be available in the Hugging Face cache
for offline runs.

Run the package tests:

```bash
python -m pytest
```

## Pipeline invariants

Changes to the pipeline must preserve:

- grouping by `orig_id`;
- zero overlap between train/test/valid groups;
- full and non-all-feature-NaN evaluation slices;
- target label counts;
- prediction and metric output schemas.
