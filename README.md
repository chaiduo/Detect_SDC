# Detect SDC

Detect SDC evaluates severity-aware silent data corruption detection in
multimodal language models. Runtime code lives under `src/detect_sdc/`; formal
ICLR-v2 artifacts are written under `artifacts/iclr_v2/`. Historical model
directories and results are retained as baselines and are not overwritten.

The active redesign and rerun contract is documented in
[`docs/sieve_iclr_revision_plan.md`](docs/sieve_iclr_revision_plan.md).

## Canonical protocol

Each dataset contributes 5,000 inputs. One frozen manifest assigns semantic
entities to Fit, Calibration, and Final test:

- EarthVQA groups by image;
- LingoQA groups all frames of one question;
- VQAv2 groups all questions for one image.

All model families reuse the same dataset manifests. Fit trains Mapping and
XGBoost, Calibration selects the operating threshold, and Final test is only
used for reporting.

Create or validate the manifests before GPU work:

```bash
PYTHONPATH=src python -m detect_sdc.cli split --dataset earthvqa
PYTHONPATH=src python -m detect_sdc.cli split --dataset lingoqa
PYTHONPATH=src python -m detect_sdc.cli split --dataset vqav2
```

The configured job stages are:

1. `profile`: generate fault-free baseline answers.
2. `collect_mapping`: collect clean Fit inter-layer samples.
3. `train_mapping`: train a group-disjoint Mapping model.
4. `inject`: retain one clean execution and every configured fault execution.
5. `label`: assign Prometheus quality and significance labels.
6. `featurize`: materialize Fit, Calibration, and Final-test CSV files.
7. `train_detector`: train on Fit and maximize F1 on Calibration.
8. `report`: load the final metrics artifact.

## Method

- Faults are one Prefill activation double-bit flip on an eligible `nn.Linear`
  output exposed by the model adapter.
- `bit_policy` supports `random`, `mantissa_only`, `low_mantissa`, and
  `low_exponent`; the last policy samples the full mantissa plus the five
  least-significant exponent bits.
- Every fault records component, layer, operation, dtype, element, bit positions,
  bit categories, before/after values, run index, and split identity.
- The Mapping model is a layer-aware residual MLP with 64-dimensional input,
  hidden width 64, eight residual blocks, and 16-dimensional layer embeddings.
- The online representation uses six adjacent layer pairs, four discrepancy
  metrics, and mean/max/min over the first two decoding steps: 72 features.
- `significant_sdc_target` is true when the output changes and Prometheus assigns
  severity 2 relative to the fault-free response.

## Detector evaluation

The canonical XGBoost path:

- keeps all-feature-NaN rows in Fit;
- uses a group-disjoint holdout inside Fit for early stopping;
- selects the threshold maximizing Significant-SDC F1 on Calibration, using
  both target classes and breaking ties with the highest threshold;
- reports full, finite, clean, injected Non-SDC, Slight-SDC, and
  Significant-SDC cohorts on untouched Final test.

```bash
PYTHONPATH=src python -m detect_sdc.cli train \
  --job qwen25_vl_earthvqa
```

## Ranger/Dr.DNA comparison

`compare_experiment` builds clean Fit profiles, then observes Ranger-style,
Dr.DNA-style, and SIEVE signals during the same fault execution. There is no
second replay campaign or answer-mismatch filtering.

```bash
./compare_experiment/run_model_comparison.sh qwen25_vl 2
```

## Configuration and validation

```bash
PYTHONPATH=src python -m detect_sdc.cli config validate \
  configs/experiments/current.yaml

PYTHONPATH=src python -m detect_sdc.cli run \
  --job qwen25_vl_earthvqa --stage inject --dry-run
```

Configuration inspection does not load model weights. GPU stages lazily import
model-specific dependencies.

## Development

```bash
python -m pip install -e '.[train,dev]'
PYTHONPATH=src python -m unittest discover tests
```

Pipeline changes must preserve:

- frozen semantic-group manifests and zero Fit/Calibration/Test overlap;
- Mapping training on Fit only and group-disjoint internal splits;
- complete clean/fault retention;
- one shared fault execution for comparison methods;
- calibrated thresholds and untouched Final-test reporting;
- stable sample UIDs, atomic outputs, and artifact checksums.
