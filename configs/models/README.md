# Model Configuration YAML

This directory contains model-level defaults used by the configured SDC
pipeline jobs. A model YAML describes how to load a model, how to generate
answers, how to instrument activations, how to train the mapping model, and how
to run fault injection.

Concrete experiment jobs are defined in `configs/experiments/current.yaml`.
Canonical jobs only override instrumentation values that are part of the
recorded experimental protocol. Mapping architecture and training parameters
are shared across datasets for a model family.

## Top-Level Fields

```yaml
name: qwen25_vl
adapter: detect_sdc.adapters.models.qwen25_vl.Qwen25VLAdapter
model_path: /path/to/model
```

- `name`: Human-readable model identifier.
- `adapter`: Python class path for the model adapter. The class must implement
  `from_config()`, `load()`, `generate()`, `model`, and `close()`.
- `model_path`: Local checkpoint or Hugging Face model path used by the
  adapter.

LLaVA-specific fields:

- `model_base`: Optional base model path for LLaVA loaders. Use `null` when the
  checkpoint is self-contained.
- `source_path`: Local LLaVA source tree. The adapter adds this path to
  `sys.path` before importing LLaVA runtime modules.

Qwen-specific fields:

- `vision.min_pixels`: Minimum visual tokenization pixel budget passed to the
  Qwen processor.
- `vision.max_pixels`: Maximum visual tokenization pixel budget passed to the
  Qwen processor.

InternVL-specific fields:

- `vision.image_size`: Tile size used by InternVL dynamic image preprocessing.
- `vision.min_tiles`: Minimum number of image tiles.
- `vision.max_tiles`: Maximum number of image tiles.
- `vision.use_thumbnail`: Add a thumbnail tile when the image is split into
  multiple tiles.
- `runtime.torch_dtype`: Model and image tensor dtype. Supported values are
  `bfloat16`, `float16`, and `float32`.
- `runtime.low_cpu_mem_usage`: Passed to `AutoModel.from_pretrained()`.
- `runtime.use_flash_attn`: Requests flash attention when supported by the
  local InternVL runtime.
- `runtime.load_in_8bit`: Enables 8-bit loading. This requires the proper
  quantization dependencies in the selected Python environment.
- `runtime.device_map`: Optional Transformers device map, for example `auto`.
  Use `null` for normal single-device loading.

## Generation

```yaml
generation:
  max_new_tokens: 50
  deterministic: true
  answer_suffix: The answer must be limited to 30 words.
```

- `max_new_tokens`: Maximum number of tokens generated per answer.
- `deterministic`: Declares deterministic generation. Current adapters use
  non-sampling generation (`do_sample=False`) for deterministic behavior.
- `answer_suffix`: Instruction injected into the model prompt to constrain the
  answer style or length.

## Instrumentation

```yaml
instrumentation:
  layer_count: 28
  projection_dim: 64
  projection_method: project
  seed: 42
```

- `layer_count`: Number of transformer layers exposed to profiler and mapping
  logic. This must match `injection.mapping_kwargs.num_layers`.
- `projection_dim`: Size of each projected activation vector. This must match
  `injection.mapping_kwargs.x_dim`.
- `projection_method`: Activation reduction/projection method. Supported values
  are `project`, `max`, `min`, and `mean`.
- `seed`: Random seed used by profiler/projection logic.

## Mapping Training

```yaml
mapping_training:
  trainer: detect_sdc.mapping.train_model
  kwargs:
    batch_size: 2048
    lr: 0.0005
    weight_decay: 0.0001
    epochs: 500
    num_workers: 8
    valid_ratio: 0.15
    test_ratio: 0.15
    cosine_weight: 1.0
    early_stop_patience: 10
    scheduler_enabled: true
    scheduler_patience: 5
    scheduler_factor: 0.5
    min_lr: 0.000001
    pin_memory: true
    persistent_workers: true
    final_metrics: detailed
    seed: 42
```

- `trainer`: Python function path for mapping model training.
- `batch_size`: Training batch size.
- `lr`: AdamW learning rate.
- `weight_decay`: AdamW weight decay.
- `epochs`: Maximum number of epochs.
- `num_workers`: DataLoader worker count.
- `valid_ratio`: Group-disjoint validation ratio within the outer Fit split.
- `test_ratio`: Group-disjoint Mapping test ratio within the outer Fit split.
- `cosine_weight`: Weight of cosine loss in the mapping regression objective.
- `early_stop_patience`: Stop after this many non-improving selection epochs.
- `scheduler_enabled`: Whether to enable `ReduceLROnPlateau`.
- `scheduler_patience`: Number of non-improving epochs before reducing LR.
- `scheduler_factor`: LR reduction factor.
- `min_lr`: Lower bound for scheduler LR.
- `pin_memory`: Enable pinned host memory for DataLoader.
- `persistent_workers`: Keep DataLoader workers alive across epochs when
  `num_workers > 0`.
- `final_metrics`: `detailed` reports RMSE/MSE/cosine/rank metrics; `loss`
  reports training-style loss metrics.
- `seed`: Random seed for split and training behavior.

## Fault Injection

```yaml
injection:
  mapping_class: detect_sdc.mapping.LayerAwareResidualMLP
  mapping_kwargs:
    x_dim: 64
    num_layers: 28
    layer_emb_dim: 16
    hidden_dim: 64
    num_blocks: 8
    dropout: 0.1
  fault_runs: 10
  num_bits: 2
  bit_policy: random
  seed: 42
```

- `mapping_class`: Python class path used to reconstruct the trained mapping
  model checkpoint during injection.
- `mapping_kwargs.x_dim`: Mapping model input/output dimension. Must equal
  `instrumentation.projection_dim`.
- `mapping_kwargs.num_layers`: Number of layer embeddings. Must equal
  `instrumentation.layer_count`.
- `mapping_kwargs.layer_emb_dim`: Source/target layer embedding dimension.
- `mapping_kwargs.hidden_dim`: Hidden width of the residual MLP.
- `mapping_kwargs.num_blocks`: Number of residual MLP blocks.
- `mapping_kwargs.dropout`: Dropout probability used by the mapping model.
- `fault_runs`: Number of fault-injection runs per clean sample.
- Every clean and fault run is retained. Selective SDC-only retention is not
  supported because it biases FPR and deployment-precision estimates.
- `num_bits`: Number of bits flipped per injected fault.
- `bit_policy`: Candidate-bit policy. `random` uses every bit;
  `mantissa_only` uses the full mantissa; `low_mantissa` uses its low-order
  subset; `low_exponent` uses the full mantissa plus the five least-significant
  exponent bits and excludes the sign bit.
- `seed`: Random seed for fault selection.

## Adding A Model

To add a new model:

1. Add a model adapter under `src/detect_sdc/adapters/models/`.
2. Add a YAML file in this directory.
3. Register the model YAML in `configs/experiments/current.yaml`.
4. Add execution jobs and output paths in `configs/experiments/current.yaml`.
5. Run configuration validation:

```bash
PYTHONPATH=src python -m detect_sdc.cli config validate \
  configs/experiments/current.yaml
```

Keep model directories such as `Qwen2.5-VL-7B/` or `llava-v1.5-7B/` as artifact
locations only. Core runtime code belongs in `src/detect_sdc/`.
