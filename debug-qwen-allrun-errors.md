status: OPEN
session_id: qwen-allrun-errors

# Debug Record: Qwen Allrun Errors

## Hypotheses
- H1: CUDA visibility and `DEVICE` disagree, so a script selects an unavailable GPU.
- H2: Runtime environment is missing Python packages or uses the wrong interpreter.
- H3: Dataset/model/checkpoint paths are missing or still point to stale outputs.
- H4: One of the dataset-specific scripts still has an argument mismatch with `allrun.sh`.

## Evidence Log
- EarthVQA session: Step 4 fails while loading `LayerAwareResidualMLP`; checkpoint has hidden dim 256 and 4 blocks, but sem instantiates a wider/deeper model.
- LingoQA session: Step 4 fails with the same `LayerAwareResidualMLP` state_dict mismatch; checkpoint has hidden dim 256 and 4 blocks, sem instantiates hidden dim 1024 and 8 blocks.
- VQAV2 session: Step 3 fails in `F.cross_entropy(logits, target)` with `RuntimeError: 0D or 1D target tensor expected, multi-target not supported`.
- cd session: an older LingoQA run first missed `torchvision`, then after installing it hit a CUDA driver/runtime mismatch. The active dataset sessions show code-level errors above.

## Fix Log
- Fixed EarthVQA semantic detection to instantiate `LayerAwareResidualMLP` with the same architecture used by training: `hidden_dim=256`, `num_blocks=4`.
- Fixed LingoQA semantic detection to instantiate `LayerAwareResidualMLP` with the same architecture used by training: `hidden_dim=256`, `num_blocks=4`.
- Fixed VQAv2 mapping training to treat `y` as a 64-dimensional regression target, not a class id:
  - `y` tensor dtype changed to `float32`.
  - model head changed from classifier logits to 64-dimensional residual prediction.
  - training/evaluation loss changed from cross entropy to MSE + cosine loss.
  - final metrics changed to RMSE, cosine similarity, mean values, and rank hamming.

## Verification
- `py_compile` passed for top-level Qwen dataset scripts.
- `bash -n` passed for all three `allrun.sh` files.
- EarthVQA checkpoint loads successfully into the fixed semantic mapping model.
- LingoQA checkpoint loads successfully into the fixed semantic mapping model.
- VQAv2 one-epoch CPU smoke training reaches final metrics and the smoke checkpoint loads into the fixed mapping model.
