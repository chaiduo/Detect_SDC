# 48D Fit-Holdout Stability

## Protocol

- Repeats: 20 paired random seeds.
- Seeds: 42-61.
- Data used for selection: Fit only.
- Split unit: semantic group.
- 48D and 72D use identical train/holdout groups for each seed.
- Each holdout threshold maximizes Significant-SDC F1.
- Calibration and Final Test are not read by this experiment.

## Nine-Task Macro F1

| Configuration | Mean | Std | Min | Max |
|---|---:|---:|---:|---:|
| 72D | 93.85% | 0.84% | 91.92% | 95.51% |
| 48D | 93.97% | 0.80% | 92.59% | 95.74% |

Paired 48D - 72D difference:

- Mean: +0.115 pp
- Standard deviation: 0.345 pp
- 95% CI of mean: [-0.036, +0.266] pp
- 48D wins: 12/20 seeds

## Per-Task Validation F1

| Task | 72D Mean | 48D Mean | Paired Change | 48D Wins |
|---|---:|---:|---:|---:|
| internvl3_earthvqa | 98.10% | 98.05% | -0.055 pp | 1/20 |
| internvl3_lingoqa | 95.66% | 95.31% | -0.345 pp | 3/20 |
| internvl3_vqav2 | 96.29% | 96.35% | +0.060 pp | 8/20 |
| llava15_earthvqa | 94.64% | 94.75% | +0.108 pp | 9/20 |
| llava15_lingoqa | 92.27% | 91.99% | -0.284 pp | 5/20 |
| llava15_vqav2 | 93.95% | 94.71% | +0.758 pp | 9/20 |
| qwen25_vl_earthvqa | 93.66% | 93.91% | +0.253 pp | 7/20 |
| qwen25_vl_lingoqa | 94.28% | 94.06% | -0.219 pp | 3/20 |
| qwen25_vl_vqav2 | 85.82% | 86.58% | +0.760 pp | 10/20 |

## Interpretation

The paired 95% confidence interval crosses zero. The result supports 48D as a compact configuration with comparable validation performance, but not as significantly better than 72D.
