# Layer-Pair Subset Search

## Protocol

- Inputs: the nine completed random-bit experiments for Qwen2.5-VL,
  LLaVA-1.5, and InternVL3 on EarthVQA, LingoQA, and VQAv2.
- Candidates: all 63 non-empty subsets of the six current layer pairs.
- Each layer pair contributes 12 features.
- Feature-subset selection uses only the group-aware holdout inside Fit.
- Calibration is used only to maximize Significant-SDC F1 and freeze the
  operating threshold.
- Final Test is evaluated with that frozen threshold.
- Finite uses the same canonical cohort as the 72-feature model: samples
  whose complete 72-feature vector is not all NaN.

## Initial Six-Experiment Shared Pair Set

Before InternVL completed, the highest mean Fit-holdout F1 across the six
Qwen and LLaVA experiments was:

```text
(6,7), (22,23), (25,26), (26,27)
```

This reduces the detector from 72 to 48 features.

| Configuration | Fit-holdout F1 | Test Full F1 | Test Finite F1 |
|---|---:|---:|---:|
| Current 72D | 92.91% | 89.06% | 77.20% |
| Shared 48D | 93.34% | 90.03% | 79.53% |
| Change | +0.43 pp | +0.97 pp | +2.33 pp |

### Per-Experiment Test F1 With Shared 48D Set

| Experiment | Full 72D | Full 48D | Finite 72D | Finite 48D |
|---|---:|---:|---:|---:|
| Qwen/EarthVQA | 89.63% | 90.02% | 86.09% | 86.61% |
| Qwen/LingoQA | 96.59% | 96.84% | 94.78% | 95.17% |
| Qwen/VQAv2 | 87.01% | 88.89% | 81.56% | 84.06% |
| LLaVA/EarthVQA | 96.97% | 96.95% | 89.66% | 89.41% |
| LLaVA/LingoQA | 72.14% | 72.10% | 42.15% | 42.48% |
| LLaVA/VQAv2 | 92.04% | 95.38% | 68.97% | 79.45% |

The shared subset improves four of six Full results. The two declines are
below 0.05 percentage points.

This pair set was frozen before the InternVL results were available. On the
three InternVL experiments, it changes mean Full F1 from 95.63% to 96.04%
and mean Finite F1 from 94.71% to 95.19%. It therefore remains the strongest
prospectively checked compact design, although it does not resolve the
LLaVA/LingoQA false positives.

## Nine-Experiment Constrained Search

After all three InternVL experiments completed, the same 63 subsets were
evaluated on all nine model/dataset combinations. For a fair comparison,
the 72D baseline and each candidate were retrained by the same subset-search
script. The strongest Test-set trade-off is:

```text
(6,7), (24,25), (26,27)
```

This configuration uses 36 features. Relative to the retrained 72D baseline,
it improves LLaVA/LingoQA while limiting the loss on every other experiment
to less than 0.2 percentage points.

| Experiment | Full 72D | Full 36D | Change | Finite 72D | Finite 36D | Change |
|---|---:|---:|---:|---:|---:|---:|
| Qwen/EarthVQA | 89.63% | 91.24% | +1.61 pp | 86.09% | 88.17% | +2.08 pp |
| Qwen/LingoQA | 96.59% | 96.84% | +0.25 pp | 94.78% | 95.17% | +0.39 pp |
| Qwen/VQAv2 | 87.01% | 88.53% | +1.52 pp | 81.56% | 83.57% | +2.01 pp |
| LLaVA/EarthVQA | 96.97% | 96.97% | +0.00 pp | 89.66% | 89.66% | +0.00 pp |
| LLaVA/LingoQA | 72.14% | 93.52% | +21.38 pp | 42.15% | 80.00% | +37.85 pp |
| LLaVA/VQAv2 | 92.04% | 95.41% | +3.38 pp | 68.97% | 80.00% | +11.03 pp |
| InternVL/EarthVQA | 98.17% | 98.07% | -0.11 pp | 97.70% | 97.57% | -0.13 pp |
| InternVL/LingoQA | 95.09% | 94.94% | -0.15 pp | 94.16% | 93.97% | -0.19 pp |
| InternVL/VQAv2 | 93.62% | 94.46% | +0.84 pp | 92.27% | 93.27% | +1.00 pp |
| **Macro** | **91.25%** | **94.44%** | **+3.19 pp** | **83.04%** | **89.04%** | **+6.00 pp** |

For LLaVA/LingoQA, the 36D candidate reduces Final-Test false positives
from 109 to 2. Its Full precision is 98.81%, recall is 88.77%, and F1 is
93.52%.

### Validity Constraint

This 36D configuration was identified after inspecting Final-Test behavior.
It is therefore an exploratory candidate, not an unbiased replacement for
the current 72D main result. In addition, its LLaVA/LingoQA Fit-holdout F1
is 0.51 percentage points below the 72D baseline. No candidate both improves
LLaVA/LingoQA Fit-holdout F1 and keeps every other Fit-holdout decline within
2 percentage points.

The defensible next protocol is to freeze this 36D pair set before a new
fault campaign or a new random split/seed evaluation. Until that prospective
validation is complete, retain the validation-selected 48D configuration for
the paper's primary results and report 36D as a post-hoc ablation.

## Per-Experiment Fit-Selected Subsets

| Experiment | Fit-selected pairs | Features | Full F1 | Finite F1 |
|---|---|---:|---:|---:|
| Qwen/EarthVQA | 6-7, 22-23, 25-26, 26-27 | 48 | 90.02% | 86.61% |
| Qwen/LingoQA | 6-7, 23-24, 26-27 | 36 | 92.52% | 88.81% |
| Qwen/VQAv2 | 26-27 | 12 | 86.79% | 81.23% |
| LLaVA/EarthVQA | 6-7, 22-23, 23-24 | 36 | 90.57% | 72.22% |
| LLaVA/LingoQA | 6-7, 22-23, 24-25 | 36 | 93.79% | 80.70% |
| LLaVA/VQAv2 | 6-7 | 12 | 95.03% | 77.14% |
| InternVL/EarthVQA | 6-7, 23-24, 25-26, 26-27 | 48 | 98.17% | 97.70% |
| InternVL/LingoQA | 6-7, 22-23 | 24 | 86.80% | 84.51% |
| InternVL/VQAv2 | 6-7, 26-27 | 24 | 90.84% | 88.97% |

Per-experiment selection is unstable: it substantially improves
LLaVA/LingoQA and LLaVA/VQAv2, but degrades Qwen/LingoQA and
LLaVA/EarthVQA on Final Test. It should not be used as the default design.

## Interpretation

The frozen 48D subset is the primary detector configuration: it is the
strongest protocol-valid compact design and was successfully checked on the
previously unseen InternVL family. The new 36D subset
`(6,7), (24,25), (26,27)` is the strongest empirical trade-off for fixing
LLaVA/LingoQA across the current nine Final Tests, but it is a post-hoc
result and requires a new prospective validation before it can replace 48D.
