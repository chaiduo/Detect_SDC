# Retraining after filtering deviation above 1e+12

## Protocol

- Retain every clean row.
- Retain injected rows only when values are finite and `abs(after-before) <= 1e+12`.
- Apply the same rule independently to Fit, Calibration, and Final Test.
- Retrain XGBoost on filtered Fit and maximize F1 on filtered Calibration.
- Compare against the original model on the exact same filtered Final Test.

## Per-job results

| Job | Cohort | Positives | Original R/P/F1 | Retrained R/P/F1 | Delta F1 |
|---|---|---:|---:|---:|---:|
| qwen25_vl_earthvqa | full | 37 | 10.81/33.33/16.33 | 10.81/100.00/19.51 | +3.19 pp |
| qwen25_vl_earthvqa | finite | 37 | 10.81/33.33/16.33 | 10.81/100.00/19.51 | +3.19 pp |
| qwen25_vl_lingoqa | full | 13 | 15.38/100.00/26.67 | 0.00/0.00/0.00 | -26.67 pp |
| qwen25_vl_lingoqa | finite | 13 | 15.38/100.00/26.67 | 0.00/0.00/0.00 | -26.67 pp |
| qwen25_vl_vqav2 | full | 30 | 10.00/10.34/10.17 | 10.00/10.00/10.00 | -0.17 pp |
| qwen25_vl_vqav2 | finite | 30 | 10.00/10.34/10.17 | 10.00/10.00/10.00 | -0.17 pp |
| llava15_earthvqa | full | 7 | 28.57/33.33/30.77 | 28.57/28.57/28.57 | -2.20 pp |
| llava15_earthvqa | finite | 7 | 28.57/33.33/30.77 | 28.57/28.57/28.57 | -2.20 pp |
| llava15_lingoqa | full | 20 | 15.00/2.68/4.55 | 10.00/1.80/3.05 | -1.49 pp |
| llava15_lingoqa | finite | 20 | 15.00/2.68/4.55 | 10.00/1.80/3.05 | -1.49 pp |
| llava15_vqav2 | full | 15 | 13.33/13.33/13.33 | 0.00/0.00/0.00 | -13.33 pp |
| llava15_vqav2 | finite | 15 | 13.33/14.29/13.79 | 0.00/0.00/0.00 | -13.79 pp |

## Macro average

| Cohort | Original R/P/F1 | Retrained R/P/F1 | Delta F1 |
|---|---:|---:|---:|
| full | 15.52/32.17/16.97 | 9.90/23.40/10.19 | -6.78 pp |
| finite | 15.52/32.33/17.05 | 9.90/23.40/10.19 | -6.86 pp |

## Pooled micro average

| Cohort | Original R/P/F1 | Retrained R/P/F1 | Delta F1 |
|---|---:|---:|---:|
| full | 13.11/9.09/10.74 | 9.02/7.14/7.97 | -2.77 pp |
| finite | 13.11/9.14/10.77 | 9.02/7.19/8.00 | -2.77 pp |

## Interpretation

This is a conditional fault-distribution experiment. Its metrics do not replace the main random-bit Full/Finite results.
A positive delta demonstrates benefit only on the filtered distribution; it does not recover the removed catastrophic faults.
