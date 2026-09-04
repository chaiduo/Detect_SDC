# Deviation-bucket evaluation

## Protocol

- Train one detector per model/dataset job on the complete Fit split.
- Select one threshold by maximizing Significant-SDC F1 on the complete Calibration split.
- Freeze the detector and threshold before partitioning Final Test.
- Define deviation as `abs(after - before)`; NaN/Inf is a separate bucket.
- Full-injected contains every injected Final Test row.
- Finite-injected excludes only rows whose 72 SIEVE features are all NaN.
- Buckets are diagnostic strata unavailable to the deployed detector.

Included jobs: qwen25_vl_earthvqa, qwen25_vl_lingoqa, qwen25_vl_vqav2, llava15_earthvqa, llava15_lingoqa, llava15_vqav2.
Skipped incomplete jobs: none.

## Aggregate Full-injected results

| Bucket | Samples | Significant | Sig. rate | Ranger F1 | Dr.DNA F1 | SIEVE F1 |
|---|---:|---:|---:|---:|---:|---:|
| zero | 0 | 0 | 0.00% | N/A | N/A | N/A |
| (0,1] | 33,993 | 58 | 0.17% | 0.00% | 0.00% | 0.00% |
| (1,1e6] | 7,355 | 54 | 0.73% | 12.12% | 8.33% | 16.00% |
| (1e6,1e12] | 238 | 11 | 4.62% | 84.21% * | 14.29% * | 85.71% * |
| (1e12,1e18] | 52 | 5 | 9.62% | 100.00% * | 75.00% * | 100.00% * |
| (1e18,1e24] | 397 | 36 | 9.07% | 77.42% | 58.18% | 89.86% |
| (1e24,1e30] | 216 | 21 | 9.72% | 81.08% * | 62.86% * | 97.56% * |
| (1e30,1e36] | 360 | 35 | 9.72% | 78.69% | 67.86% | 98.55% |
| >1e36 | 2,050 | 493 | 24.05% | 88.49% | 80.79% | 97.96% |
| non_finite | 545 | 531 | 97.43% | 98.76% | 98.95% | 99.91% |

## Main findings

- `1,024/1,244` (82.32%) Significant-SDC rows have `abs(after-before) > 1e36` or a non-finite injected value.
- At `abs(after-before) <= 1e6`, SIEVE sees 112 positives among 41,348 rows and reaches Recall=7.14%, Precision=5.19%, F1=6.02%.
- In the extreme buckets, SIEVE reaches Recall=98.54%, Precision=99.41%, F1=98.97%.

## Interpretation rules

- `underpowered` means fewer than 30 Significant-SDC samples in the bucket.
- `no_positive` means Recall and F1 are not evidential for that bucket.
- Do not compare raw bucket precision without considering bucket prevalence.
- Do not train or select thresholds from Final Test bucket membership.

Detailed per-job Full/Finite metrics and cluster-bootstrap intervals are in `bucket_metrics.csv`.
