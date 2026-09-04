# Deviation-bucket evaluation

## Protocol

- Train one detector per model/dataset job on the complete Fit split.
- Select one threshold by maximizing Significant-SDC F1 on the complete Calibration split.
- Freeze the detector and threshold before partitioning Final Test.
- Define deviation as `abs(after - before)`; NaN/Inf is a separate bucket.
- Full-injected contains every injected Final Test row.
- Finite-injected excludes only rows whose 72 SIEVE features are all NaN.
- Buckets are diagnostic strata unavailable to the deployed detector.

Included jobs: internvl3_lingoqa.
Skipped incomplete jobs: none.

## Aggregate Full-injected results

| Bucket | Samples | Significant | Sig. rate | Ranger F1 | Dr.DNA F1 | SIEVE F1 |
|---|---:|---:|---:|---:|---:|---:|
| zero | 0 | 0 | 0.00% | N/A | N/A | N/A |
| (0,1] | 5,096 | 31 | 0.61% | 0.00% | 0.00% | 0.00% |
| (1,1e6] | 1,346 | 6 | 0.45% | 4.76% * | 0.00% * | 0.00% * |
| (1e6,1e12] | 78 | 21 | 26.92% | 84.44% * | 77.78% * | 90.00% * |
| (1e12,1e18] | 25 | 7 | 28.00% | 93.33% * | 100.00% * | 100.00% * |
| (1e18,1e24] | 145 | 57 | 39.31% | 96.61% | 94.44% | 98.21% |
| (1e24,1e30] | 58 | 27 | 46.55% | 96.15% * | 82.61% * | 100.00% * |
| (1e30,1e36] | 113 | 41 | 36.28% | 88.61% | 73.85% | 100.00% |
| >1e36 | 531 | 214 | 40.30% | 93.33% | 80.45% | 98.58% |
| non_finite | 95 | 93 | 97.89% | 99.46% | 99.46% | 97.80% |

## Main findings

- `307/497` (61.77%) Significant-SDC rows have `abs(after-before) > 1e36` or a non-finite injected value.
- At `abs(after-before) <= 1e6`, SIEVE sees 37 positives among 6,442 rows and reaches Recall=0.00%, Precision=0.00%, F1=0.00%.
- In the extreme buckets, SIEVE reaches Recall=96.74%, Precision=100.00%, F1=98.34%.

## Interpretation rules

- `underpowered` means fewer than 30 Significant-SDC samples in the bucket.
- `no_positive` means Recall and F1 are not evidential for that bucket.
- Do not compare raw bucket precision without considering bucket prevalence.
- Do not train or select thresholds from Final Test bucket membership.

Detailed per-job Full/Finite metrics and cluster-bootstrap intervals are in `bucket_metrics.csv`.
