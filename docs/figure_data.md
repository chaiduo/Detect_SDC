# Portable Figure Data

> **历史 portable data**：当前 `figures/` 数据对应 ICLR-v1 结果。完成 `artifacts/iclr_v2/` 正式重跑后必须重新导出 manifest，禁止在新论文中混用旧图。


论文图片不应依赖数十 GB 的 `labels.jsonl` 或完整 72 维训练 CSV。仓库采用两层
数据保留策略：

1. 原始 JSONL、checkpoint 和完整特征 CSV 用于重新分析与训练，存放在外部
   artifact archive。
2. Git 保存绘图所需的最小充分统计量，用于在普通 CPU 机器上快速重画论文图。

## Portable data

| Figure family | Portable source |
|---|---|
| Significant-SDC share | `figures/significant_sdc_share.csv` |
| Fault quadrants | `figures/fault_quadrant_counts.csv` |
| Cross-layer CosSim | `figures/*cosine*.csv` |
| Four discrepancy metrics | `figures/qwen_lingoqa_sdc_metrics_by_layer_pair.csv` |
| Projection preservation | `analysis/qwen_lingoqa_projection_preservation.json` |
| Online K-step ablation | `figures/online_step_ablation_{detailed,aggregate}.csv` |
| Feature ablations | `analysis/{feature,compact_feature}_ablation_20260815/` |
| Online overhead | `analysis/online_overhead_20260814/` and its reverse run |
| 72-D versus 6-D overhead | `analysis/online_feature_profile_comparison_20260815/` |
| Method comparison | `compare_experiment/results/summary/*.csv` |

`figures/figure_data_manifest.json` records every portable source file's path,
size, and SHA-256. It intentionally excludes PNG/PDF because those are outputs,
and excludes `.ubj`, `.pt`, and JSONL because they are not needed for plotting.

## Export

After changing raw experiment data, regenerate the compact counts and manifest:

```bash
Qwen2.5-VL-7B/.venv/bin/python \
  scripts/export_portable_figure_data.py
```

This command streams the large files and only writes aggregate CSV/JSON. It
does not modify raw experiment artifacts.

## Replot

On another machine, clone the repository and install Python plotting
dependencies. No model, dataset, GPU, Mapping checkpoint, Detector, JSONL, or
full feature CSV is required:

```bash
python scripts/replot_paper_figures.py
```

To verify without replacing the tracked figures:

```bash
python scripts/replot_paper_figures.py \
  --output-dir /tmp/detect_sdc_figures
```

The command validates `figure_data_manifest.json` before plotting. Use
`--skip-manifest-check` only while intentionally updating portable data.

## Scope

The compact package reproduces plotted aggregate values and figure layout. It
does not support changing the cohort definition, threshold sweep, monitored
layers, feature aggregation, or statistical protocol. Those operations require
the external raw artifact archive documented in
[`reproducibility.md`](reproducibility.md).
