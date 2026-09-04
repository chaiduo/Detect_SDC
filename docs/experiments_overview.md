# Detect_SDC 实验总览与复现索引

> **历史结果说明（ICLR v1）**：本文档中的数值来自旧的 `orig_id` 85/15 划分、部分 fault-run 保留和旧 Mapping 配置。它们仅作为历史基线，不得用于新的投稿主表。当前协议见 [`sieve_iclr_revision_plan.md`](sieve_iclr_revision_plan.md)，新产物写入 `artifacts/iclr_v2/`。


本文档统一梳理 Detect_SDC 项目中已经完成的科学实验、消融实验、在线部署实验
和对比实验。它是实验台账与结果索引，不替代各实验目录中的原始
`metrics_summary.json`、CSV 或详细分析文档。

完整的软件环境、外部依赖、随机种子、数据切分、逐项执行命令和 artifact
校验规则见 [`reproducibility.md`](reproducibility.md)。复现实验时应同时使用
本文档的结果口径和该执行规范。

## 1. 研究目标

项目研究多模态大模型推理过程中的 Silent Data Corruption（SDC），重点检测
会造成严重语义错误的 Significant SDC。当前生产目标定义为：

```text
significant_sdc_target =
    (pred_answer != clean_answer) and (significance == 2)
```

其中，`significance` 由 Prometheus LLM judge 给出的回答质量分数转换得到。
本文所有“主检测结果”和消融结果均针对 Significant SDC 二分类任务。

## 2. 统一实验设置

### 2.1 模型与数据集

| Model | Decoder layers | Hidden size | Mapping hidden dim |
|---|---:|---:|---:|
| Qwen2.5-VL-7B | 28 | 3584 | 64 |
| InternVL3-8B | 28 | 3584 | 64 |
| LLaVA-1.5-7B | 32 | 4096 | 256 |

每个模型均在以下三个数据集上执行实验：

- EarthVQA
- LingoQA
- VQAv2

完整实验矩阵为 `3 models x 3 datasets = 9 jobs`，定义于
`configs/experiments/current.yaml`。

### 2.2 故障与运行配置

- 故障类型：Decoder activation 上的 double-bit flip。
- 故障运行次数：每个 job 10 个 fault runs。
- 最大生成长度：50 tokens。
- 投影方式：64 维 Column-Orthogonal Projection，默认 `project`。
- Mapping：FP32 Layer-Aware Residual MLP。
- Detector：CPU XGBoost，`learning_rate=0.01`。
- 数据划分：按 `orig_id` 分组，禁止同一原始问题跨训练集和验证集。

### 2.3 默认监测与特征

默认监测 6 个相邻层对：

```text
(6,7), (22,23), (23,24), (24,25), (25,26), (26,27)
```

在线实现只对这些层对涉及的 8 个不同 Decoder 层注册 hook。每个层对计算：

- Cosine Similarity
- Mean Difference
- Standard-Deviation Difference
- L2 Distance

每项指标在 decoding-step 维度上计算 mean、max 和 min，因此默认表示为：

```text
6 layer pairs x 4 metrics x 3 statistics = 72 features
```

### 2.4 结果口径

主 detector 只训练一次，同时报告：

- `Full`：完整验证集合，包含 all-feature-NaN 样本；
- `Non-all-NaN`：至少有一个有限特征的固定验证子集。

消融实验必须使用由 Full 配置定义的固定 non-all-NaN cohort，不能让不同配置
各自删除样本。正式 detector 与方法对比均在 Calibration 上最大化
Significant-SDC F1，并冻结阈值用于 Final Test。

## 3. 端到端主实验

### 3.1 八阶段流水线

规范流水线包含：

```text
profile -> collect_mapping -> train_mapping -> inject ->
label -> featurize -> train_detector -> report
```

各阶段的标准文件为：

| Stage | Artifact |
|---|---|
| profile | `json/profile.json` |
| collect_mapping | `json/mapping.jsonl` |
| train_mapping | `model/*mapping_model.pt` |
| inject | `json/injection.jsonl` |
| label | `json/labels.jsonl` |
| featurize | `train_data/*train_set.csv`, `*valid_set.csv` |
| train_detector | `output/train_with_nan/metrics_summary.json` |

### 3.2 Significant-SDC 检测结果

下表报告 72 维 Full detector 的 Significant-SDC Precision、Recall 和 F1。

| Model | Dataset | Full P/R/F1 | Non-all-NaN P/R/F1 |
|---|---|---:|---:|
| Qwen2.5-VL-7B | EarthVQA | 90.48 / 90.87 / 90.67 | 87.71 / 88.20 / 87.96 |
| Qwen2.5-VL-7B | LingoQA | 98.08 / 88.31 / 92.94 | 97.24 / 83.93 / 90.10 |
| Qwen2.5-VL-7B | VQAv2 | 99.07 / 77.82 / 87.17 | 98.62 / 70.10 / 81.95 |
| InternVL3-8B | EarthVQA | 99.32 / 97.10 / 98.19 | 99.15 / 96.44 / 97.78 |
| InternVL3-8B | LingoQA | 99.17 / 96.00 / 97.56 | 98.98 / 95.09 / 96.99 |
| InternVL3-8B | VQAv2 | 98.88 / 93.04 / 95.87 | 98.61 / 91.49 / 94.92 |
| LLaVA-1.5-7B | EarthVQA | 97.48 / 87.57 / 92.26 | 91.49 / 66.15 / 76.79 |
| LLaVA-1.5-7B | LingoQA | 94.00 / 89.81 / 91.86 | 95.74 / 73.77 / 83.33 |
| LLaVA-1.5-7B | VQAv2 | 92.41 / 91.78 / 92.10 | 93.94 / 72.09 / 81.58 |

Non-all-NaN F1 的模型平均值分别为：

| Model | Mean F1 |
|---|---:|
| Qwen2.5-VL-7B | 86.67 |
| InternVL3-8B | 96.56 |
| LLaVA-1.5-7B | 80.57 |
| **Macro average** | **87.93** |

结果来源：

```text
<model>/<dataset>/output/train_with_nan/metrics_summary.json
```

## 4. Significant SDC 现象分析

### 4.1 Significant SDC 占比

该实验合并各 job 的训练集与验证集，统计 non-all-NaN 样本中
Significant SDC 占全部 SDC 的比例。

| Model | EarthVQA | LingoQA | VQAv2 |
|---|---:|---:|---:|
| Qwen2.5-VL-7B | 8.68% | 11.35% | 14.48% |
| InternVL3-8B | 18.61% | 24.11% | 24.40% |
| LLaVA-1.5-7B | 43.99% | 37.43% | 81.86% |

用途：说明并非所有 SDC 都具有同等语义后果，保护目标需要从“所有 SDC”收缩到
真正影响回答质量的 Significant SDC。

脚本与图片：

```text
scripts/plot_significant_sdc_share.py
figures/significant_sdc_share.png
figures/significant_sdc_share.pdf
```

### 4.2 数值偏差与语义后果四象限

该实验按故障值的绝对偏差是否超过阈值，以及样本是否为 Significant SDC，
构造四象限统计。主阈值为 1，并对 Qwen-LingoQA 补充阈值 5 和 10。

主要结论：

- 数值偏差与语义后果相关，但不等价；
- 小数值偏差仍可能造成 Significant SDC；
- 大数值偏差也可能不产生显著语义影响；
- 只使用范围阈值会漏检有限值语义错误，并可能对无害偏差误报警。

脚本与图片：

```text
scripts/plot_fault_quadrant_comparison.py
scripts/plot_qwen_lingoqa_fault_quadrants.py
figures/fault_quadrant_comparison_non_nan.{png,pdf}
figures/*_fault_quadrants_threshold1.{png,pdf}
figures/qwen_lingoqa_fault_quadrants_threshold{5,10}.{png,pdf}
```

### 4.3 跨层差异分布

该实验将记录划分为 Non-SDC、SDC 和 Significant SDC，先对单个样本在多个
decoding step 上求均值，再对类别聚合，分析不同层对的 CosSim、MeanDiff、
StdDiff 和 L2Distance。

主要结论：

- Significant SDC 在后部层对上的偏移通常更明显；
- Non-SDC 与普通 SDC 的差异较小，说明仅检测任意 SDC 并不等同于识别严重错误；
- 不同层对的有限样本数可能不同，因为采用逐值 finite-only 聚合。

脚本与结果：

```text
scripts/plot_sdc_cosine_by_layer_pair.py
scripts/plot_sdc_metrics_by_layer_pair.py
scripts/plot_lingoqa_cosine_across_models.py
figures/*cosine*three_models.{csv,png,pdf}
figures/qwen_lingoqa_sdc_metrics_by_layer_pair.{csv,png,pdf}
```

## 5. 投影与 Fault-Free Feature Predictor

### 5.1 正交投影保持性

在 Qwen2.5-VL-7B 和 LingoQA 上运行 128 个无故障样本，比较 3584 维原始
activation 与 64 维正交投影后各层关系矩阵。

| Metric | Value |
|---|---:|
| Mean-RSM Pearson correlation | 0.9090 |
| Mean-RSM Spearman correlation | 0.9024 |
| Projection orthogonality relative error | 2.51e-8 |

结论：64 维投影较好保留跨层关系，可用于降低 Predictor 和差异计算成本。

脚本与结果：

```text
scripts/analyze_projection_preservation.py
analysis/qwen_lingoqa_projection_preservation.json
figures/qwen_lingoqa_projection_preservation.{png,pdf}
```

### 5.2 Fault-Free Feature Predictor 预测质量

最终验证集上的 MSE 和 Cosine Similarity 如下：

| Model | EarthVQA MSE/CosSim | LingoQA MSE/CosSim | VQAv2 MSE/CosSim |
|---|---:|---:|---:|
| Qwen2.5-VL-7B | 0.1128 / 0.8194 | 0.1637 / 0.7436 | 0.2139 / 0.6414 |
| InternVL3-8B | 0.0919 / 0.7593 | 0.1191 / 0.6802 | 0.1292 / 0.5899 |
| LLaVA-1.5-7B | 0.0035 / 0.9470 | 0.0067 / 0.8729 | 0.0144 / 0.7636 |

这些指标只能在同一模型内部解释，MSE 受激活尺度影响，不能直接跨模型排序。
Predictor 的目标不是精确重建，而是提供稳定的 fault-free reference。InternVL
的 Predictor CosSim 较低但 detector F1 最高，也说明残差可分性比绝对重建
误差更重要。

## 6. 监测层对消融

### 6.1 Leave-One-Pair-Out

每次从默认 6 个层对中移除一个层对，并在 Full 配置定义的固定 non-all-NaN
cohort 上评估。

| Model | Full | w/o 6-7 | w/o 22-23 | w/o 23-24 | w/o 24-25 | w/o 25-26 | w/o 26-27 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-VL-7B | 86.67 | 87.79 | 86.61 | 87.28 | 86.33 | 86.99 | 86.76 |
| InternVL3-8B | 96.56 | 96.16 | 96.38 | 96.47 | 95.87 | 96.55 | 95.92 |
| LLaVA-1.5-7B | 80.57 | 80.60 | 79.64 | 79.46 | 78.65 | 80.10 | 79.08 |

主要结论：

- `(24,25)` 是跨模型最稳定的重要层对；
- `(26,27)` 对 InternVL 和 LLaVA 也有明显贡献；
- `(6,7)` 对 Qwen 较冗余，但其贡献具有模型依赖性；
- 单层对贡献不能脱离其他层对独立解释。

脚本与结果：

```text
scripts/run_layer_pair_design_ablation.py
<model>/pair_ablation_leave_one_out_20260813/<dataset>/summary.csv
```

### 6.2 监测布局与预算

比较 Current、Spread-6、Late-dense、Even-adjacent、Mid-late-dense 和
All-adjacent。不同模型根据其 28/32 层深度自适应生成层对。

| Model | Current | Spread-6 | Late-dense | Even-adjacent | Mid-late-dense | All-adjacent |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-VL-7B | 86.67 | 88.07 | 82.05 | **89.01** | 86.39 | 87.44 |
| InternVL3-8B | 96.56 | 96.70 | 94.44 | **97.30** | 95.16 | 97.11 |
| LLaVA-1.5-7B | 80.57 | 80.61 | 61.85 | 81.32 | 70.18 | **81.78** |

主要结论：

- 均匀覆盖网络深度通常优于只密集监测后层；
- 增加层对数量不会带来单调收益；
- All-adjacent 相比 Current 增加超过 4 倍特征，但平均收益有限；
- Current 的 72 维配置是检测性能与在线成本之间的保守折中。

脚本与结果：

```text
scripts/run_layer_pair_design_ablation.py
<model>/pair_ablation_{project,design}_20260813/<dataset>/summary.csv
```

## 7. 在线检测 step 消融

使用生成轨迹前 `K` 个 decoding step，而不是离线后缀窗口。每个 K 独立训练
XGBoost，复用主实验 train/valid UID，并固定 K=50 定义的 non-all-NaN cohort。

| K | Mean F1 | Delta vs. K=50 | Mean observed steps |
|---:|---:|---:|---:|
| 1 | 86.15 | -1.78 pp | 1.00 |
| 2 | 87.25 | -0.68 pp | 1.88 |
| 4 | 86.45 | -1.48 pp | 3.47 |
| 8 | 87.13 | -0.80 pp | 6.48 |
| 12 | 87.17 | -0.76 pp | 9.25 |
| 16 | 88.09 | +0.16 pp | 11.64 |
| 24 | **88.38** | +0.45 pp | 14.74 |
| 32 | 87.81 | -0.13 pp | 15.93 |
| 50 | 87.93 | 0 | 17.05 |

部署点选择 `K=2`，原因是：

- 相比完整轨迹，平均 F1 仅下降 0.68 个百分点；
- 平均观测长度从 17.05 降至 1.88 step，减少约 89%；
- K=2 比 K=4、8、12 延迟更低，平均 F1 反而更高。

脚本与结果：

```text
scripts/run_online_step_ablation.py
scripts/plot_online_step_ablation.py
figures/online_step_ablation_aggregate.csv
figures/online_step_ablation_detailed.csv
figures/online_step_ablation.{png,pdf}
```

## 8. 差异特征消融

### 8.1 特征组 Leave-One-Out 与 Single-Group-Only

| Model | Full | w/o CosSim | w/o MeanDiff | w/o StdDiff | w/o L2Dist |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-VL-7B | 86.67 | 86.28 | 87.39 | 86.71 | 86.48 |
| InternVL3-8B | 96.56 | 96.16 | 95.72 | 96.60 | 96.50 |
| LLaVA-1.5-7B | 80.57 | 78.62 | 78.72 | 79.64 | 79.98 |
| **Average** | **87.93** | 87.02 | 87.28 | 87.65 | 87.65 |

单独使用一个 18 维特征组时，平均 F1 为：

| CosSim | MeanDiff | StdDiff | L2Dist |
|---:|---:|---:|---:|
| 86.21 | 86.26 | **87.00** | 86.83 |

Full 在九组平均上最佳。各指标存在冗余，但不同模型的最佳单组不一致，因此四类
指标共同使用具有更稳定的跨模型表现。

详细文档与数据：

```text
docs/feature_group_ablation_20260815.md
analysis/feature_ablation_20260815/
scripts/run_feature_group_ablation.py
```

### 8.2 四种指标与三种统计量交叉消融

每个配置只保留一种指标和一种统计量，共 6 维特征。

| Metric | Mean | Max | Min |
|---|---:|---:|---:|
| CosSim | 86.28 | 85.12 | 83.63 |
| MeanDiff | 82.86 | 84.82 | 84.06 |
| StdDiff | 86.12 | 84.09 | 85.12 |
| L2Distance | **87.34** | 86.22 | 83.88 |

L2Distance-Mean 比 72 维 Full 仅低 0.59 个百分点，但在 LLaVA-VQAv2 上下降
4.08 个百分点。实验不支持为三个模型统一替换为同一个 6 维配置。

详细文档与数据：

```text
docs/compact_feature_ablation_20260815.md
analysis/compact_feature_ablation_20260815/
scripts/run_compact_feature_ablation.py
```

### 8.3 6 维 Detector 深度和 XGBoost 微基准

对在线 K=2 CosSim-Mean 特征将 XGBoost `max_depth` 从 1 扫描到 6：

- Depth 6：平均 F1 84.83%，平均 8,822 个节点；
- Depth 2：平均 F1 84.56%，平均 407 个节点；
- 72 维 Full：平均 F1 87.25%。

Depth 2 将 XGBoost 模型显著缩小，但固定单线程 batch-1 推断仅从平均
132.28 us 降至 114.57 us，绝对节省约 17.7 us。在线主要成本仍来自 Hook、
投影和 Predictor。

脚本：

```text
```

## 9. 在线部署开销

### 9.1 正式 72 维部署实验

硬件为 NVIDIA H20，数据集为 LingoQA，batch size 为 1，
`max_new_tokens=50`。每模型固定 50 个样本，执行全样本预热，并使用正向和
反向模式顺序；每模型每模式共 200 个端到端观测。

| Model | Baseline | Full SIEVE | Overhead (95% CI) | Throughput change |
|---|---:|---:|---:|---:|
| Qwen2.5-VL-7B | 538.04 ms | 544.06 ms | +1.12% [0.81%, 1.42%] | -0.91% |
| InternVL3-8B | 671.75 ms | 676.66 ms | +0.73% [0.36%, 1.31%] | -0.72% |
| LLaVA-1.5-7B | 252.50 ms | 255.88 ms | +1.34% [1.11%, 1.58%] | -1.32% |
| **Average** | - | - | **+1.06%** | **-0.98%** |

Step-hook 三模型平均变化为 -0.02%，置信区间均包含 0，因此其开销处于测量
噪声内。完整组件分解、显存和 detection-ready latency 见：

```text
docs/online_deployment_overhead_20260814.md
analysis/online_overhead_20260814/combined/
scripts/benchmark_online_overhead.py
scripts/summarize_online_overhead.py
```

### 9.2 72 维与 6 维同轮比较

同一进程、相同模型、样本和 XGBoost 深度下，6 维相对 72 维的端到端延迟变化：

| Model | 6-D vs. 72-D |
|---|---:|
| Qwen2.5-VL-7B | -0.04% [-0.36%, 0.30%] |
| InternVL3-8B | -0.01% [-0.34%, 0.31%] |
| LLaVA-1.5-7B | -0.35% [-0.92%, 0.25%] |
| **Average** | **-0.13%** |

所有置信区间均包含 0，说明减少 Detector 输入维度没有带来统计显著的端到端
加速。考虑到 72 维检测性能更稳定，默认在线实现继续使用 72 维。

结果：

```text
analysis/online_feature_profile_comparison_20260815/
docs/compact_feature_ablation_20260815.md
```

## 10. Ranger-style 与 Dr.DNA-style 检测对比

由于 Ranger 和 Dr.DNA 没有可直接用于当前 PyTorch VLM 的完整实现，项目实现
的是机制等价基线，不声称复现原系统：

- Ranger-style：基于无故障 profiling 的层级 activation range；
- Dr.DNA-style：Individual DNA、Layer DNA 和 Extreme-neuron 分数；
- SIEVE：NaN/Inf 快速路径加语义差异 detector。

三种方法复用同一批故障记录、8 个监测层和前两个 decoding step。阈值分别在
独立 calibration split 上最大化 Significant-SDC F1，最终指标在不相交的
test split 上报告。

### 10.1 九组宏平均

| Method | Cohort | Sig-SDC Recall | Sig-SDC Precision | Sig-SDC F1 |
|---|---|---:|---:|---:|
| Ranger-style | Full | 81.40 | **92.90** | 86.57 |
| Dr.DNA-style | Full | 74.64 | 92.47 | 82.02 |
| **SIEVE** | Full | **91.60** | 92.54 | **91.88** |
| Ranger-style | Finite-only | 42.12 | 72.85 | 52.89 |
| Dr.DNA-style | Finite-only | 22.35 | 59.24 | 29.83 |
| **SIEVE** | Finite-only | **70.58** | **76.67** | **72.03** |

主要结论：

- Full cohort 上 SIEVE 的 Significant-SDC Recall 比 Ranger-style 和
  Dr.DNA-style 分别高 10.20 和 16.96 个百分点；
- 在排除 NaN/Inf 的 finite-only cohort 上，SIEVE 的优势进一步扩大；
- 结果支持核心论点：数值范围和统计分布信号难以覆盖有限值语义错误。

代码与结果：

```text
compare_experiment/README.md
compare_experiment/configs/detection_comparison.yaml
compare_experiment/results/summary/macro_average_metrics.csv
compare_experiment/results/summary/detailed_metrics.csv
```

## 11. 实验与论文结构映射

| 论文位置 | 推荐实验 |
|---|---|
| Motivation | Significant SDC 占比、数值偏差四象限 |
| Predictor evaluation | Mapping MSE/CosSim、投影保持性 |
| Main comparison | Ranger-style、Dr.DNA-style、SIEVE |
| Representation analysis | 跨层 CosSim/差异分布 |
| Main ablation | Leave-One-Pair-Out、特征组消融 |
| Online deployment | K-step 消融、72 维端到端开销 |
| Appendix | 监测布局、4x3 紧凑特征、XGBoost 深度、逐 job 对比表 |

## 12. 结果文件索引

| 内容 | 入口 |
|---|---|
| 主配置 | `configs/experiments/current.yaml` |
| 主 detector 定义 | `docs/xgboost_current_methods_and_results.md` |
| Significant SDC 叙事 | `docs/significant_sdc_story.md` |
| 在线开销 | `docs/online_deployment_overhead_20260814.md` |
| 特征组消融 | `docs/feature_group_ablation_20260815.md` |
| 紧凑特征消融 | `docs/compact_feature_ablation_20260815.md` |
| 方法对比 | `compare_experiment/README.md` |
| 论文图片 | `figures/` |
| 跨机器重画与紧凑数据 | `docs/figure_data.md` |
| 汇总数据 | `analysis/` |

## 13. 复现入口

以下命令仅用于快速检查。完整的从原始数据重跑、下游复现、消融、在线开销和
方法对比命令见 [`reproducibility.md`](reproducibility.md)。

验证完整配置：

```bash
PYTHONPATH=src python -m detect_sdc.cli config validate \
  configs/experiments/current.yaml
```

运行单个主实验：

```bash
PYTHONPATH=src python -m detect_sdc.cli run \
  --job qwen25_vl_earthvqa \
  --stage profile --dry-run
```

重新训练 detector：

```bash
PYTHONPATH=src python -m detect_sdc.cli train \
  --job qwen25_vl_earthvqa
```

汇总方法对比：

```bash
PYTHONPATH=src:. python -m compare_experiment.summarize_results
```

## 14. 使用结果时的注意事项

1. 主实验、detector 消融和方法对比均使用独立 calibration/test split，并在
   Calibration 上最大化 Significant-SDC F1。
2. 6 维与 72 维的跨轮开销不能直接归因于特征维度；应使用同轮配对结果。
3. Mapping MSE 受模型 activation 尺度影响，不能跨模型直接比较。
4. `pair_sweep/` 和旧 `max` 投影结果属于历史探索，不应替代当前 `project`
   配置下的正式结果。
5. 旧三分类 detector 和“训练时删除 all-feature-NaN”结果已废弃；生产口径
   以 `output/train_with_nan/metrics_summary.json` 和当前二分类定义为准。
6. 原始 JSONL、模型 checkpoint 和部分 detector 二进制未纳入 Git；仓库中的
   CSV、JSON summary、论文图片与脚本用于结果核验和复现入口。
7. 复现前必须核对 `reproducibility/reference_sha256.txt`、对应模型的完整
   environment freeze，以及归档 artifact 的 `sample_uid` 集合。
