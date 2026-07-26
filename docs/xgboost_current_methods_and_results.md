# 当前 XGBoost 方法与效果整理

本文档整理 `/data01/cd_workspace/Detect_SDC/llava-v1.5-7B` 下三个数据集当前 `train_xgboost.py` 的统一三分类方法、泄漏控制、`all_feature_nan` 处理方式，以及外部 `valid_set.csv` 上的最新结果。

涉及数据集：

- `EarthVQA`
- `LingoQA`
- `VQAv2`

核心目标：

在给定 `non-SDC FPR <= 0.02` 的误报预算下，最大化显著 SDC 的召回率。显著 SDC 定义为 `label=1 && significance=2`。


## 1. 当前统一方法

三个数据集当前都已统一为三分类 severity classifier：

- `class 0`: `label=0`，非 SDC
- `class 1`: `label=1 && significance!=2`，SDC 但非显著
- `class 2`: `label=1 && significance=2`，显著 SDC

模型输出：

- `P(class=0)`
- `P(class=1)`
- `P(class=2)`

派生分数：

- `prob_any_sdc = P(class=1) + P(class=2)`
- `prob_significant_sdc = P(class=2)`

最终显著 SDC 保护决策使用：

```text
alert = prob_significant_sdc >= threshold
```

阈值选择策略：

```text
maximize target_sdc_recall
subject to non_sdc_fpr <= 0.02
```

其中 `target_sdc_recall` 只统计 `label=1 && significance=2` 的召回率；`non_sdc_fpr` 只统计 `label=0` 样本的误报率。


## 2. 特征与泄漏控制

每个样本的特征来自故障推理过程中多个 attention block pair 的异常统计。当前主要 block pair 包括：

- `p6_7`
- `p22_23`
- `p23_24`
- `p24_25`
- `p25_26`
- `p26_27`

每个 pair 上统计：

- `cos_sim`
- `mean_diff`
- `std_diff`
- `l2_distance`

并在跨 decoding step 维度上聚合：

- `mean`
- `min`
- `max`

当前特征列显式排除：

- 原始标签：`label`, `significance`
- 派生标签：`binary_label`, `ternary_label`
- 分组/样本标识：`group_id`, `group_uid`, `sample_id`, `sample_uid`, `orig_id`
- step 边界：`step_start`, `step_end`
- 泄漏风险元特征：`total_steps`, `last_k_steps`, `num_steps_used`

重要修正：

此前一版三分类脚本在构造 `binary_label/ternary_label` 后，没有把这两个派生标签从特征列中排除，导致结果出现直接标签泄漏。修正后当前三个数据集的特征数均为 `72`。


## 3. all_feature_nan 处理

训练阶段：

当前脚本设置 `DROP_ALL_FEATURE_NAN = True`，即训练时剔除所有核心监控特征全为 NaN 的样本。这样做是为了避免模型把上游空输出或特征提取失败模式直接当作捷径。

评估阶段：

当前报告两套 valid 结果：

- `valid_full`: 不去除 `all_feature_nan`，完整 valid 集评估。
- `valid_non_all_nan`: 去除 `all_feature_nan` 后评估。

剔除统计：

| 数据集 | train 剔除 | valid 剔除 | valid 剔除样本分布 |
|---|---:|---:|---|
| `EarthVQA` | `535` | `100` | `label=1, significance=0: 10`; `label=1, significance=2: 90` |
| `LingoQA` | `549` | `102` | `label=1, significance=0: 26`; `label=1, significance=2: 76` |
| `VQAv2` | `540` | `91` | `label=1, significance=0: 25`; `label=1, significance=2: 66` |


## 4. 指标与矩阵口径

脚本同时报告三类矩阵：

- `ternary_argmax_confusion_matrix`: 三分类 argmax 矩阵。行是真实 `class 0/1/2`，列是预测 `class 0/1/2`。
- `significant_sdc_decision_confusion_matrix`: 显著 SDC 保护决策矩阵。行是真实 `non-target/target`，列是 `no-alert/alert`。
- `label_level_protection_confusion_matrix`: 按原始 `label` 观察保护覆盖。行是真实 `label=0/1`，列是 `no-alert/alert`。

论文主指标应使用 `significant_sdc_decision_confusion_matrix`，因为目标是保护显著 SDC。`label_level_protection_confusion_matrix` 会把普通 SDC 未报警统计为 FN，只适合作为保护覆盖面的辅助观察。

二分类决策矩阵格式固定为：

```text
[[TN, FP],
 [FN, TP]]
```

其中正类是显著 SDC，即 `label=1 && significance=2`。


## 5. valid_full 结果：不去除 all_feature_nan

### 5.1 汇总表

| 数据集 | valid 样本数 | non-SDC | SDC | 显著 SDC | 阈值 | 显著 SDC recall | non-SDC FPR | target alert precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `EarthVQA` | `1699` | `1477` | `222` | `155` | `0.10` | `132/155 = 0.8516` | `17/1477 = 0.0115` | `132/171 = 0.7719` |
| `LingoQA` | `2419` | `2189` | `230` | `122` | `0.10` | `116/122 = 0.9508` | `16/2189 = 0.0073` | `116/183 = 0.6339` |
| `VQAv2` | `1627` | `1485` | `142` | `98` | `0.30` | `92/98 = 0.9388` | `8/1485 = 0.0054` | `92/137 = 0.6715` |

### 5.2 EarthVQA

```text
ternary_argmax_confusion_matrix:
[[1462   13    2]
 [  36   14   17]
 [  24    2  129]]

significant_sdc_decision_confusion_matrix:
[[1505   39]
 [  23  132]]
```

### 5.3 LingoQA

```text
ternary_argmax_confusion_matrix:
[[2167   20    2]
 [  31   65   12]
 [   8   78   36]]

significant_sdc_decision_confusion_matrix:
[[2230   67]
 [   6  116]]
```

### 5.4 VQAv2

```text
ternary_argmax_confusion_matrix:
[[1481    0    4]
 [   7    0   37]
 [   5    3   90]]

significant_sdc_decision_confusion_matrix:
[[1484   45]
 [   6   92]]
```


## 6. valid_non_all_nan 结果：去除 all_feature_nan

### 6.1 汇总表

| 数据集 | valid 样本数 | non-SDC | SDC | 显著 SDC | 阈值 | 显著 SDC recall | non-SDC FPR | target alert precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `EarthVQA` | `1599` | `1477` | `122` | `65` | `0.10` | `42/65 = 0.6462` | `17/1477 = 0.0115` | `42/71 = 0.5915` |
| `LingoQA` | `2317` | `2189` | `128` | `46` | `0.10` | `40/46 = 0.8696` | `16/2189 = 0.0073` | `40/81 = 0.4938` |
| `VQAv2` | `1536` | `1485` | `51` | `32` | `0.30` | `26/32 = 0.8125` | `8/1485 = 0.0054` | `26/46 = 0.5652` |

### 6.2 EarthVQA

```text
ternary_argmax_confusion_matrix:
[[1462   13    2]
 [  36   14    7]
 [  24    2   39]]

significant_sdc_decision_confusion_matrix:
[[1505   29]
 [  23   42]]
```

### 6.3 LingoQA

```text
ternary_argmax_confusion_matrix:
[[2167   20    2]
 [  31   39   12]
 [   8    2   36]]

significant_sdc_decision_confusion_matrix:
[[2230   41]
 [   6   40]]
```

### 6.4 VQAv2

```text
ternary_argmax_confusion_matrix:
[[1481    0    4]
 [   7    0   12]
 [   5    3   24]]

significant_sdc_decision_confusion_matrix:
[[1484   20]
 [   6   26]]
```


## 7. 本地混淆矩阵图片

### 7.1 EarthVQA

- `valid_full` 三分类 argmax: `/data01/cd_workspace/Detect_SDC/llava-v1.5-7B/EarthVQA/output/figure/valid_full_ternary_argmax_confusion_matrix.png`
- `valid_full` 显著 SDC 决策: `/data01/cd_workspace/Detect_SDC/llava-v1.5-7B/EarthVQA/output/figure/valid_full_significant_sdc_decision_confusion_matrix.png`
- `valid_non_all_nan` 三分类 argmax: `/data01/cd_workspace/Detect_SDC/llava-v1.5-7B/EarthVQA/output/figure/valid_non_all_nan_ternary_argmax_confusion_matrix.png`
- `valid_non_all_nan` 显著 SDC 决策: `/data01/cd_workspace/Detect_SDC/llava-v1.5-7B/EarthVQA/output/figure/valid_non_all_nan_significant_sdc_decision_confusion_matrix.png`

### 7.2 LingoQA

- `valid_full` 三分类 argmax: `/data01/cd_workspace/Detect_SDC/llava-v1.5-7B/LingoQA/output/figure/valid_full_ternary_argmax_confusion_matrix.png`
- `valid_full` 显著 SDC 决策: `/data01/cd_workspace/Detect_SDC/llava-v1.5-7B/LingoQA/output/figure/valid_full_significant_sdc_decision_confusion_matrix.png`
- `valid_non_all_nan` 三分类 argmax: `/data01/cd_workspace/Detect_SDC/llava-v1.5-7B/LingoQA/output/figure/valid_non_all_nan_ternary_argmax_confusion_matrix.png`
- `valid_non_all_nan` 显著 SDC 决策: `/data01/cd_workspace/Detect_SDC/llava-v1.5-7B/LingoQA/output/figure/valid_non_all_nan_significant_sdc_decision_confusion_matrix.png`

### 7.3 VQAv2

- `valid_full` 三分类 argmax: `/data01/cd_workspace/Detect_SDC/llava-v1.5-7B/VQAv2/output/figure/valid_full_ternary_argmax_confusion_matrix.png`
- `valid_full` 显著 SDC 决策: `/data01/cd_workspace/Detect_SDC/llava-v1.5-7B/VQAv2/output/figure/valid_full_significant_sdc_decision_confusion_matrix.png`
- `valid_non_all_nan` 三分类 argmax: `/data01/cd_workspace/Detect_SDC/llava-v1.5-7B/VQAv2/output/figure/valid_non_all_nan_ternary_argmax_confusion_matrix.png`
- `valid_non_all_nan` 显著 SDC 决策: `/data01/cd_workspace/Detect_SDC/llava-v1.5-7B/VQAv2/output/figure/valid_non_all_nan_significant_sdc_decision_confusion_matrix.png`


## 8. 当前结论

1. 不去除 `all_feature_nan` 时，三个数据集的显著 SDC 召回率都较高：`EarthVQA 0.8516`、`LingoQA 0.9508`、`VQAv2 0.9388`。
2. 去除 `all_feature_nan` 后，性能下降明显，说明全 NaN 样本确实是一类强模式，不能混入普通特征结论中解释。
3. 在 `non-SDC FPR <= 0.02` 约束下，去除全 NaN 后仍能召回一部分显著 SDC：`EarthVQA 0.6462`、`LingoQA 0.8696`、`VQAv2 0.8125`。
4. 三分类 argmax 矩阵用于观察模型本身是否把样本分到 `0/1/2`；显著 SDC 决策矩阵才是保护策略的主结果。
5. 后续论文实验需要继续做 `p6_7` 消融：去掉全部 `p6_7` 特征重跑，以及只保留 `p6_7` 特征重跑。
