# 当前 XGBoost 方法

本文档描述重构后的唯一生产训练口径。旧三分类结果以及训练时剔除
`all_feature_nan` 的实验表格已经失效；历史快照保存在
`baselines/pre_refactor_20260727/baseline.yaml`。

## 1. 任务定义

生产 detector 只训练一个二分类目标：

```text
0 = non_significant_sdc
1 = significant_sdc
```

正类定义：

```text
pred_answer != clean_answer and significance == 2
```

CSV 中的 canonical 列名是 `significant_sdc_target`。兼容逻辑支持：

- 三分类旧标签：`label == 2` 为正类。
- 二分类旧标签：`label == 1 and significance == 2` 为正类。
- 已存在的 target 必须与旧标签推导结果一致，否则报错。

Qwen2.5-VL 与 LLaVA 使用同一实现：
`src/detect_sdc/detector/xgboost.py`。模型级差异只保留学习率：

- Qwen2.5-VL: `0.02`
- LLaVA 1.5: `0.01`

## 2. 特征

默认使用 6 个 layer pair：

- `p6_7`
- `p22_23`
- `p23_24`
- `p24_25`
- `p25_26`
- `p26_27`

每个 pair 对 `cos_sim`、`mean_diff`、`std_diff`、`l2_distance` 做
`mean`、`min`、`max` 聚合，共 72 维。标签、身份列以及
`total_steps`、`last_k_steps`、`num_steps_used` 不进入模型。

非有限值和超出 float32 范围的数值转换为 NaN，由 XGBoost 原生处理。

## 3. 分组与泄漏控制

训练集内部的 train/test 切分只调用
`detect_sdc.splitting.split_by_group`：

- group column: `orig_id`
- holdout ratio: `0.15`
- random state: `42`
- train/test 的 `orig_id` 交集必须为空
- 重复 `sample_uid`、缺失 group、行丢失都会直接报错

外部 valid CSV 不参与拟合。

## 4. all_feature_nan 策略

所有特征均为 NaN 的样本保留在训练中。这是生产路径的固定策略，不再提供
“训练时删除 NaN”开关。

同一个拟合模型报告两个验证切片：

- `valid_full_metrics`: 完整 valid，包含 all-feature-NaN。
- `valid_non_all_nan_metrics`: 至少有一个有限特征的样本。

两个指标不能来自两次独立训练。

## 5. 训练参数

公共默认值位于 `configs/experiments/current.yaml`：

```yaml
n_estimators: 10000
max_depth: 6
min_child_weight: 1.0
subsample: 0.9
colsample_bytree: 0.9
early_stopping_rounds: 500
test_ratio: 0.15
random_state: 42
device: cpu
```

类别不平衡通过训练样本权重处理。

## 6. 输出

每个 job 输出：

- `metrics_summary.json`
- test、valid_full、valid_non_all_nan 的 prediction CSV
- 三个切片各自的 wrong-prediction CSV
- significant-SDC feature importance CSV

二分类混淆矩阵统一为：

```text
[[TN, FP],
 [FN, TP]]
```

主指标读取 `target_significant_sdc` 或 `target_class_1`。

## 7. 运行方式

```bash
PYTHONPATH=src python -m detect_sdc.cli train \
  --job llava15_earthvqa
```

六个旧 `train_xgboost.py` 只是这个入口的兼容包装器。

## 8. Layer-pair 实验

阈值扫描和 ROC 分析位于
`detect_sdc.detector.layer_pair_sweep`，不属于生产 detector。它默认：

- 只训练 significant-SDC 目标。
- 保留 all-feature-NaN 训练样本。
- 使用 `orig_id` 分组切分。

显式 `drop_all_feature_nan` 和 `both` 仅用于消融实验，结果不得替代生产口径。

## 9. 结果来源

最新指标以各数据集 `output/metrics_summary.json` 为准。冻结基线用于验证重构
没有修改输入 CSV；它不是当前二分类训练结果的替代品。
