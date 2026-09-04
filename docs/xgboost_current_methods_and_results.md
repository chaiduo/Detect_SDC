# 当前 XGBoost Significant-SDC Detector 协议

本文档描述 ICLR-v2 唯一生产训练口径。旧三分类、默认 0.5 阈值、两文件
train/valid 和训练时删除 all-feature-NaN 的路径均已废弃。

## 1. 目标

```text
0 = non-significant execution
1 = pred_answer != clean_answer and significance == 2
```

`significance` 衡量 faulty response 相对 fault-free response 的增量语义影响。

## 2. 输入特征

默认监控六个层对：

```text
(6,7), (22,23), (23,24), (24,25), (25,26), (26,27)
```

前两个 decoding steps 上计算 CosSim、MeanDiff、StdDiff 和 L2Distance，分别
聚合 mean/max/min，共 72 维。非有限值转为 NaN，由 XGBoost 原生处理。

身份、故障 metadata、split、标签和 step 计数不进入模型。

## 3. 外层协议

- Fit：训练 XGBoost；
- Fit 内部按 `semantic_group_id` 留出 15% 用于 early stopping；
- Calibration：使用全部正负样本选择 Significant-SDC F1 最大的阈值；
- Final test：只报告结果。

三个集合的 semantic group 和 sample UID 必须完全不重叠。

## 4. 阈值与指标

默认目标为 Calibration 上 Significant-SDC F1 最大，预测规则为：

```text
positive_probability > calibrated_threshold
```

Final test 分别报告：

- Full；
- Finite-only；
- Clean execution；
- Injected Non-SDC；
- Slight SDC；
- Significant SDC。

all-feature-NaN 行保留在训练与 Full 测试中。

## 5. 训练参数

```yaml
n_estimators: 10000
learning_rate: 0.01
max_depth: 6
min_child_weight: 1.0
subsample: 0.9
colsample_bytree: 0.9
early_stopping_rounds: 500
test_ratio: 0.15
random_state: 42
device: cpu
```

类别不平衡通过 Fit model-train 子集上的 sample weights 处理。

## 6. 输出

```text
artifacts/iclr_v2/<job>/output/
├── metrics_summary.json
├── significant_sdc_detector.ubj
├── significant_sdc_feature_importance.csv
└── significant_sdc_binary_*_predictions.csv
```

`metrics_summary.json` 保存特征顺序、内部 split、训练参数、阈值校准、各 cohort
指标和模型路径。

## 7. 命令

```bash
PYTHONPATH=src python -m detect_sdc.cli train \
  --job qwen25_vl_earthvqa
```

消融实验必须调用同一个 calibrated detector backend，复用冻结的
Fit/Calibration/Final-test manifest。
