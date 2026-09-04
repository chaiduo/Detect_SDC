# SIEVE ICLR 投稿修改与实验重跑计划

本文档用于统一 SIEVE 项目的实验协议、代码整改、正式重跑、补充实验和论文修改顺序。目标不是局部修补现有结果，而是在一次协议冻结后形成可审计、可复现、可用于 ICLR 投稿的最终实验基线。

> 执行原则：先修复会影响结论可信度的问题，再进行正式 GPU 重跑。P0 和 P1 未通过验收前，不更新论文主结果。正式重跑前只允许小规模 smoke test。

论文原稿：[Not All Errors Matter: Severity-Aware Soft Error Detection for Vision Language Models](https://kcnvpnf70w73.feishu.cn/wiki/NXXzwAY26iQ5sgkHViXciEltnFh)

## 1. 已确认的实验决策

| 项目 | 冻结决策 |
|---|---|
| 数据规模 | 每个数据集固定使用 5,000 个输入；三个模型共享同一份数据划分清单 |
| 故障阶段 | 保持当前 Prefill 阶段单次 activation 双 bit flip |
| 故障模块 | 保持当前 ModelAdapter 暴露模型中的 `nn.Linear` 输出；新增组件分类 |
| Mapping 架构 | 统一 `x_dim=64`、`layer_emb_dim=16`、`hidden_dim=64`、`num_blocks=8`、`dropout=0.1` |
| 模型层数 | Qwen/InternVL 使用 28，LLaVA 使用 32 |
| 检测窗口 | 正式在线配置固定为前 `K=2` 个 decoding step；`K=50` 作为完整轨迹参考 |
| 方法对比 | Ranger-style、Dr.DNA-style 和 SIEVE 使用同一批故障执行，不再单独 replay |

故障组件后续划分为：

- Vision-path Linear
- Language-decoder Linear
- Projector/merger Linear
- `lm_head`

## 2. 优先级总览

| 优先级 | 目标 | 退出条件 | 是否阻塞正式重跑 |
|---|---|---|---|
| P0 | 冻结科学协议 | 数据分组、故障模型、标签语义和评估角色全部确定 | 是 |
| P1 | 修复代码与数据契约 | 严格隔离测试和小规模端到端 smoke test 通过 | 是 |
| P2 | 正式重跑主实验 | 九个 job 的完整 artifact、指标和校验和齐全 | 是 |
| P3 | 补足 ICLR 证据链 | Judge、基线、部署指标、泛化和稳健性实验闭环 | 阻塞投稿 |
| P4 | 重写论文与图表 | 论文、代码和 artifact 无事实冲突，英文稿完整 | 阻塞投稿 |
| P5 | 复现与发布检查 | 第三方可以按文档复现表格和图片 | 阻塞终稿 |

## 3. P0：冻结科学协议

### 3.1 数据划分

所有划分必须在模型运行和故障注入之前完成，并由数据集级 manifest 固化。划分依据是语义实体，而不是故障执行行或当前 `orig_id`。

| 数据集 | 5,000 输入的构成 | `semantic_group_id` | 划分约束 |
|---|---:|---|---|
| EarthVQA | 2,500 张图，每图最多 2 个问题 | `image_filename` | 同一图像的全部问题不得跨集合 |
| LingoQA | 1,000 条记录，归属于 500 个 `question_id`，每组共 10 个展开输入 | `question_id` | 同一问题的全部帧和重复记录不得跨集合 |
| VQAv2 | 5,000 个问题，涉及约 883 张图 | `image_id` | 同一图像的全部问题不得跨集合 |

推荐外层划分：

| 集合 | 目标规模 | 唯一用途 |
|---|---:|---|
| Fit | 约 3,500 | 训练 Mapping Predictor 和 Significant-SDC Detector |
| Calibration | 约 750 | 为三种方法分别选择 Significant-SDC F1 最大的阈值 |
| Final test | 约 750 | 协议和阈值冻结后，仅用于最终评估 |

Mapping 和 XGBoost 需要 early stopping 时，只在 Fit 内按 `semantic_group_id` 划分：

```text
Fit 3500
├── 85% model train：约 2975
└── 15% dev：约 525
```

约束：

- 禁止根据故障标签或最终指标重新平衡 split。
- clean run、fault run、Mapping telemetry 和派生特征必须继承所属 `semantic_group_id` 的 split。
- 三个模型必须复用同一份 dataset manifest。
- 保存每个集合的 group ID、sample ID 和 SHA-256。

### 3.2 Mapping 训练协议

九个 job 分别训练独立 Predictor 权重，但统一架构：

```yaml
x_dim: 64
layer_emb_dim: 16
hidden_dim: 64
num_blocks: 8
dropout: 0.1
num_layers: 28  # Qwen/InternVL
# num_layers: 32  # LLaVA
```

建议统一训练参数：

```yaml
batch_size: 2048
lr: 0.0005
weight_decay: 0.0001
epochs: 500
cosine_weight: 1.0
early_stop_patience: 10
seed: 42
```

具体要求：

- Mapping 训练数据只能来自 Fit。
- Mapping 内部 train/dev 必须按 `semantic_group_id` 分组。
- 禁止按 `sample × step × layer-pair` telemetry 行随机拆分。
- checkpoint 保存架构、训练参数、split hash、最佳 epoch 和最终指标。
- Predictor 最终性能可以在冻结配置后报告于 Final test clean traces，但不得据此继续调参。

### 3.3 故障注入协议

保持当前 Prefill 和模块范围，但必须准确记录实际故障空间：

- 每个模型导出 eligible `nn.Linear` module manifest。
- 记录模块名称、组件类别、Decoder 层号、算子类型和输出形状。
- 记录模块、元素和 bit 位的具体采样规则。
- 每条故障记录保存 `component`、`layer`、`op_type`、`dtype`、`bit_category`、`idx`、`bit_positions`、`before`、`after`、`injection_step`、`run_index` 和 seed。
- 正式 calibration/test 的所有 fault runs 必须完整保留，不再只保留导致 SDC 的记录。

当前模型范围存在差异，论文必须披露：

- Qwen 和 LLaVA 可覆盖 Vision、Language、projector/merger 和 `lm_head`。
- InternVL Adapter 当前主要暴露 Language Model。
- 跨模型宏平均结果不能被表述为完全相同的组件级故障分布。

### 3.4 监督目标与评估协议

Significant SDC 表示：

```text
faulty response 相对同一输入下 fault-free response 的重大语义偏差
```

它不表示模型相对数据集 ground truth 的绝对错误。论文、表格和图片统一使用 `Fault-free response`，避免使用 `Correct answer`。

主方法比较统一使用：

- 前 `K=2` 个 decoding step；
- Fit 训练；
- Calibration 按 Significant-SDC F1 最大化选择阈值；
- Final test 报告最终结果；
- clean execution、injected Non-SDC、Slight SDC 和 Significant SDC 分开统计。

## 4. P1：代码与数据契约整改

| 工作包 | 修改内容 | 完成定义 |
|---|---|---|
| Split manifest | 新增 `semantic_group_id`，使用固定 seed 和稳定 hash 生成 Fit/Calibration/Test | group overlap=0，样本全集无丢失和重复 |
| Mapping 身份字段 | Mapping JSONL 增加 `orig_id` 和 `semantic_group_id` | Mapping train/dev group overlap=0，Final test 不参与训练 |
| Mapping 配置 | 所有 job 使用 hidden=64、blocks=8，移除历史架构和训练参数覆盖 | 除 `num_layers` 外架构一致 |
| 故障目标登记 | 导出 eligible Linear 模块并分类 | 每条故障可映射到唯一组件 |
| 完整故障保留 | 移除后续 run 的 SDC-only 策略 | 每个输入拥有完整 clean/fault 记录 |
| 单次多方法采集 | 同一 fault inference 保存 SIEVE features、Ranger score、Dr.DNA score 和 non-finite 状态 | 三种方法共享完全相同的 fault UID |
| 特征语义对齐 | 固定 Hook、MeanDiff、StdDiff、CosSim 和 Predictor 定义 | 公式、代码、列名和测试一致 |
| 统计置信区间 | bootstrap 改为按 `semantic_group_id` 整组重采样 | 主指标和方法差值均有 clustered 95% CI |
| 回归测试 | 增加 split、Mapping、故障范围、完整 run 和 artifact schema 测试 | 完整测试和 smoke test 通过 |

### 4.1 必须修正的方法描述

- 实际 Hook 是相邻 Decoder 层 `self_attn.o_proj` 的输出，不是完整 Decoder block 的输入和输出。
- `MeanDiff` 和 `StdDiff` 当前是有符号差，不是绝对差。
- 当前特征是 cosine similarity，不是 cosine distance。
- 每个 job 使用一个共享的 Layer-Aware Predictor，对六个层对批量推理。
- 建议将 Fault-Free Feature Predictor 改称 `Nominal Inter-Layer Predictor`，或严格解释它是使用 clean traces 训练的正常层间映射模型，而非真正的反事实 fault-free feature 生成器。

## 5. P2：正式主实验重跑

正式运行前先选择一个 job，用 50–100 个输入执行 smoke test。Smoke test 不进入论文。

正式 campaign 顺序：

1. 读取冻结的 dataset split manifest。
2. 在 Fit clean traces 上训练统一架构的 Mapping Predictor。
3. 在 Fit clean traces 上建立 Ranger 和 Dr.DNA profile。
4. 对 Fit、Calibration、Final test 执行完整故障注入。
5. 在同一次故障推理中保存三种方法所需信号。
6. 使用缓存优先的 Prometheus 标注。
7. 使用 Fit fault records 训练 SIEVE XGBoost。
8. 使用 Calibration 选择三种方法各自的最大 F1 阈值。
9. 在 Final test 生成主指标、分层指标和置信区间。

### 5.1 现有产物复用策略

| 产物 | 是否复用 | 处理方式 |
|---|---|---|
| 模型、数据集、环境 freeze | 是 | 继续使用并重新验证校验和 |
| 现有 `mapping.jsonl` | 条件复用 | 建立 `sample_id → semantic_group_id` 的可验证映射后，只提取 Fit 行 |
| 现有 Mapping checkpoint | 否 | 架构和训练集合均改变，必须重训 |
| 现有 `profile.json` 答案 | 不作为最终基线 | 正式 campaign 内重新生成，避免 Qwen 漂移影响 clean FPR |
| 现有 injection telemetry | 否 | 来自旧 Predictor，且缺少后续 run 的 Non-SDC |
| 现有 Prometheus 标签 | 作为缓存 | 仅在 question、fault-free response 和 faulty response 完全一致时复用 |
| 现有 Detector、表格和图片 | 否 | 保留为历史基线，新实验完成后重新生成 |

### 5.2 新 artifact 的最低字段

每个正式 job 必须保存：

- 原始输入数和 semantic group 数；
- Fit/Calibration/Test 的样本数、group 数和 SHA-256；
- clean、Non-SDC、Slight SDC、Significant SDC 数量；
- fault component、层、算子和 bit 类型分布；
- Judge parse failure；
- finite、non-finite、all-feature-NaN 分布；
- Mapping、Detector checkpoint 及完整配置；
- 每个阶段输入输出文件的 SHA-256。

## 6. P3：ICLR 必需证据

### 6.1 Judge 有效性

- 从九个 job 分层抽取 300–500 个样本。
- 覆盖三个 Judge 分数、finite/non-finite、模型和数据集。
- 至少两名人工标注者独立盲评并仲裁分歧。
- 报告人工一致率、Cohen's kappa 或 Krippendorff's alpha。
- 报告 Prometheus 相对人工标签的混淆矩阵。
- 增加第二 Judge 或 rubric 扰动实验。

### 6.2 部署指标

- clean execution FPR；
- injected Non-SDC FPR；
- Slight-SDC trigger rate；
- Significant-SDC Recall、Precision 和 F1；
- ROC、PR；
- 0.1%、0.5%、1%、2%、5% FPR operating points；
- 不同故障先验下的 expected precision 或每百万次推理报警量；
- full、finite-only 和 non-finite-only；
- NaN/Inf 快速路径对 Full 指标的贡献；
- 按 `semantic_group_id` 计算的 clustered bootstrap 95% CI。

### 6.3 公平基线

除 Ranger-style 和 Dr.DNA-style 外，至少加入：

- NaN/Inf-only；
- residual L2 threshold；
- identity predictor，即直接比较相邻实际层；
- raw activation statistics + XGBoost；
- clean-only one-class detector；
- 可行的 logit/entropy baseline。

所有方法必须共享同一 Fit、Calibration、Final test、故障实例、监测窗口和
Calibration 最大 F1 阈值协议。

### 6.4 泛化与鲁棒性

- 必做：同一模型 leave-one-dataset-out。
- 分别报告 Predictor 固定、仅迁移 Detector，以及 Predictor 与 Detector 同时迁移。
- 加分项：仅使用 Fit 统计量标准化 discrepancy features 后进行 leave-one-model-out。
- 按 vision/language/projector/lm_head、注入层、算子类型、bit 类别和传播距离分组报告 Recall/FPR。
- 若自然采样下组件样本不足，再追加定向且等量的 component-wise fault injection。

### 6.5 投影与架构消融

- 比较 `projection_dim=32/64/128/256`。
- 报告 pairwise-distance distortion 的 mean、median、p95 和 max。
- 比较 Mapping `num_blocks=4/8`，验证主配置选择 8 的必要性。
- 同时报告 Predictor 误差、Detector 指标、延迟和显存。

## 7. P4：论文修改

| 位置 | 修改要求 |
|---|---|
| Abstract/Introduction | 将“完整故障样本”改为包含 non-finite 的 full test cohort；Significant SDC 占比准确写为 8.7%–81.9% |
| Threat Model | 准确描述 Prefill、Linear output activation、采样方式和模型实际模块范围 |
| Method | 按代码描述 `o_proj` 相邻层映射、有符号差、CosSim、统一 Mapping 架构和共享 Predictor |
| Dataset/Split | 给出 5,000 输入、semantic group、Fit/Calibration/Test 数量和零重叠检查 |
| Table 1 | 统一 Predictor train/dev/test 命名，只报告严格未参与训练的数据 |
| Table 2/4 | 表题明确 K、阈值策略、cohort、split 和 non-finite 策略，解释数字差异 |
| Table 6 | 补齐 K=50，区分观测窗口和真实计算开销 |
| Figure 1 | 去除手写字体和高饱和配色；改为 Fault-free response；定义两种 deviation |
| Figure 4 | 改为白底顶会风格；修正英文；准确展示一个共享 Predictor |
| Figure 6 | 裁掉大面积空白并统一排版 |
| Figure 7 | 改为互斥类别：Non-SDC、Non-significant SDC、Significant SDC |
| Related Work | 准确区分范围检测、分布检测、ABFT 和语义评估 |
| Conclusion/Limitations | 明确 workload-calibrated 或跨数据集泛化能力，以及软件故障注入和未集成恢复机制的边界 |
| 全文 | 实验冻结后再英文重写，补齐 References、artifact statement、统计方法和硬件环境 |

## 8. P5：复现与发布

- 生成 raw datasets、model metadata、split manifests、Mapping checkpoints、fault records、labels、features 和 summaries 的完整 SHA-256 manifest。
- 将绝对路径改为环境变量或本地 override。
- 提供 Analysis、Detector 和 Full 三个复现层级。
- 保存每个正式 job 的配置快照、Git commit、Python environment、GPU、driver、CUDA 和随机种子。
- 确保 portable figure 数据能在无模型、无 GPU 环境中重画全部论文图片。

## 9. 阶段门禁与关键路径

1. **Gate 0：协议冻结。** Split、Mapping、故障分布、K、标签和阈值协议全部确认，之后不允许隐式修改口径。
2. **Gate 1：代码验收。** 单元测试、配置校验、smoke test 和 artifact schema 检查全部通过。
3. **Gate 2：主结果验收。** 九个 job 完成；Final test 未用于调参；clean、finite 和 non-finite 指标完整。
4. **Gate 3：论文证据验收。** Judge 人工验证完成；SIEVE 相对最强简单基线的优势在 finite-only 和跨数据集实验中仍成立，并报告置信区间。
5. **Gate 4：投稿验收。** 论文、代码、配置和结果文件一致，全部表图可以从冻结 artifact 自动生成。

关键路径：

```text
P0 协议冻结
→ P1 split/Mapping/单次多方法采集改造
→ smoke test
→ P2 九组正式重跑
→ P3 Judge、基线和泛化实验
→ P4 英文论文与图表
→ P5 artifact 检查
```

在线开销测试必须在最终实现冻结后执行，避免重复测量失效版本。

## 10. 风险与切换条件

| 风险 | 预警信号 | 响应动作 |
|---|---|---|
| 严格划分后指标下降 | Final test 明显低于旧结果，跨数据集迁移失败 | 停止使用“通用表征”，改为 workload-calibrated detector，并分析旧划分偏差 |
| 统一 Mapping 容量不足 | LLaVA Predictor 和 Detector 显著退化 | 保留统一 64×8 主配置，通过预注册容量消融判断是否允许模型相关 hidden dimension |
| Judge 一致性不足 | 人工一致率低，错误集中于 score 0/1 边界 | 改进 rubric，增加仲裁和第二 Judge，冻结新标签后重训 |
| Qwen 非确定性 | 相同 fault metadata 的输出仍漂移 | 正式推理当场保存并标注，不再依赖历史答案匹配 |
| 完整保留导致存储过大 | JSONL 写入和归档成为瓶颈 | 主实验只保存 K=2、六层对必要统计；完整轨迹仅用于固定小型消融 cohort |
| 方法新颖性不足 | 简单监督基线接近或超过 SIEVE | 强化跨分布、早期检测和系统开销贡献；必要时调整投稿定位 |

## 11. 最终完成标准

任何主结果都必须能够回溯到：

- 冻结的数据 split；
- 明确的故障分布；
- 独立的 Calibration 和 Final test；
- 可验证的 Judge 标签；
- 统一的代码路径；
- 完整的 artifact 校验和。

任何论文中的方法描述和数值，都必须能在代码、配置或冻结结果文件中找到唯一对应来源。
