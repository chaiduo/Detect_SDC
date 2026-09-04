# 从 SDC 检测到显著 SDC 保护：方法动机、技术路线与论文叙事

> **废弃叙事草稿**：本文档包含未进入当前代码的 SAFER 双头方案，不得作为 SIEVE 方法描述或投稿依据。当前协议见 `docs/sieve_iclr_revision_plan.md`。


## 1. 故事主线

这项工作的核心逻辑可以概括为三步：

1. 我们首先提出了一个用于检测 `SDC` 的方案。该方案能够较好地区分 `SDC` 与 `non-SDC`，并且已经取得了不错的检测效果。
2. 但是，系统级保护并不一定需要对所有 `SDC` 一视同仁。若希望进一步降低保护开销，更合理的目标是只对真正会造成明显语义错误或任务失败的 `显著 SDC` 进行保护。
3. 因此，我们引入了一个新的目标：不仅要检测 `SDC`，还要进一步识别其中的 `显著 SDC`，并在此基础上提高 `显著 SDC` 的召回率，同时尽量降低 `non-SDC` 的误报率。

换句话说，这项工作的重点不是把原始 `SDC` 检测器推翻重做，而是在已有检测能力之上，把问题从“是否发生 SDC”进一步推进到“是否发生值得保护的显著 SDC”。


## 2. 为什么只做 SDC 二分类还不够

最直接的 `SDC` 检测任务是一个标准二分类问题：

- `label = 1`: 样本发生 `SDC`
- `label = 0`: 样本为 `non-SDC`

对应的模型学习目标可以写成：

`p_sdc(x) = P(label = 1 | x)`

这类二分类器能够回答“这个样本像不像 SDC”，但它无法区分不同 `SDC` 的后果是否严重。

这会带来一个实际问题：如果系统对所有被判为 `SDC` 的样本都施加同等保护，那么保护开销会偏大。因为在真实系统里，并不是每一个 `SDC` 都会导致明显的语义错误、用户可感知错误或下游任务失效。真正值得优先保护的，是那些会造成显著后果的 `SDC`。

因此，问题自然演化为：

- 第一阶段：检测 `SDC`
- 第二阶段：在 `SDC` 中进一步识别 `显著 SDC`

这就是后续方法设计的出发点。


## 3. 如何定义“显著 SDC”

为了把“显著”这个概念从主观描述变成可训练标签，我们使用 `LLM-judge` 对样本进行语义层面的质量标注。具体实现位于 `detect_sdc.labeling`。

`quality_score` 的评分规则为：

- `quality_score = 2`: 回答正确，与参考答案一致或语义等价
- `quality_score = 1`: 回答有轻微偏差，但仍部分正确
- `quality_score = 0`: 回答错误，或存在严重语义错误

下游严重度定义为：

`significance = 2 - quality_score`

在本文的叙事里，我们把：

- `significance = 2`

定义为**显著错误样本**。进一步地，当一个样本同时满足：

- `label = 1`，即发生 `SDC`
- `significance = 2`，即语义后果严重

我们就称其为：

- `显著 SDC`

于是，研究目标从简单的 `SDC vs non-SDC`，转变为更贴近系统需求的目标：

> 在给定误报预算的条件下，尽量提高 `显著 SDC` 的召回率。


## 4. 新的研究问题：保护开销与保护收益之间的 tradeoff

有了 `significance` 标注之后，系统的优化目标就不再只是整体精度或整体 F1，而是一个更实际的 tradeoff：

- 一方面，希望尽可能召回更多 `显著 SDC`，避免漏掉真正危险的样本；
- 另一方面，希望尽可能降低 `non-SDC` 的误报率，否则会对大量正常样本触发不必要的保护，增加系统开销。

这本质上是一个受约束优化问题，而不是一个单纯追求总体分类准确率的问题。用符号表示就是：

`maximize Recall(label = 1, significance = 2)`

subject to

`FPR(label = 0) <= tau`

其中 `tau` 是可接受的 `non-SDC` 误报预算。

这也是为什么仅仅优化标准二分类损失、或仅仅根据 F1 选阈值，并不一定能得到我们真正想要的 operating point。


## 5. 我们的方法

我们的方法可以概括为：

**严重度感知的因子化风险建模 + FPR 约束阈值选择**

如果需要一个方法名，推荐使用：

- **SAFER**
- 全称：**Severity-Aware Factorized Error-Risk Detector**

这个名字对应三层含义：

- `Severity-Aware`：方法显式建模 `significance`
- `Factorized`：方法把风险拆成两个条件概率来建模
- `Error-Risk`：目标不是普通分类，而是面向显著错误风险的检测


## 6. 方法细节

### 6.1 基础 SDC 检测器

我们将基础 `SDC` 检测建模为一个**中间表征一致性判别问题**。其出发点是一个经验上很强的 baseline：发生 `SDC` 的样本与未发生 `SDC` 的样本，往往会在若干关键 `attention block` 的中间表征上表现出可分离差异。基于这一观察，我们不直接从最终答案预测 `SDC`，而是先刻画故障推理相对正常推理的中间层偏移，再据此进行二分类。

设样本为 `s`，解码 step 为 `t`，被监控的 `attention block` 为 `i`。记该 block 在 step `t` 的输入和输出分别为 `h_{i,t}^{in}` 与 `h_{i,t}^{out}`。我们首先在**无故障正常推理**数据上训练一个 `mapping model` `f_theta`，学习从 block 输入预测其正常输出：

`hat{h}_{i,t}^{out} = f_theta(h_{i,t}^{in}, i)`

实现上采用残差式参数化：

`hat{h}_{i,t}^{out} = h_{i,t}^{in} + g_theta(h_{i,t}^{in}, e_i)`

其中 `e_i` 是层编号嵌入。训练目标同时约束数值误差与方向一致性：

`L_map = ||hat{h}_{i,t}^{out} - h_{i,t}^{out}||_2^2 + lambda * (1 - cos(hat{h}_{i,t}^{out}, h_{i,t}^{out}))`

因此，`f_theta` 可以被视为 block `i` 在正常条件下输入输出映射的近似器。

在检测阶段，我们对模型注入故障，例如对某个激活值执行 `2-bit flip`，并运行故障推理。对于每个被监控的 block `i` 与 step `t`，我们同时获得：

- 故障推理下的真实输出 `tilde{h}_{i,t}^{out}`
- 由 mapping model 预测的参考输出 `hat{h}_{i,t}^{out}`

随后定义逐 step 的差异特征。当前使用四类度量：

`m_{i,t}^{cos} = cos(tilde{h}_{i,t}^{out}, hat{h}_{i,t}^{out})`

`m_{i,t}^{mean} = |mean(tilde{h}_{i,t}^{out}) - mean(hat{h}_{i,t}^{out})|`

`m_{i,t}^{std} = |std(tilde{h}_{i,t}^{out}) - std(hat{h}_{i,t}^{out})|`

`m_{i,t}^{l2} = ||tilde{h}_{i,t}^{out} - hat{h}_{i,t}^{out}||_2`

我们只在一个预选 block 集合 `B` 上提取这些统计量，例如 `B = {(6,7), (24,25), (25,26), (26,27), ...}`。对样本 `s`，设其解码过程中被观测到的 step 集合为 `T_s`。对任意度量 `m` 和任意 block `i in B`，我们在时间维度上做聚合，得到：

`phi_{i}^{m,mean}(s) = (1 / |T_s|) * sum_{t in T_s} m_{i,t}`

`phi_{i}^{m,min}(s) = min_{t in T_s} m_{i,t}`

`phi_{i}^{m,max}(s) = max_{t in T_s} m_{i,t}`

最终，样本 `s` 的检测特征向量定义为：

`x_s = concat({phi_{i}^{m,agg}(s) | i in B, m in M, agg in {mean, min, max}})`

其中 `M = {cos, mean, std, l2}`。在实现中，这些特征对应于：

- `cos_sim_mean_*`, `cos_sim_min_*`, `cos_sim_max_*`
- `mean_diff_mean_*`, `mean_diff_min_*`, `mean_diff_max_*`
- `std_diff_mean_*`, `std_diff_min_*`, `std_diff_max_*`
- `l2_distance_mean_*`, `l2_distance_min_*`, `l2_distance_max_*`

标签由故障是否改变最终输出决定。记故障推理输出为 `y_s^{fault}`，无故障输出为 `y_s^{clean}`，则：

`label_s = 1, if y_s^{fault} != y_s^{clean}`

`label_s = 0, otherwise`

因此，这一检测器关注的是“故障是否已经传播并造成行为级输出偏移”，而不是仅仅识别底层 fault 是否发生。

最后，我们使用 `XGBoost` 在特征 `x_s` 上训练二分类器，得到：

`p_sdc(x_s) = P(label_s = 1 | x_s)`

该设计的关键优点在于：它把高维、跨层、跨 step 的异常传播模式压缩成了结构化统计特征，同时保留了 `SDC` 与 `non-SDC` 在中间表征上可分的经验优势。本文后续提出的显著 `SDC` 检测方法，并不是替换这条基础链路，而是在这套 `mapping model + 中间层统计特征 + XGBoost` 的基础 `SDC` 检测器之上继续建模严重程度。


### 6.2 严重度感知的样本加权

仅靠普通二分类训练，模型会默认把所有正样本视为同样重要。为了让模型更关注 `显著 SDC`，我们引入严重度感知的样本权重。

训练时把样本分为三类：

- 普通负样本：`label = 0`
- 普通正样本：`label = 1`
- 高优先级正样本：`label = 1 and significance = 2`

其中，对 `label = 1 and significance = 2` 的样本给予更高权重，使模型在训练时优先优化这部分样本的召回。

同时，为了避免模型一味提高召回而把很多正常样本误报为风险样本，我们还对一些困难负样本进行加权，尤其是那些在严重度标签上具有迷惑性的 `non-SDC` 样本。这样做的目的是控制 `non-SDC` 的误报率。

从直觉上看，这一步是在告诉模型：

- 真正危险的样本不能漏掉；
- 看起来危险但实际上不是 `SDC` 的样本也不能乱报。


### 6.3 双头因子化建模

为了同时建模“是否是 SDC”和“如果是 SDC，它是否严重”，我们没有把问题简单改成一个新的单标签分类任务，而是采用双头建模。

主头学习：

`p_sdc(x) = P(label = 1 | x)`

辅助头只在正样本上学习：

`p_sig2(x) = P(significance = 2 | label = 1, x)`

然后定义最终的显著风险分数：

`severe_score(x) = p_sdc(x) * p_sig2(x)`

这个分数可以近似理解为：

`P(label = 1 and significance = 2 | x)`

这种写法的好处是语义清晰：

1. 先判断样本是否发生 `SDC`
2. 再判断如果发生 `SDC`，它是否属于高严重度

相比把所有标签强行揉成一个单任务，这种因子化建模更贴近问题结构，也更容易解释给读者和老师。


### 6.4 FPR 约束下的阈值选择

训练完模型后，最终决策并不按默认阈值 `0.5`，也不简单按整体 F1 最大来选。

我们在验证集上枚举阈值，并在满足：

`FPR(non-SDC) <= tau`

的所有候选阈值中，选择能够最大化：

`Recall(label = 1 and significance = 2)`

的 operating point。

因此，最终部署目标与训练目标是一致的：不是追求全局平均意义上的最优，而是在可接受误报预算下，优先保住显著 `SDC` 的召回。


## 7. 为什么这个方法合理

这个方法之所以合理，核心原因有三点。

### 7.1 它把“是否值得保护”显式建模了

原始二分类器只能回答“是否像 SDC”，但真正的系统决策问题是“这个样本值不值得付出保护开销”。引入 `significance` 后，模型终于开始对“保护收益”建模。


### 7.2 它保留了基础检测器的能力

我们没有放弃 `SDC` 检测，而是在已有检测能力上继续细分风险层级。因此，这种方法可以被视为对原系统的增强，而不是另一套完全割裂的管线。


### 7.3 它直接对应真实部署目标

实际部署里，最怕两件事：

- 漏掉真正严重的 `SDC`
- 对大量正常样本误触发保护

我们的方法用 `significance=2 recall` 和 `non-SDC FPR` 这两个指标直接刻画这两个目标，因此比只看 `accuracy` 或 `F1` 更符合系统需求。


## 8. 当前实验结果如何讲

目前最清楚的对比方式是：

- 基线：只用 `prob_sdc`，并在同样的 `FPR` 预算下选阈值
- 方法：使用 `severe_score = prob_sdc * P(significance=2 | SDC)`，并在同样的 `FPR` 预算下选阈值

这里比较的是**同样 FPR 预算下，显著 SDC 的召回能否更高；或者在相同召回下，FPR 能否更低**。

### 8.1 LingoQA

验证集结果如下：

| 方法 | target_sdc_recall | non_sdc_fpr |
| --- | ---: | ---: |
| `prob_sdc` under FPR budget | 0.9344 | 0.01599 |
| `severe_score` under FPR budget | 0.9344 | 0.01599 |

解读：

- `显著 SDC` 召回率保持不变：`0.9344 -> 0.9344`
- `non-SDC` 误报率保持不变：`0.01599 -> 0.01599`

这说明在清洗后的 `LingoQA` 数据上，当前 severe-score 方案没有带来额外收益；在给定的 `FPR` 预算下，它与 `prob_sdc` 基线收敛到了同一个 operating point。


### 8.2 EarthVQA

验证集结果如下：

| 方法 | target_sdc_recall | non_sdc_fpr |
| --- | ---: | ---: |
| `prob_sdc` under FPR budget | 0.8645 | 0.01151 |
| `severe_score` under FPR budget | 0.8710 | 0.00812 |

解读：

- `显著 SDC` 召回率提升：`0.8645 -> 0.8710`
- 绝对提升：`+0.0065`
- `non-SDC` 误报率下降：`0.01151 -> 0.00812`
- 相对下降约 `29.4%`

这说明在 `EarthVQA` 上，新方法同时做到了两件事：

- 提高 `significance=2` 的召回；
- 进一步降低 `non-SDC` 的误报率。

这正是本文故事里最想强调的结果。


### 8.3 VQAv2

验证集结果如下：

| 方法 | target_sdc_recall | non_sdc_fpr |
| --- | ---: | ---: |
| `prob_sdc` under FPR budget | 0.9490 | 0.00000 |
| `severe_score` under FPR budget | 0.9490 | 0.00135 |

解读：

- `显著 SDC` 召回率保持不变：`0.9490 -> 0.9490`
- `non-SDC` 误报率上升：`0.00000 -> 0.00135`
- 对应从 `0` 个误报增加到 `2` 个误报

这说明在 `VQAv2` 上，当前 severe-score 方案并没有带来额外收益；在保持 `significance=2` 召回不变的情况下，`prob_sdc` 基线反而取得了更低的误报率。这个结果同样重要，因为它说明该方法的收益具有数据集依赖性，不同任务上的最优 operating point 并不完全一致。


## 9. 论文里可以怎么讲

### 9.1 一段较完整的中文叙事

可以直接这样表述：

> 我们首先构建了一个有效的 SDC 检测器，用于区分 SDC 与 non-SDC 样本。然而，在实际系统中，对所有 SDC 一视同仁地施加保护会带来较高开销，而真正需要重点保护的是那些会造成明显语义错误的显著 SDC。为此，我们进一步引入了基于 LLM-judge 的严重度标注机制，将 significance=2 的样本定义为具有显著语义错误的高风险样本。在此基础上，我们将任务从普通 SDC 二分类扩展为一个受约束的风险检测问题：在给定 non-SDC 误报预算的前提下，最大化显著 SDC 的召回率。为实现这一目标，我们提出了一种严重度感知的因子化风险建模方法。该方法一方面在训练阶段对显著 SDC 与困难负样本施加差异化权重，另一方面在建模阶段将风险分解为 P(SDC|x) 与 P(significance=2|SDC,x) 两部分，并通过两者乘积构造显著风险分数。最终，我们在验证集上按照 FPR 约束选择阈值，从而直接优化系统真正关心的 operating point。实验表明，该方法在不同数据集上的收益存在明显差异：在 `EarthVQA` 上，它能够同时提升显著 SDC 召回并降低 non-SDC 误报率；在 `LingoQA` 上，它与基线收敛到相同的 operating point；在 `VQAv2` 上，则未带来额外收益。


### 9.2 英文方法摘要版本

> We first build an effective SDC detector to distinguish SDC from non-SDC samples. However, protecting every detected SDC incurs unnecessary overhead, while in practice the most critical cases are those that lead to severe semantic failures. To identify such high-impact cases, we annotate sample significance using an LLM-as-a-judge pipeline and define `significance = 2` as severe errors. Based on this annotation, we reformulate the problem from standard SDC detection to a constrained risk detection task: maximize the recall of severe SDCs under a fixed false-positive-rate budget on non-SDC samples. To this end, we propose SAFER, a severity-aware factorized error-risk detector. SAFER combines cost-sensitive training, a dual-head factorization of `P(SDC|x)` and `P(significance=2 | SDC, x)`, and FPR-constrained threshold selection. Experiments on cleaned datasets show clearly dataset-dependent behavior: SAFER improves both severe-SDC recall and non-SDC FPR on EarthVQA, collapses to the same operating point as the baseline on LingoQA, and provides no additional gain on VQAv2.

## 10. 可继续补充的内容

后续如果要把文档进一步扩成论文方法节，还可以继续补三部分：

1. 基础 `SDC` 检测器的特征定义与模型结构
2. `significance` 标注的一致性分析与误差案例
3. 更完整的跨数据集结果表与消融实验，例如不同 block 选择、不同权重系数，以及 `VQAv2` 上的误差案例分析
