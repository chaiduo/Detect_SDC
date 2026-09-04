# 在线部署开销实验

> **历史 ICLR-v1 结果**：本页数值使用旧 split 和 Detector 协议，仅供回归参考；v2 正式重跑后需整体替换。


## 实验协议

我们在 NVIDIA H20 GPU 上使用 LingoQA 测量在线部署开销。三个模型均采用
batch size 1 和 `max_new_tokens=50`。在线 SIEVE 仅对层
`{6, 7, 22, 23, 24, 25, 26, 27}` 注册 activation hook，并保留
`lm_head` step hook；检测器聚合前两个 decoding step 的 6 个层对，共
72 维特征。对于不足两个 step 即遇到 EOS 的输出，检测器使用已有的
`min(K,T)` 个 step 完成判断。

每个模型固定使用 50 个样本。正式计时前先执行 5 个常规 warmup，并对全部
50 个测量样本执行一次不计时的 Vanilla 生成，以消除视觉输入 shape 的首次
缓存开销。每种模式分别按正向和反向顺序执行 2 次：

```text
Vanilla -> Step-hook -> Monitor -> Predictor -> Full SIEVE
Full SIEVE -> Predictor -> Monitor -> Step-hook -> Vanilla
```

因此，每个模型的每种模式包含 200 次端到端观测。每次计时均使用
`torch.cuda.synchronize()` 包围完整生成过程。开销置信区间通过对同一运行
顺序、重复编号和样本编号下的 SIEVE/Vanilla 配对观测执行 10,000 次
bootstrap 得到。

## 端到端结果

**表 X：SIEVE 在 NVIDIA H20 上的在线部署开销。延迟单位为 ms，显存单位为
MiB。括号中为端到端开销的 95% bootstrap 置信区间。**

| Model | Vanilla | Full SIEVE | E2E overhead | Throughput change | Extra memory |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-VL-7B | 538.04 | 544.06 | +1.12% [0.81%, 1.42%] | -0.91% | 1.18 |
| InternVL3-8B | 671.75 | 676.66 | +0.73% [0.36%, 1.31%] | -0.72% | 1.18 |
| LLaVA-1.5-7B | 252.50 | 255.88 | +1.34% [1.11%, 1.58%] | -1.32% | 145.68 |
| Average | - | - | **+1.06%** | **-0.98%** | - |

显存增量使用正向顺序中的峰值显存计算，因为反向顺序会在运行 Vanilla 前
加载 predictor。LLaVA 的额外显存更高，主要来自其更大的 mapping MLP：
LLaVA 使用 `hidden_dim=256`，而 Qwen 和 InternVL 使用
`hidden_dim=64`。

## 组件分解

**表 Y：相对于 Vanilla 的端到端延迟变化。负值表示变化落在运行波动范围
内，不表示监测能够加速推理。**

| Model | Step-hook | Monitor + Projection | Predictor + Aggregation | Full SIEVE |
|---|---:|---:|---:|---:|
| Qwen2.5-VL-7B | +0.18% | +0.50% | +0.91% | +1.12% |
| InternVL3-8B | -0.17% | -0.47% | -0.06% | +0.73% |
| LLaVA-1.5-7B | -0.06% | +0.56% | +1.08% | +1.34% |
| Average | **-0.02%** | +0.20% | +0.64% | **+1.06%** |

三个模型的 Step-hook 95% 置信区间均包含 0，平均变化仅为 -0.02%，说明保留
`lm_head` step hook 不会引入可分辨的端到端开销。组件值不严格单调是因为
其量级接近运行噪声；最终部署成本应以 Full SIEVE 相对于 Vanilla 的配对
结果为准。

## 检测延迟

**表 Z：Full SIEVE 的检测就绪延迟。`Input-to-decision` 从输入处理开始计时，
`Post-prefill` 从 prefill 完成后计时。**

| Model | Input-to-decision mean / p95 | Post-prefill mean / p95 | Observed steps |
|---|---:|---:|---:|
| Qwen2.5-VL-7B | 352.86 / 356.12 | 148.85 / 151.39 | 2: 200 |
| InternVL3-8B | 366.94 / 375.85 | 44.33 / 46.27 | 2: 200 |
| LLaVA-1.5-7B | 208.87 / 221.69 | 93.86 / 103.47 | 1: 40; 2: 160 |

LLaVA 的 40 次单 step 观测对应提前产生 EOS 的短输出；这些输出在 EOS 时
立即使用已有 step 完成检测，因此检测覆盖率仍为 100%。

## 论文正文建议

**在线部署开销。** 我们进一步在 NVIDIA H20 GPU 上评估 SIEVE 的在线部署
成本。实验使用 LingoQA、batch size 1 和最大 50 个生成 token，并固定选取
50 个样本。在线实现仅监测 6 个层对涉及的 8 个 Transformer 层，保留
`lm_head` step hook，并在前两个 decoding step 后完成投影、无故障特征预测、
72 维差异特征聚合和 XGBoost 推断。为降低运行顺序和 GPU 热状态带来的偏差，
我们以正向和反向顺序分别重复测量，每个模型和模式共获得 200 次端到端观测。

如表 X 所示，完整 SIEVE 在 Qwen2.5-VL-7B、InternVL3-8B 和
LLaVA-1.5-7B 上的端到端延迟开销分别为 1.12%、0.73% 和 1.34%，三模型平均
仅为 1.06%；相应的吞吐下降平均为 0.98%。单独启用 Step-hook 时，三模型的
平均延迟变化为 -0.02%，且各模型的 95% 置信区间均包含 0，表明其成本处于
测量噪声范围内，因此无需为降低开销而移除 step 边界 hook。Qwen 和 InternVL
仅增加 1.18 MiB 峰值显存；LLaVA 增加 145.68 MiB，主要源于其采用了更大的
mapping MLP。

SIEVE 无需等待完整生成结束即可给出检测结果。如表 Z 所示，检测在输入开始后
平均 208.87--366.94 ms 内就绪；若排除 prefill，仅需额外
44.33--148.85 ms 即可在前两个 decoding step 内完成判断。对于在第二个
step 前产生 EOS 的短输出，SIEVE 使用已有 step 立即检测，从而在保持 100%
检测覆盖率的同时避免额外等待。上述结果说明，前两个 step 的在线部署方案能够
以约 1.1% 的端到端开销实现早期 Significant SDC 检测。
