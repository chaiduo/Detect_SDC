# drdna/detection_metrics.py

import numpy as np

# 论文里的 λ1, λ2, λ3
LAMBDA1 = 1.0
LAMBDA2 = 1.0
LAMBDA3 = 1.0

# τ1 中 +∞ 的实现替代值（论文说可以用“足够大”的数；这里用 1）
INF_SCORE = 1.0


def compute_tau1_layer(layer_profile, layer_act, INF_SCORE: float = INF_SCORE) -> float:
    """
    τ1: Individual DNA，通过 profiling sites 的直方图衡量。

    layer_profile 需要包含：
      - "indices": np.ndarray[int], shape [ns]
      - "site_hist_bins": np.ndarray[float], shape [ns, num_bins+1]
      - "site_hist_freqs": np.ndarray[float], shape [ns, num_bins] (已归一化)

    layer_act:
      - np.ndarray[float], shape [hidden] （当前样本该层/该 layer_key 的平均激活）
    """
    indices = layer_profile["indices"]             # [ns]
    site_bins = layer_profile["site_hist_bins"]   # [ns, num_bins+1]
    site_freqs = layer_profile["site_hist_freqs"] # [ns, num_bins]

    ns = int(len(indices))
    scores = np.zeros(ns, dtype=np.float32)

    for k, idx in enumerate(indices):
        y = layer_act[int(idx)]

        # 非有限值：直接判异常
        if not np.isfinite(y):
            scores[k] = float(INF_SCORE)
            continue

        bins = site_bins[k]
        freqs = site_freqs[k]

        # y 超出所有 bin 范围：按最异常计
        if y < bins[0] or y > bins[-1]:
            scores[k] = 1.0
            continue

        # 找到 bin j，使得 bins[j] <= y < bins[j+1]
        j = np.searchsorted(bins, y, side="right") - 1
        j = int(np.clip(j, 0, len(freqs) - 1))

        fj = freqs[j]
        scores[k] = 1.0 - float(fj)  # 频率越高越“正常”

    return float(scores.mean())


def compute_tau2_layer(layer_profile, layer_act) -> float:
    """
    τ2: Layer DNA，使用 EMD (Wasserstein-1) 比较 layer-level histogram。

    layer_profile 需要包含：
      - "indices": [ns]
      - "layer_hist_bins": [num_bins+1]   （bin edges）
      - "layer_hist_freqs": [num_bins]    （profiling 分布，已归一化）

    layer_act: [hidden]
    """
    indices = layer_profile["indices"]
    prof_bins = layer_profile["layer_hist_bins"]
    prof_freqs = layer_profile["layer_hist_freqs"]

    # 当前样本在这些 sites 上的激活
    values = layer_act[indices]

    # 去掉非有限值，避免 histogram 产生不可控结果
    values = values[np.isfinite(values)]

    # 注意：np.histogram 返回 (hist, bin_edges)
    det_hist, _ = np.histogram(values, bins=prof_bins)
    det_freqs = det_hist.astype(np.float32)

    total = float(det_freqs.sum())
    if total > 0:
        det_freqs /= total
    else:
        det_freqs = np.zeros_like(det_freqs, dtype=np.float32)

    # EMD（离散 1D 情况下，用 CDF 差的 L1）
    F1 = np.cumsum(prof_freqs, dtype=np.float32)
    F2 = np.cumsum(det_freqs, dtype=np.float32)
    emd = float(np.sum(np.abs(F1 - F2)))

    return emd


def compute_tau3_layer(layer_profile, layer_act) -> float:
    """
    τ3: Extreme Neurons（极值 neuron index 是否一致）：
      - layer_profile:
          - "extreme_min_idx"
          - "extreme_max_idx"
      - layer_act: [hidden]
      返回 0, 0.5, 1
    """
    prof_min = int(layer_profile["extreme_min_idx"])
    prof_max = int(layer_profile["extreme_max_idx"])

    # 非有限值会让 argmin/argmax 行为很怪：这里用“忽略非有限值”的版本
    act = np.asarray(layer_act, dtype=np.float32)
    finite = np.isfinite(act)

    if not finite.any():
        # 全是 nan/inf：直接认为最不匹配
        return 1.0

    # 对非有限位置做掩码：min 用 +inf，max 用 -inf
    act_for_min = act.copy()
    act_for_min[~finite] = np.inf
    cur_min = int(np.argmin(act_for_min))

    act_for_max = act.copy()
    act_for_max[~finite] = -np.inf
    cur_max = int(np.argmax(act_for_max))

    mismatch = 0
    if cur_min != prof_min:
        mismatch += 1
    if cur_max != prof_max:
        mismatch += 1

    if mismatch == 0:
        return 0.0
    elif mismatch == 1:
        return 0.5
    else:
        return 1.0


def compute_tau_l(tau1, tau2, tau3) -> float:
    """
    Eq.(1) 中的 τ_l = λ1 τ1 + λ2 τ2 + λ3 τ3
    """
    return float(LAMBDA1) * float(tau1) + float(LAMBDA2) * float(tau2) + float(LAMBDA3) * float(tau3)
