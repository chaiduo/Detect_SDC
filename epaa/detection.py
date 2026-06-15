# drdna/detection.py

import numpy as np

from .detection_metrics import (
    compute_tau1_layer,
    compute_tau2_layer,
    compute_tau3_layer,
    compute_tau_l,
)

consecutive_exceed_count = 0
total_exceed_count = 0

def compute_T_by_step(
    model_profile, 
    step, 
    activations_by_step
):
    """ 累加当前样本固定step下的T """
    T_step = 0.0    
    T_ref = 0.0
    for i in range(28):
        layer_key = f"L{i}.self_attn.o_proj"
        key = (step, layer_key)
        prof = model_profile.get(key)
        if prof is None:
            return 1, 0
        layer_act = activations_by_step[layer_key].detach()
        layer_act = layer_act.mean(dim=0).to("cpu").numpy().astype(np.float32)
        tau1 = compute_tau1_layer(prof, layer_act)
        tau2 = compute_tau2_layer(prof, layer_act)
        tau3 = compute_tau3_layer(prof, layer_act)
        tau_l = compute_tau_l(tau1, tau2, tau3)

        T_step += float(tau_l)
        T_ref += prof['tau_ref']

    return T_step, T_ref


def detect_significant_error(
    T_sample: float,     #当前样本，累积到step下的T
    T_sample_ref: float, #当前样本，累积到step下的T_ref
    margin_threshold=0.1,
    exceed_k = 5,
):
    global consecutive_exceed_count 
    global total_exceed_count
    # 全局 margin（基于的累计）
    margin_global = (T_sample - T_sample_ref) / (T_sample_ref + 1e-6)
    if margin_global > margin_threshold:
        consecutive_exceed_count += 1
        total_exceed_count += 1
    else:
        consecutive_exceed_count = 0

    return consecutive_exceed_count >= exceed_k, margin_global
        
