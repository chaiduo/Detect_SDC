import json
import math
import csv
from collections import defaultdict

import pandas as pd
import numpy as np
# from nltk.translate.meteor_score import meteor_score
# import nltk
# nltk.download('punkt_tab')
# nltk.download('wordnet')
from tqdm import tqdm
# from sentence_transformers import CrossEncoder
from sklearn.model_selection import GroupShuffleSplit


# =========================
# 配置区域
# =========================
INPUT_JSON = "./json/detect_EarthVQA_llava_with_sem_project_labeled.jsonl"

SELECTED_LAYER_PAIRS = [
    (6,7),
    (22,23),
    (23,24),
    (24,25),
    (25,26),
    (26,27),
]

# 用于 l2_distance 的 layer pairs
DISTANCE_PAIRS = [
    (6,7),
    (22,23),
    (23,24),
    (24,25),
    (25,26),
    (26,27),
]

# 固定窗口配置
LAST_K_STEPS = 50  # 只统计最后 k 个 step

class SimilarityEvaluator:
    def __init__(self, model_name):
        self.model = CrossEncoder(model_name)

    def score(self, text1, text2):
        return float(self.model.predict([(str(text1), str(text2))])[0])


def compute_bleu_and_meteor(reference_sentence, candidate_sentence):
    reference_tokens = nltk.word_tokenize(reference_sentence.lower())
    candidate_tokens = nltk.word_tokenize(candidate_sentence.lower())
    meteor = meteor_score([reference_tokens], candidate_tokens)
    return meteor


# =========================
# 读取 JSON / JSONL
# =========================
def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        return []

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
    except json.JSONDecodeError:
        pass

    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


# =========================
# 按 step 聚合 records
# =========================
def build_step_pair_map(records):
    """
    构建:
    step_pair_map[step][(src_layer, tgt_layer)] = record
    """
    step_pair_map = defaultdict(dict)
    for r in records:
        step = r["step"]
        pair = (r["src_layer"], r["tgt_layer"])
        step_pair_map[step][pair] = r
    return step_pair_map


# =========================
# 安全统计函数
# =========================
def safe_mean(values):
    return float(np.mean(values)) if values else np.nan

def safe_min(values):
    return float(np.min(values)) if values else np.nan

def safe_max(values):
    return float(np.max(values)) if values else np.nan

def collect_values_for_pair(step_pair_map, step_group, pair, key):
    """
    收集指定 step 组内指定 pair 的值
    """
    values = []
    for step in step_group:
        rec = step_pair_map.get(step, {}).get(pair)
        if rec is not None and key in rec:
            values.append(rec[key])
    return values


def extract_features_for_step_group(step_pair_map, step_group, selected_pairs):
    """
    对一个 step 组提取特征，每个 layer pair 单独输出，不聚合
    """
    feat = {}

    for pair in selected_pairs:
        pair_key = f"p{pair[0]}_{pair[1]}"

        # 1) cos_sim
        cos_values = collect_values_for_pair(step_pair_map, step_group, pair, "cos_sim")
        feat[f"cos_sim_mean_{pair_key}"] = safe_mean(cos_values)
        feat[f"cos_sim_max_{pair_key}"] = safe_max(cos_values)
        feat[f"cos_sim_min_{pair_key}"] = safe_min(cos_values)

        # 2) mean_diff
        mean_diff_values = collect_values_for_pair(step_pair_map, step_group, pair, "mean_diff")
        feat[f"mean_diff_mean_{pair_key}"] = safe_mean(mean_diff_values)
        feat[f"mean_diff_max_{pair_key}"] = safe_max(mean_diff_values)
        feat[f"mean_diff_min_{pair_key}"] = safe_min(mean_diff_values)

        # 3) std_diff
        std_diff_values = collect_values_for_pair(step_pair_map, step_group, pair, "std_diff")
        feat[f"std_diff_mean_{pair_key}"] = safe_mean(std_diff_values)
        feat[f"std_diff_max_{pair_key}"] = safe_max(std_diff_values)
        feat[f"std_diff_min_{pair_key}"] = safe_min(std_diff_values)

    # 4) l2_distance (仅对 DISTANCE_PAIRS)
    for pair in DISTANCE_PAIRS:
        pair_key = f"p{pair[0]}_{pair[1]}"
        l2_values = collect_values_for_pair(step_pair_map, step_group, pair, "l2_distance")
        feat[f"l2_distance_mean_{pair_key}"] = safe_mean(l2_values)
        feat[f"l2_distance_max_{pair_key}"] = safe_max(l2_values)
        feat[f"l2_distance_min_{pair_key}"] = safe_min(l2_values)

    return feat


def extract_features_from_sample(sample, sample_idx, last_k_steps=10):
    """
    一个原始样本 -> 1 个样本
    只统计当前样本的最后 k 个 step
    """
    records = sample.get("mean_std_cos", {}).get("records", [])
    if not records:
        return None

    dtel_score = abs(sample.get("dtel_score", 0))
    pred_answer = str(sample.get("pred_answer", ""))
    clean_answer = str(sample.get("clean_answer", ""))
    significance = sample.get("significance", 0)
    if pred_answer != clean_answer:
        label = 1
    else:
        label = 0
    if significance == -1:
         return None
    if pred_answer == clean_answer:
        significance = 0
    
    # 
    # dtel_semantics = abs(se.score(clean_answer, clean_answer) - se.score(clean_answer, pred_answer))
    # label = 0
    # if dtel_semantics == 0: 
    #     label = 0
    # elif dtel_semantics <= 0.5:
    #     label = 1
    # else:
    #     label = 2

    # if dtel_score != 0:
    #     fault = sample.get("fault", None)
    #     if fault is not None:
    #         after = float(fault.get("after", None))
    #         if math.isnan(after) or pred_answer.startswith("!!!!!!") or pred_answer.endswith("!!!!!!"):
    #             label = 1

    # meteor = compute_bleu_and_meteor(clean_answer, pred_answer)
    # if dtel_semantics > 0.5 and meteor > 0.5:
    #     label = 1

    orig_id = sample.get("id", None)
    sample_uid = f"{orig_id}_{sample_idx}"

    step_pair_map = build_step_pair_map(records)
    all_steps = sorted(step_pair_map.keys())

    # 取最后 k 个 step
    window_steps = all_steps[-last_k_steps:] if len(all_steps) >= last_k_steps else all_steps

    if len(window_steps) == 0:
        return None

    feat = extract_features_for_step_group(step_pair_map, window_steps, SELECTED_LAYER_PAIRS)

    row = {
        "orig_id": orig_id,
        "sample_uid": sample_uid,
        "total_steps": len(all_steps),
        "last_k_steps": last_k_steps,
        "num_steps_used": len(window_steps),
        **feat,
        "significance": significance,
        "label": label
    }

    return row


# =========================
# 主流程
# =========================
def main():
    data = load_data(INPUT_JSON)
    all_rows = []

    for sample_idx, sample in enumerate(tqdm(data, desc="Processing samples")):
        row = extract_features_from_sample(
            sample,
            sample_idx=sample_idx,
            last_k_steps=LAST_K_STEPS
        )
        if row is not None:
            all_rows.append(row)

    df = pd.DataFrame(all_rows)

    feature_cols = [
        c for c in df.columns
        if c not in ["orig_id", "sample_uid", "total_steps", "last_k_steps",
                     "num_steps_used", "label"]
    ]

    print("特征数 =", len(feature_cols))
    print("特征列 =", feature_cols)

    print("sample_uid 是否全局唯一：", df["sample_uid"].is_unique)

    # 按 orig_id 切分
    orig_id_num = pd.to_numeric(df["orig_id"], errors="coerce")
    if orig_id_num.isna().any():
        bad_count = int(orig_id_num.isna().sum())
        raise ValueError(f"orig_id 中有 {bad_count} 个值无法转成数值，不能按阈值切分")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, valid_idx = next(gss.split(df, groups=df['orig_id']))

    train_df = df.iloc[train_idx].copy()
    valid_df = df.iloc[valid_idx].copy()
    train_orig_ids = set(train_df["orig_id"].unique())
    valid_orig_ids = set(valid_df["orig_id"].unique())
    overlap = train_orig_ids.intersection(valid_orig_ids)
    assert len(overlap) == 0, "train 和 valid 的 orig_id 有重叠，存在泄漏风险"

    import os
    output_dir = "./train_data"
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "llava-v1.5-7B_train_set.csv"), index=False, encoding="utf-8-sig")
    valid_df.to_csv(os.path.join(output_dir, "llava-v1.5-7B_valid_set.csv"), index=False, encoding="utf-8-sig")

    print("=" * 60)
    print(f"train_set.csv 行数: {len(train_df)}")
    print(f"valid_set.csv 行数: {len(valid_df)}")
    print(f"train orig_id unique: {train_df['orig_id'].nunique()}")
    print(f"valid orig_id unique: {valid_df['orig_id'].nunique()}")
    print(f"train/valid orig_id overlap: {len(overlap)}")
    print("=" * 60)


if __name__ == "__main__":
    # se = SimilarityEvaluator("cross-encoder/stsb-roberta-base")
    main()
