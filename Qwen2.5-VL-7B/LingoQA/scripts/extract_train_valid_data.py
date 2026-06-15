# 提取xgboost训练的CSV文件
import json
import math
import csv
from collections import defaultdict
import pandas as pd
import numpy as np
from nltk.translate.meteor_score import meteor_score
import nltk
from tqdm import tqdm
from sentence_transformers import CrossEncoder
# =========================
# 配置区域
# =========================
INPUT_JSON = "/data1/home/dataset_share/cd_data/Qwen2.5-VL-7B/final/detect_LingoQA_Qwen_with_sem.jsonl"    
OUTPUT_CSV = "sdc_features.csv"    # 输出特征文件
STEP_GROUP_SIZE = 50  # 越大效果越好，因为能获得所有信息；越小效果越差，因为只能看到一个窗口的特征

COS_PAIRS       = [(6, 7), (23, 24), (24, 25), (25, 26), (26, 27)]
MEAN_DIFF_PAIRS = [(6, 7), (23, 24), (24, 25), (25, 26), (26, 27)]
STD_DIFF_PAIRS  = [(6, 7), (23, 24), (24, 25), (25, 26), (26, 27)]


class SimilarityEvaluator:
    def __init__(self, model_name):
        self.model = CrossEncoder(model_name)

    def score(self, text1, text2):
        return float(self.model.predict([(str(text1), str(text2))])[0])
    
def compute_bleu_and_meteor(reference_sentence, candidate_sentence):
    # 使用 word_tokenize
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

    # 优先尝试整个文件作为 JSON 解析
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
    except json.JSONDecodeError:
        pass

    # 否则按 JSONL 解析
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


def safe_mean(values):
    return float(np.mean(values)) if values else np.nan

def safe_min(values):
    return float(np.min(values)) if values else np.nan

def safe_max(values):
    return float(np.max(values)) if values else np.nan

def extract_group_features(step_pair_map, step_list):
    """
    对一个 step 分组提取 20 个特征：
    - cos_sim: 6个pair * (mean, min) = 12
    - mean_diff: 2个pair * (mean, max) = 4
    - std_diff: 2个pair * (mean, max) = 4
    """
    feat = {}

    # 1) cos_sim
    for pair in COS_PAIRS:
        values = []
        for step in step_list:
            rec = step_pair_map.get(step, {}).get(pair)
            if rec is not None and "cos_sim" in rec:
                values.append(rec["cos_sim"])

        feat[f"cos_sim_mean_{pair[0]}_{pair[1]}"] = safe_mean(values)
        feat[f"cos_sim_min_{pair[0]}_{pair[1]}"] = safe_min(values)
        feat[f"cos_sim_range_{pair[0]}_{pair[1]}"] = (
            safe_max(values) - safe_min(values) if len(values) > 0 else np.nan
        )

    # 2) mean_diff
    for pair in MEAN_DIFF_PAIRS:
        raw_values = []
        abs_values = []
        for step in step_list:
            rec = step_pair_map.get(step, {}).get(pair)
            if rec is not None and "mean_diff" in rec:
                v = rec["mean_diff"]
                raw_values.append(v)
                abs_values.append(abs(v))

        feat[f"mean_diff_mean_{pair[0]}_{pair[1]}"] = safe_mean(abs_values)
        feat[f"mean_diff_max_{pair[0]}_{pair[1]}"] = safe_max(abs_values)
        feat[f"mean_diff_range_{pair[0]}_{pair[1]}"] = (
            safe_max(raw_values) - safe_min(raw_values) if len(raw_values) > 0 else np.nan
        )

    # 3) std_diff
    for pair in STD_DIFF_PAIRS:
        raw_values = []
        abs_values = []
        for step in step_list:
            rec = step_pair_map.get(step, {}).get(pair)
            if rec is not None and "std_diff" in rec:
                v = rec["std_diff"]
                raw_values.append(v)
                abs_values.append(abs(v))

        feat[f"std_diff_mean_{pair[0]}_{pair[1]}"] = safe_mean(abs_values)
        feat[f"std_diff_max_{pair[0]}_{pair[1]}"] = safe_max(abs_values)
        feat[f"std_diff_range_{pair[0]}_{pair[1]}"] = (
            safe_max(raw_values) - safe_min(raw_values) if len(raw_values) > 0 else np.nan
        )

    return feat


def extract_features_from_sample(sample, sample_idx, step_group_size=6, threshold=0.3):
    """
    一个原始样本 -> 多个分组样本
    """
    records = sample.get("mean_std_cos", {}).get("records", [])
    if not records:
        return []

    dtel_score = abs(sample.get("dtel_score", None))
    gt_answer = str(sample.get("gt_answer", None))
    clean_answer = str(sample.get("clean_answer", None))
    pred_answer = str(sample.get("pred_answer", None))

    # 打标签的艺术！！！！！！
    dtel_semantics = sample.get("dtel_semantics", None)
    # dtel_semantics = abs(se.score(clean_answer, clean_answer) - se.score(clean_answer, pred_answer))
    label = 0
    if dtel_semantics == 0: 
        label = 0
    elif dtel_semantics <= 0.5:
        label = 1
    else:
        label = 2
    #先按照cross-encoder打标签
    # label = 0
    # if dtel_score > 0 and dtel_score < threshold:
    #     label = 1
    # elif dtel_score >= threshold:
    #     label = 2
    
    # #METEOR作为辅助
    # meteor = compute_bleu_and_meteor(clean_answer, pred_answer)
    # if sim_score > 0.5 and meteor > 0.4:
    #     label = 1

    # #给较大偏差数据打标签
    if dtel_score != 0:
        fault = sample.get("fault", None)
        if fault is not None:
            after = float(fault.get("after", None))
            if math.isnan(after) or pred_answer.startswith("!!!!!!") or pred_answer.endswith("!!!!!!"):
                label = 2
    
    # label = sample.get("is_sdc", None)
    orig_id = sample.get("id", None)

    # 保证每条原始 sample 都有唯一标识
    sample_uid = f"{orig_id}_{sample_idx}"

    step_pair_map = build_step_pair_map(records)
    all_steps = sorted(step_pair_map.keys())

    rows = []
    for group_idx, start in enumerate(range(0, len(all_steps), step_group_size)):
        step_group = all_steps[start:start + step_group_size]
        feat = extract_group_features(step_pair_map, step_group)

        group_uid = f"{sample_uid}_g{group_idx}"

        row = {
            "orig_id": orig_id,
            "sample_uid": sample_uid,
            "group_id": group_idx,         # 局部编号
            "group_uid": group_uid,        # 全局唯一编号
            "step_start": step_group[0],
            "step_end": step_group[-1],
            "num_steps_in_group": len(step_group),
            **feat,
            "label": label
        }
        rows.append(row)

    return rows


# =========================
# 主流程
# =========================
def main():
    data = load_data(INPUT_JSON)
    all_rows = []

    for sample_idx, sample in enumerate(data):
        rows = extract_features_from_sample(
            sample,
            sample_idx=sample_idx,
            step_group_size=STEP_GROUP_SIZE,
            threshold=0.5
        )
        if rows is not None: 
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    feature_cols = [
        c for c in df.columns
        if c not in ["orig_id", "sample_uid", "group_id", "group_uid",
                     "step_start", "step_end", "num_steps_in_group", "label"]
    ]

    print("特征数 =", len(feature_cols))
    print("特征列 =", feature_cols)

    print("sample_uid 是否唯一按原始 sample 区分：", df["orig_id"].nunique())
    print("group_uid 是否全局唯一：", df["group_uid"].is_unique)

    # 按 orig_id 切分
    orig_id_num = pd.to_numeric(df["orig_id"], errors="coerce")
    if orig_id_num.isna().any():
        bad_count = int(orig_id_num.isna().sum())
        raise ValueError(f"orig_id 中有 {bad_count} 个值无法转成数值，不能按阈值切分")

    train_df = df[orig_id_num < 4250].copy()
    valid_df = df[orig_id_num >= 4250].copy()

    train_orig_ids = set(train_df["orig_id"].unique())
    valid_orig_ids = set(valid_df["orig_id"].unique())
    overlap = train_orig_ids.intersection(valid_orig_ids)

    assert len(overlap) == 0, "train 和 valid 的 orig_id 有重叠，存在泄漏风险"

    train_df.to_csv("./train_data/Qwen2.5_LingoQA_train_set.csv", index=False, encoding="utf-8-sig")
    valid_df.to_csv("./train_data/Qwen2.5_LingoQA_valid_set.csv", index=False, encoding="utf-8-sig")

    print("=" * 60)
    print(f"train_set.csv 行数: {len(train_df)}")
    print(f"valid_set.csv 行数: {len(valid_df)}")
    print(f"train orig_id unique: {train_df['orig_id'].nunique()}")
    print(f"valid orig_id unique: {valid_df['orig_id'].nunique()}")
    print(f"train/valid orig_id overlap: {len(overlap)}")
    print("=" * 60)

if __name__ == "__main__":
    x=0
    se = SimilarityEvaluator("/data0/home/lc/cd/stsb-roberta-base")
    main()
