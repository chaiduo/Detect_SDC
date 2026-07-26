import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from binary_xgboost_common import run_binary_xgboost, run_binary_xgboost_compare_nan_modes


RANDOM_STATE = 42
LAST_K_STEPS = 50
GROUP_COL = "orig_id"

DATASETS = {
    "LingoQA": {
        "input_json": os.path.join(
            PROJECT_DIR,
            "LingoQA/json/detect_LingoQA_Qwen_with_sem_labeled.jsonl",
        ),
        "train_name": "Qwen2.5_LingoQA_train_set.csv",
        "valid_name": "Qwen2.5_LingoQA_valid_set.csv",
    },
    "EarthVQA": {
        "input_json": os.path.join(
            PROJECT_DIR,
            "EarthVQA/json/detect_EarthVQA_Qwen_with_sem_project_labeled.jsonl",
        ),
        "train_name": "Qwen2.5_EarthVQA_train_set.csv",
        "valid_name": "Qwen2.5_EarthVQA_valid_set.csv",
    },
    "VQAv2": {
        "input_json": os.path.join(
            PROJECT_DIR,
            "VQAv2/json/detect_VQAv2_Qwen_with_sem_project_labeled.jsonl",
        ),
        "train_name": "Qwen2.5_VQAv2_train_set.csv",
        "valid_name": "Qwen2.5_VQAv2_valid_set.csv",
    },
}

PAIR_CONFIGS = {
    "current": [(6, 7), (22, 23), (23, 24), (24, 25), (25, 26), (26, 27)],
    "spread_6": [(0, 1), (5, 6), (10, 11), (15, 16), (20, 21), (26, 27)],
    "even_adjacent": [(i, i + 1) for i in range(0, 27, 2)],
    "late_dense": [(i, i + 1) for i in range(20, 27)],
    "mid_late_dense": [(i, i + 1) for i in range(12, 27)],
    "all_adjacent": [(i, i + 1) for i in range(27)],
}


def build_label_and_significance(sample):
    if "significance" not in sample:
        return None, None

    try:
        significance = int(sample.get("significance"))
    except (TypeError, ValueError):
        return None, None

    pred_answer = str(sample.get("pred_answer", ""))
    clean_answer = str(sample.get("clean_answer", ""))
    label = 1 if pred_answer != clean_answer else 0
    if pred_answer == clean_answer:
        significance = 0

    if significance not in (0, 1, 2):
        return None, None

    return label, significance


def build_step_pair_map(records):
    step_pair_map = defaultdict(dict)
    for record in records:
        step = record["step"]
        pair = (record["src_layer"], record["tgt_layer"])
        step_pair_map[step][pair] = record
    return step_pair_map


def safe_mean(values):
    return float(np.mean(values)) if values else np.nan


def safe_min(values):
    return float(np.min(values)) if values else np.nan


def safe_max(values):
    return float(np.max(values)) if values else np.nan


def collect_values_for_pair(step_pair_map, step_group, pair, key):
    values = []
    for step in step_group:
        record = step_pair_map.get(step, {}).get(pair)
        if record is not None and key in record:
            values.append(record[key])
    return values


def extract_features_for_pairs(step_pair_map, step_group, pairs):
    features = {}
    for pair in pairs:
        pair_key = f"p{pair[0]}_{pair[1]}"

        cos_values = collect_values_for_pair(step_pair_map, step_group, pair, "cos_sim")
        features[f"cos_sim_mean_{pair_key}"] = safe_mean(cos_values)
        features[f"cos_sim_max_{pair_key}"] = safe_max(cos_values)
        features[f"cos_sim_min_{pair_key}"] = safe_min(cos_values)

        mean_diff_values = collect_values_for_pair(step_pair_map, step_group, pair, "mean_diff")
        features[f"mean_diff_mean_{pair_key}"] = safe_mean(mean_diff_values)
        features[f"mean_diff_max_{pair_key}"] = safe_max(mean_diff_values)
        features[f"mean_diff_min_{pair_key}"] = safe_min(mean_diff_values)

        std_diff_values = collect_values_for_pair(step_pair_map, step_group, pair, "std_diff")
        features[f"std_diff_mean_{pair_key}"] = safe_mean(std_diff_values)
        features[f"std_diff_max_{pair_key}"] = safe_max(std_diff_values)
        features[f"std_diff_min_{pair_key}"] = safe_min(std_diff_values)

    for pair in pairs:
        pair_key = f"p{pair[0]}_{pair[1]}"
        l2_values = collect_values_for_pair(step_pair_map, step_group, pair, "l2_distance")
        features[f"l2_distance_mean_{pair_key}"] = safe_mean(l2_values)
        features[f"l2_distance_max_{pair_key}"] = safe_max(l2_values)
        features[f"l2_distance_min_{pair_key}"] = safe_min(l2_values)

    return features


def make_row(sample, sample_idx, pairs):
    records = sample.get("mean_std_cos", {}).get("records", [])
    if not records:
        return None

    label, significance = build_label_and_significance(sample)
    if label is None:
        return None

    step_pair_map = build_step_pair_map(records)
    all_steps = sorted(step_pair_map.keys())
    if not all_steps:
        return None

    window_steps = all_steps[-LAST_K_STEPS:] if len(all_steps) >= LAST_K_STEPS else all_steps
    orig_id = sample.get("id", None)
    return {
        "orig_id": orig_id,
        "sample_uid": f"{orig_id}_{sample_idx}",
        "total_steps": len(all_steps),
        "last_k_steps": LAST_K_STEPS,
        "num_steps_used": len(window_steps),
        **extract_features_for_pairs(step_pair_map, window_steps, pairs),
        "significance": significance,
        "label": label,
    }


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def split_and_write(df, output_dir, train_name, valid_name):
    os.makedirs(output_dir, exist_ok=True)
    if df.empty:
        raise ValueError("No rows extracted")

    orig_id_num = pd.to_numeric(df["orig_id"], errors="coerce")
    if orig_id_num.isna().any():
        raise ValueError(f"orig_id 中有 {int(orig_id_num.isna().sum())} 个值无法转成数值")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=RANDOM_STATE)
    train_idx, valid_idx = next(gss.split(df, groups=df[GROUP_COL]))
    train_df = df.iloc[train_idx].copy()
    valid_df = df.iloc[valid_idx].copy()

    overlap = set(train_df[GROUP_COL].unique()) & set(valid_df[GROUP_COL].unique())
    if overlap:
        raise AssertionError(f"train/valid {GROUP_COL} overlap: {len(overlap)}")

    train_csv = os.path.join(output_dir, train_name)
    valid_csv = os.path.join(output_dir, valid_name)
    train_df.to_csv(train_csv, index=False, encoding="utf-8-sig")
    valid_df.to_csv(valid_csv, index=False, encoding="utf-8-sig")
    return train_csv, valid_csv, {
        "rows": int(len(df)),
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "orig_id_unique": int(df["orig_id"].nunique()),
        "train_orig_id_unique": int(train_df["orig_id"].nunique()),
        "valid_orig_id_unique": int(valid_df["orig_id"].nunique()),
        "label_significance_counts": {
            f"label={k[0]}, significance={k[1]}": int(v)
            for k, v in df[["label", "significance"]].value_counts().sort_index().to_dict().items()
        },
    }


def extract_dataset(dataset_name, config_names, force=False):
    dataset_cfg = DATASETS[dataset_name]
    rows_by_config = {name: [] for name in config_names}
    output_root = os.path.join(PROJECT_DIR, "pair_sweep", dataset_name)

    csv_paths = {}
    pending_configs = []
    for config_name in config_names:
        train_csv = os.path.join(output_root, config_name, "train_data", dataset_cfg["train_name"])
        valid_csv = os.path.join(output_root, config_name, "train_data", dataset_cfg["valid_name"])
        if not force and os.path.exists(train_csv) and os.path.exists(valid_csv):
            csv_paths[config_name] = (train_csv, valid_csv, None)
        else:
            pending_configs.append(config_name)

    if pending_configs:
        for sample_idx, sample in enumerate(tqdm(iter_jsonl(dataset_cfg["input_json"]), desc=f"Extract {dataset_name}")):
            records = sample.get("mean_std_cos", {}).get("records", [])
            if not records:
                continue
            label, significance = build_label_and_significance(sample)
            if label is None:
                continue

            step_pair_map = build_step_pair_map(records)
            all_steps = sorted(step_pair_map.keys())
            if not all_steps:
                continue
            window_steps = all_steps[-LAST_K_STEPS:] if len(all_steps) >= LAST_K_STEPS else all_steps
            orig_id = sample.get("id", None)

            base_row = {
                "orig_id": orig_id,
                "sample_uid": f"{orig_id}_{sample_idx}",
                "total_steps": len(all_steps),
                "last_k_steps": LAST_K_STEPS,
                "num_steps_used": len(window_steps),
                "significance": significance,
                "label": label,
            }
            for config_name in pending_configs:
                row = {
                    **base_row,
                    **extract_features_for_pairs(step_pair_map, window_steps, PAIR_CONFIGS[config_name]),
                }
                rows_by_config[config_name].append(row)

        for config_name in pending_configs:
            train_data_dir = os.path.join(output_root, config_name, "train_data")
            df = pd.DataFrame(rows_by_config[config_name])
            train_csv, valid_csv, stats = split_and_write(
                df,
                train_data_dir,
                dataset_cfg["train_name"],
                dataset_cfg["valid_name"],
            )
            csv_paths[config_name] = (train_csv, valid_csv, stats)

    return csv_paths


def summarize_metric(summary, metric_split):
    task = summary["tasks"]["significant_sdc"]
    metrics = task[metric_split]
    return {
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "threshold": metrics["threshold"],
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "tn": metrics["tn"],
    }


def run_sweep(dataset_names, config_names, nan_modes, force_extract=False, train=True):
    all_rows = []
    for dataset_name in dataset_names:
        csv_paths = extract_dataset(dataset_name, config_names, force=force_extract)
        for config_name in config_names:
            train_csv, valid_csv, extract_stats = csv_paths[config_name]
            row_base = {
                "dataset": dataset_name,
                "pair_config": config_name,
                "pairs": json.dumps(PAIR_CONFIGS[config_name]),
                "pair_count": len(PAIR_CONFIGS[config_name]),
                "feature_count": len(PAIR_CONFIGS[config_name]) * 12,
            }
            if extract_stats:
                row_base.update({f"extract_{k}": v for k, v in extract_stats.items() if k != "label_significance_counts"})

            if not train:
                all_rows.append(row_base)
                continue

            if nan_modes == "both":
                combined = run_binary_xgboost_compare_nan_modes(
                    train_csv=train_csv,
                    valid_csv=valid_csv,
                    group_col=GROUP_COL,
                    threshold_selection_mode="best_f1",
                )
                for mode_name, mode_summary in combined["mode_summaries"].items():
                    split_name = "valid_full_metrics" if mode_name == "keep_all_nan" else "valid_non_all_nan_metrics"
                    all_rows.append({
                        **row_base,
                        "nan_mode": mode_name,
                        "metric_split": split_name,
                        **summarize_metric(mode_summary, split_name),
                    })
            else:
                drop_all_feature_nan = nan_modes == "drop_all_feature_nan"
                mode_summary = run_binary_xgboost(
                    train_csv=train_csv,
                    valid_csv=valid_csv,
                    group_col=GROUP_COL,
                    threshold_selection_mode="best_f1",
                    drop_all_feature_nan=drop_all_feature_nan,
                    output_subdir=nan_modes,
                )
                split_name = "valid_non_all_nan_metrics" if drop_all_feature_nan else "valid_full_metrics"
                all_rows.append({
                    **row_base,
                    "nan_mode": nan_modes,
                    "metric_split": split_name,
                    **summarize_metric(mode_summary, split_name),
                })

    result_df = pd.DataFrame(all_rows)
    dataset_tag = "_".join(dataset_names)
    output_path = os.path.join(PROJECT_DIR, "pair_sweep", f"summary_{dataset_tag}_{nan_modes}.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved pair sweep summary to: {output_path}")
    if train and not result_df.empty:
        print(result_df.sort_values(["dataset", "nan_mode", "f1"], ascending=[True, True, False]).to_string(index=False))
    return result_df


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep Qwen2.5-VL-7B layer pair feature configs.")
    parser.add_argument("--datasets", nargs="+", default=["LingoQA", "EarthVQA", "VQAv2"], choices=sorted(DATASETS))
    parser.add_argument("--configs", nargs="+", default=list(PAIR_CONFIGS), choices=sorted(PAIR_CONFIGS))
    parser.add_argument("--nan-modes", default="drop_all_feature_nan", choices=["keep_all_nan", "drop_all_feature_nan", "both"])
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--no-train", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    run_sweep(
        dataset_names=args.datasets,
        config_names=args.configs,
        nan_modes=args.nan_modes,
        force_extract=args.force_extract,
        train=not args.no_train,
    )


if __name__ == "__main__":
    main()
