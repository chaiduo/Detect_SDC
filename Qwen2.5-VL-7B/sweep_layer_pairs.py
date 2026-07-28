import argparse
import json
import os
import sys

import pandas as pd
from tqdm import tqdm


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
REPOSITORY_ROOT = os.path.dirname(PROJECT_DIR)
SHARED_SOURCE = os.path.join(REPOSITORY_ROOT, "src")
if SHARED_SOURCE not in sys.path:
    sys.path.insert(0, SHARED_SOURCE)

from detect_sdc.detector.layer_pair_sweep import (  # noqa: E402
    run_binary_xgboost,
    run_binary_xgboost_compare_nan_modes,
)
from detect_sdc.features import (  # noqa: E402
    FeatureSpec,
    SampleSkipped,
    extract_feature_row,
    iter_json_samples,
)
from detect_sdc.features.jobs import FeatureRowCollector  # noqa: E402
from detect_sdc.splitting import (  # noqa: E402
    split_by_group,
    validate_identity_columns,
)


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


def split_and_write(df, output_dir, train_name, valid_name):
    os.makedirs(output_dir, exist_ok=True)
    if df.empty:
        raise ValueError("No rows extracted")

    validate_identity_columns(df, group_column=GROUP_COL)
    split = split_by_group(
        df,
        group_column=GROUP_COL,
        holdout_ratio=0.15,
        random_state=RANDOM_STATE,
    )
    train_df = split.train
    valid_df = split.holdout

    train_csv = os.path.join(output_dir, train_name)
    valid_csv = os.path.join(output_dir, valid_name)
    train_df.to_csv(train_csv, index=False, encoding="utf-8-sig")
    valid_df.to_csv(valid_csv, index=False, encoding="utf-8-sig")
    return train_csv, valid_csv, {
        "rows": int(len(df)),
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "orig_id_unique": int(df["orig_id"].nunique()),
        "train_orig_id_unique": split.summary.train_groups,
        "valid_orig_id_unique": split.summary.holdout_groups,
        "orig_id_overlap": split.summary.group_overlap,
        "label_significance_counts": {
            f"label={k[0]}, significance={k[1]}": int(v)
            for k, v in df[["label", "significance"]].value_counts().sort_index().to_dict().items()
        },
    }


def extract_dataset(dataset_name, config_names, force=False):
    dataset_cfg = DATASETS[dataset_name]
    collectors = {name: FeatureRowCollector() for name in config_names}
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
        specs = {
            config_name: FeatureSpec(
                selected_layer_pairs=tuple(PAIR_CONFIGS[config_name]),
                distance_pairs=tuple(PAIR_CONFIGS[config_name]),
                last_k_steps=LAST_K_STEPS,
                finite_only=True,
            )
            for config_name in pending_configs
        }
        uid_namespace = f"qwen25_vl_{dataset_name.lower()}"
        samples = iter_json_samples(dataset_cfg["input_json"])
        for sample in tqdm(samples, desc=f"Extract {dataset_name}"):
            for config_name in pending_configs:
                try:
                    row = extract_feature_row(
                        sample,
                        spec=specs[config_name],
                        uid_namespace=uid_namespace,
                    )
                except SampleSkipped:
                    continue
                collectors[config_name].add(row)

        for config_name in pending_configs:
            train_data_dir = os.path.join(output_root, config_name, "train_data")
            collector = collectors[config_name]
            df = pd.DataFrame(collector.rows)
            train_csv, valid_csv, stats = split_and_write(
                df,
                train_data_dir,
                dataset_cfg["train_name"],
                dataset_cfg["valid_name"],
            )
            stats["duplicate_samples"] = collector.duplicate_count
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
    parser.add_argument("--nan-modes", default="keep_all_nan", choices=["keep_all_nan", "drop_all_feature_nan", "both"])
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
