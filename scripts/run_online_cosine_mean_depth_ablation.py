#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from detect_sdc.detector.xgboost import (
    add_significant_sdc_target,
    binary_metrics,
    prepare_features,
    train_binary_model,
)
from detect_sdc.features.extraction import FeatureSpec
from detect_sdc.features.jobs import load_feature_job
from detect_sdc.splitting import split_by_group
from run_feature_group_ablation import (
    DATASETS,
    FULL_PAIRS,
    METRIC_KEYS,
    MODELS,
    detector_config,
)


DEPTHS = (1, 2, 3, 4, 5, 6)
ONLINE_STEPS = 2
ONLINE_SPEC = FeatureSpec(
    selected_layer_pairs=FULL_PAIRS,
    distance_pairs=FULL_PAIRS,
    last_k_steps=ONLINE_STEPS,
    finite_only=True,
    step_window="prefix",
)
FEATURE_COLUMNS = tuple(
    column
    for column in ONLINE_SPEC.feature_columns
    if column.startswith("cos_sim_mean_")
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=(*MODELS, "all"),
        default="all",
    )
    parser.add_argument(
        "--dataset",
        choices=(*DATASETS, "all"),
        default="all",
    )
    parser.add_argument("--repository-root", type=Path, default=root)
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configs/experiments/current.yaml",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def train_depth(
    *,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    depth: int,
    base_config: Any,
    destination: Path,
    baseline_f1: float,
    overwrite: bool,
) -> dict[str, Any]:
    metadata_path = destination / "detector_metadata.json"
    model_path = destination / "detector.ubj"
    if metadata_path.is_file() and model_path.is_file() and not overwrite:
        return json.loads(metadata_path.read_text(encoding="utf-8"))["result"]

    split = split_by_group(
        train,
        group_column="orig_id",
        holdout_ratio=base_config.test_ratio,
        random_state=base_config.random_state,
    )
    config = replace(base_config, max_depth=depth, verbose=False)
    model, training = train_binary_model(
        prepare_features(split.train, list(FEATURE_COLUMNS)),
        split.train["significant_sdc_target"].astype(int),
        prepare_features(split.holdout, list(FEATURE_COLUMNS)),
        split.holdout["significant_sdc_target"].astype(int),
        config=config,
    )
    probability = model.predict_proba(
        prepare_features(valid, list(FEATURE_COLUMNS))
    )
    metrics = binary_metrics(
        valid["significant_sdc_target"].astype(int),
        np.argmax(probability, axis=1),
        probability,
    )["target_significant_sdc"]
    booster = model.get_booster()
    best_iteration = training["best_iteration"]
    if best_iteration is not None:
        booster = booster[: int(best_iteration) + 1]
    complexity = _tree_complexity(booster)
    destination.mkdir(parents=True, exist_ok=True)
    booster.save_model(model_path)
    result = {
        "max_depth": depth,
        "feature_count": len(FEATURE_COLUMNS),
        **{key: metrics[key] for key in METRIC_KEYS},
        "baseline_72d_f1": baseline_f1,
        "delta_f1_pp": 100.0 * (float(metrics["f1"]) - baseline_f1),
        "best_iteration": best_iteration,
        "best_score": training["best_score"],
        **complexity,
        "model_size_bytes": model_path.stat().st_size,
    }
    _write_json(
        metadata_path,
        {
            "feature_profile": "cos_sim_mean",
            "online_steps": ONLINE_STEPS,
            "feature_columns": FEATURE_COLUMNS,
            "config": asdict(config),
            "training": training,
            "valid_metrics": metrics,
            "result": result,
        },
    )
    return result


def run_dataset(
    *,
    root: Path,
    config_path: Path,
    model: str,
    dataset: str,
    output_base: Path,
    overwrite: bool,
) -> list[dict[str, Any]]:
    model_key, model_directory = MODELS[model]
    load_feature_job(
        config_path,
        f"{model_key}_{DATASETS[dataset]}",
        repository_root=root,
    )
    input_root = (
        root
        / model_directory
        / "online_step_ablation_20260814"
        / dataset
        / "k_2"
    )
    train = add_significant_sdc_target(
        pd.read_csv(input_root / "train_data/train.csv")
    )
    valid = add_significant_sdc_target(
        pd.read_csv(input_root / "train_data/valid_fixed_k50.csv")
    )
    missing = sorted(
        set(FEATURE_COLUMNS) - set(train)
        | set(FEATURE_COLUMNS) - set(valid)
    )
    if missing:
        raise ValueError(f"Missing compact online features: {missing}")
    baseline_summary = json.loads(
        (input_root / "output/metrics_summary.json").read_text(
            encoding="utf-8"
        )
    )
    baseline_metrics = baseline_summary["valid_full_metrics"][
        "target_significant_sdc"
    ]
    baseline_f1 = float(baseline_metrics["f1"])
    base_config = detector_config(config_path, model_key)
    dataset_root = output_base / dataset
    rows = []
    for depth in DEPTHS:
        print(
            f"[online-cos-mean-depth] {model} {dataset} depth={depth}",
            flush=True,
        )
        result = train_depth(
            train=train,
            valid=valid,
            depth=depth,
            base_config=base_config,
            destination=dataset_root / f"depth_{depth}",
            baseline_f1=baseline_f1,
            overwrite=overwrite,
        )
        rows.append(
            {
                "model": model,
                "dataset": dataset,
                "feature_profile": "cos_sim_mean",
                "online_steps": ONLINE_STEPS,
                "validation_rows": len(valid),
                "positive_samples": int(
                    valid["significant_sdc_target"].sum()
                ),
                "negative_samples": int(
                    len(valid) - valid["significant_sdc_target"].sum()
                ),
                **result,
            }
        )
        _write_summary(dataset_root / "summary.csv", rows)
    return rows


def _tree_complexity(booster: Any) -> dict[str, Any]:
    trees = booster.get_dump(dump_format="json")
    nodes = 0
    leaves = 0
    observed_depth = 0
    for tree in trees:
        tree_nodes, tree_leaves, tree_depth = _count_tree(
            json.loads(tree)
        )
        nodes += tree_nodes
        leaves += tree_leaves
        observed_depth = max(observed_depth, tree_depth)
    tree_count = len(trees)
    return {
        "tree_count": tree_count,
        "total_nodes": nodes,
        "total_leaves": leaves,
        "observed_max_depth": observed_depth,
        "mean_nodes_per_tree": (
            float(nodes / tree_count) if tree_count else 0.0
        ),
    }


def _count_tree(node: Mapping[str, Any], depth: int = 0) -> tuple[int, int, int]:
    children = node.get("children")
    if not children:
        return 1, 1, depth
    nodes = 1
    leaves = 0
    maximum = depth
    for child in children:
        child_nodes, child_leaves, child_depth = _count_tree(
            child,
            depth + 1,
        )
        nodes += child_nodes
        leaves += child_leaves
        maximum = max(maximum, child_depth)
    return nodes, leaves, maximum


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    if len(FEATURE_COLUMNS) != len(FULL_PAIRS):
        raise AssertionError("CosSim-Mean must contain one feature per pair")
    args = parse_args()
    root = args.repository_root.resolve()
    config_path = args.config.resolve()
    models = tuple(MODELS) if args.model == "all" else (args.model,)
    datasets = tuple(DATASETS) if args.dataset == "all" else (args.dataset,)
    all_rows = []
    for model in models:
        _, model_directory = MODELS[model]
        output_base = (
            args.output_root.resolve()
            if args.output_root is not None
            else root
            / model_directory
            / "online_cosine_mean_depth_ablation_20260815"
        )
        model_rows = []
        for dataset in datasets:
            model_rows.extend(
                run_dataset(
                    root=root,
                    config_path=config_path,
                    model=model,
                    dataset=dataset,
                    output_base=output_base,
                    overwrite=args.overwrite,
                )
            )
        _write_summary(output_base / "summary_all_datasets.csv", model_rows)
        all_rows.extend(model_rows)
    if args.output_root is not None:
        _write_summary(
            args.output_root.resolve() / "summary_all.csv",
            all_rows,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
