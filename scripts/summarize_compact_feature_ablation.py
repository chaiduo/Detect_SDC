#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


MODELS = {
    "Qwen2.5-VL-7B": "Qwen2.5-VL-7B",
    "InternVL3-8B": "InternVL3-8B",
    "LLaVA-1.5-7B": "llava-v1.5-7B",
}
DATASETS = ("EarthVQA", "LingoQA", "VQAv2")
METRICS = ("cos_sim", "mean_diff", "std_diff", "l2_distance")
STATISTICS = ("mean", "max", "min")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=root)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "analysis/compact_feature_ablation_20260815",
    )
    return parser.parse_args()


def load_results(root: Path) -> pd.DataFrame:
    frames = []
    for model, directory in MODELS.items():
        path = (
            root
            / directory
            / "compact_feature_ablation_20260815"
            / "summary_all_datasets.csv"
        )
        frame = pd.read_csv(path)
        if set(frame["model"]) != {model}:
            raise ValueError(f"Unexpected model label in {path}")
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    expected = {
        (model, dataset, metric, statistic)
        for model in MODELS
        for dataset in DATASETS
        for metric in METRICS
        for statistic in STATISTICS
    }
    actual = set(
        result[["model", "dataset", "metric", "statistic"]].itertuples(
            index=False,
            name=None,
        )
    )
    if actual != expected:
        raise ValueError(
            f"Result matrix mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    if set(result["feature_count"]) != {6}:
        raise ValueError("Every compact configuration must contain 6 features")
    return result


def ranking(results: pd.DataFrame) -> pd.DataFrame:
    ranked = (
        results.groupby(["metric", "statistic"], sort=False)
        .agg(
            mean_f1=("f1", "mean"),
            std_f1=("f1", "std"),
            mean_delta_f1_pp=("delta_f1_pp", "mean"),
            worst_delta_f1_pp=("delta_f1_pp", "min"),
            best_delta_f1_pp=("delta_f1_pp", "max"),
        )
        .reset_index()
    )
    ranked["mean_f1_percent"] = 100.0 * ranked.pop("mean_f1")
    ranked["std_f1_percent"] = 100.0 * ranked.pop("std_f1")
    return ranked.sort_values("mean_f1_percent", ascending=False)


def best_by_model_dataset(results: pd.DataFrame) -> pd.DataFrame:
    indices = results.groupby(["model", "dataset"], sort=False)["f1"].idxmax()
    columns = (
        "model",
        "dataset",
        "configuration",
        "metric",
        "statistic",
        "f1",
        "full_f1",
        "delta_f1_pp",
    )
    best = results.loc[indices, columns].copy()
    best["f1_percent"] = 100.0 * best.pop("f1")
    best["full_f1_percent"] = 100.0 * best.pop("full_f1")
    return best


def main() -> int:
    args = parse_args()
    root = args.repository_root.resolve()
    destination = args.output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    results = load_results(root)
    ranked = ranking(results)
    average_matrix = (
        results.pivot_table(
            index="metric",
            columns="statistic",
            values="f1",
        )
        .loc[list(METRICS), list(STATISTICS)]
        * 100.0
    )
    model_average = (
        results.groupby(
            ["model", "metric", "statistic"],
            sort=False,
        )["f1"]
        .mean()
        .mul(100.0)
        .rename("mean_f1_percent")
        .reset_index()
    )
    best = best_by_model_dataset(results)

    results.to_csv(destination / "detailed_results.csv", index=False)
    ranked.to_csv(destination / "configuration_ranking.csv", index=False)
    average_matrix.to_csv(destination / "average_f1_matrix.csv")
    model_average.to_csv(destination / "model_average_results.csv", index=False)
    best.to_csv(destination / "best_by_model_dataset.csv", index=False)

    winner = ranked.iloc[0]
    full_mean = float(results["full_f1"].drop_duplicates().mean() * 100.0)
    summary = {
        "models": list(MODELS),
        "datasets": list(DATASETS),
        "configuration_count": len(METRICS) * len(STATISTICS),
        "detector_count": len(results),
        "features_per_configuration": 6,
        "full_feature_count": 72,
        "full_mean_f1_percent": full_mean,
        "best_configuration": (
            f"{winner['metric']}_{winner['statistic']}"
        ),
        "best_mean_f1_percent": float(winner["mean_f1_percent"]),
        "best_mean_delta_f1_pp": float(winner["mean_delta_f1_pp"]),
        "best_worst_delta_f1_pp": float(winner["worst_delta_f1_pp"]),
        "best_meets_average_loss_at_most_1pp": bool(
            winner["mean_delta_f1_pp"] >= -1.0
        ),
        "best_meets_every_case_loss_at_most_2pp": bool(
            winner["worst_delta_f1_pp"] >= -2.0
        ),
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
