#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


MODELS = {
    "Qwen2.5-VL-7B": "qwen25_vl",
    "InternVL3-8B": "internvl3",
    "LLaVA-1.5-7B": "llava15",
}
DATASETS = ("EarthVQA", "LingoQA", "VQAv2")
LEAVE_ONE_OUT = (
    "full",
    "without_cos_sim",
    "without_mean_diff",
    "without_std_diff",
    "without_l2_distance",
)
SINGLE_GROUP = (
    "only_cos_sim",
    "only_mean_diff",
    "only_std_diff",
    "only_l2_distance",
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=root)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "analysis/iclr_v2/feature_groups",
    )
    return parser.parse_args()


def load_results(root: Path) -> pd.DataFrame:
    frames = []
    for model, directory in MODELS.items():
        path = (
            root
            / "artifacts/iclr_v2/ablations/feature_groups"
            / directory
            / "summary_all_datasets.csv"
        )
        frame = pd.read_csv(path)
        if set(frame["model"]) != {model}:
            raise ValueError(f"Unexpected model label in {path}")
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    expected = {
        (model, dataset, configuration)
        for model in MODELS
        for dataset in DATASETS
        for configuration in (*LEAVE_ONE_OUT, *SINGLE_GROUP)
    }
    actual = set(
        result[["model", "dataset", "configuration"]].itertuples(
            index=False,
            name=None,
        )
    )
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"Result matrix mismatch: missing={missing}, extra={extra}")
    return result


def average_by_model(
    results: pd.DataFrame,
    configurations: tuple[str, ...],
) -> pd.DataFrame:
    selected = results.loc[results["configuration"].isin(configurations)]
    averaged = (
        selected.groupby(["model", "configuration"], sort=False)["f1"]
        .mean()
        .unstack()
        .loc[list(MODELS), list(configurations)]
    )
    averaged.loc["Average"] = averaged.mean(axis=0)
    return averaged * 100.0


def average_by_configuration(results: pd.DataFrame) -> pd.DataFrame:
    rows = (
        results.groupby(["ablation_type", "configuration"], sort=False)
        .agg(
            mean_f1=("f1", "mean"),
            std_f1=("f1", "std"),
            mean_delta_f1_pp=("delta_f1_pp", "mean"),
            min_delta_f1_pp=("delta_f1_pp", "min"),
            max_delta_f1_pp=("delta_f1_pp", "max"),
        )
        .reset_index()
    )
    rows["mean_f1_percent"] = 100.0 * rows.pop("mean_f1")
    rows["std_f1_percent"] = 100.0 * rows.pop("std_f1")
    return rows


def main() -> int:
    args = parse_args()
    root = args.repository_root.resolve()
    destination = args.output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    results = load_results(root)
    leave_one_out = average_by_model(results, LEAVE_ONE_OUT)
    single_group = average_by_model(results, SINGLE_GROUP)
    aggregate = average_by_configuration(results)

    results.to_csv(destination / "detailed_results.csv", index=False)
    leave_one_out.to_csv(destination / "leave_one_out_by_model.csv")
    single_group.to_csv(destination / "single_group_by_model.csv")
    aggregate.to_csv(destination / "aggregate_by_configuration.csv", index=False)

    full_mean = float(
        aggregate.loc[
            aggregate["configuration"] == "full",
            "mean_f1_percent",
        ].iloc[0]
    )
    summary = {
        "models": list(MODELS),
        "datasets": list(DATASETS),
        "feature_groups": [
            "cos_sim",
            "mean_diff",
            "std_diff",
            "l2_distance",
        ],
        "features_per_group": 18,
        "full_feature_count": 72,
        "evaluation_cohort": "fixed_full_non_all_nan",
        "result_rows": len(results),
        "full_mean_f1_percent": full_mean,
        "leave_one_out_mean_f1_percent": {
            configuration: float(
                aggregate.loc[
                    aggregate["configuration"] == configuration,
                    "mean_f1_percent",
                ].iloc[0]
            )
            for configuration in LEAVE_ONE_OUT[1:]
        },
        "single_group_mean_f1_percent": {
            configuration: float(
                aggregate.loc[
                    aggregate["configuration"] == configuration,
                    "mean_f1_percent",
                ].iloc[0]
            )
            for configuration in SINGLE_GROUP
        },
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
