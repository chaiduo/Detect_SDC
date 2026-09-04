#!/usr/bin/env python3

"""Evaluate 48D statistic-removal ablations under the v2 split protocol."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pandas as pd

from detect_sdc.config import load_yaml
from detect_sdc.detector.xgboost import (
    XGBoostConfig,
    add_significant_sdc_target,
    all_feature_nan_mask,
    binary_metrics,
    calibrate_threshold_max_f1,
    get_feature_columns,
    prepare_features,
    train_binary_model,
)
from detect_sdc.features.jobs import load_feature_job
from detect_sdc.splitting import split_by_group


PAIRS = ((6, 7), (22, 23), (25, 26), (26, 27))
STATISTICS = ("mean", "max", "min")
CONFIGURATIONS = {
    "48D_full": STATISTICS,
    "48D_without_all_min": ("mean", "max"),
    "48D_without_all_max": ("mean", "min"),
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run 48D statistic-removal ablations for one v2 feature job."
    )
    parser.add_argument("--job", default="llava15_lingoqa")
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configs/experiments/current.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "analysis/llava15_lingoqa_48d_statistics_ablation",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=8)
    return parser.parse_args()


def detector_config(
    config: dict[str, Any],
    model: str,
    *,
    seed: int,
    n_jobs: int,
) -> XGBoostConfig:
    values = dict(config["detector"]["xgboost"]["common"])
    values.update(config["detector"]["xgboost"]["by_model"].get(model, {}))
    return replace(
        XGBoostConfig.from_mapping(values),
        random_state=seed,
        n_jobs=n_jobs,
        verbose=False,
    )


def selected_feature_columns(
    feature_columns: list[str],
    statistics: tuple[str, ...],
) -> list[str]:
    pair_suffixes = tuple(f"_p{left}_{right}" for left, right in PAIRS)
    selected = [
        column
        for column in feature_columns
        if column.endswith(pair_suffixes)
        and any(f"_{statistic}_p" in column for statistic in statistics)
    ]
    expected = len(PAIRS) * 4 * len(statistics)
    if len(selected) != expected:
        raise ValueError(
            f"Expected {expected} features for statistics={statistics}, "
            f"found {len(selected)}"
        )
    return selected


def target_metrics(
    model: Any,
    frame: pd.DataFrame,
    feature_columns: list[str],
    threshold: float,
) -> dict[str, Any]:
    probability = model.predict_proba(prepare_features(frame, feature_columns))
    prediction = (probability[:, 1] > threshold).astype(int)
    metrics = binary_metrics(
        frame["significant_sdc_target"].astype(int),
        prediction,
        probability,
    )["target_significant_sdc"]
    return {
        key: metrics[key]
        for key in ("precision", "recall", "f1", "false_positive_rate", "tp", "fp", "fn", "tn")
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# 48D Statistic-Removal Ablation",
        "",
        "## Protocol",
        "",
        "- Job: LLaVA-1.5-7B / LingoQA.",
        "- Data: current v2 random-bit injection campaign.",
        "- Outer split: existing Fit / Calibration / Final Test semantic-group split.",
        "- Training: Fit with a semantic-group-aware internal holdout for early stopping.",
        "- Threshold: maximum Significant-SDC F1 on Calibration; then frozen.",
        "- Full: all Final Test rows.",
        "- Finite: Final Test rows whose canonical 72D vector is not all NaN.",
        "- The three configurations use identical rows, group split, XGBoost settings, and seed.",
        "",
        "## Results",
        "",
        "| Configuration | Features | Calibration threshold | Full P | Full R | Full F1 | Full FPR | Finite P | Finite R | Finite F1 | Finite FPR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in summary["results"]:
        full = result["full"]
        finite = result["finite"]
        lines.append(
            f"| {result['configuration']} | {result['feature_count']} | "
            f"{result['calibration']['threshold']:.6f} | "
            f"{full['precision']:.2%} | {full['recall']:.2%} | {full['f1']:.2%} | "
            f"{full['false_positive_rate']:.3%} | "
            f"{finite['precision']:.2%} | {finite['recall']:.2%} | {finite['f1']:.2%} | "
            f"{finite['false_positive_rate']:.3%} |"
        )
    lines.extend(
        [
            "",
            "## Configuration Definitions",
            "",
            "- `48D_full`: mean, max, and min for all four metrics across four layer pairs.",
            "- `48D_without_all_min`: remove the 16 min features, retaining mean and max (32D).",
            "- `48D_without_all_max`: remove the 16 max features, retaining mean and min (32D).",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = args.config.resolve()
    config = load_yaml(config_path)
    job = load_feature_job(config_path, args.job, repository_root=root)
    frames = {
        split: add_significant_sdc_target(pd.read_csv(path))
        for split, path in (
            ("fit", job.fit_output),
            ("calibration", job.calibration_output),
            ("test", job.test_output),
        )
    }
    canonical_features = get_feature_columns(frames["fit"])
    configuration = detector_config(
        config,
        job.model,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )
    grouped = split_by_group(
        frames["fit"],
        group_column=job.group_column,
        holdout_ratio=configuration.test_ratio,
        random_state=configuration.random_state,
    )
    finite = ~all_feature_nan_mask(frames["test"], canonical_features)
    results: list[dict[str, Any]] = []

    for name, statistics in CONFIGURATIONS.items():
        feature_columns = selected_feature_columns(canonical_features, statistics)
        print(f"[{args.job}] {name}: {len(feature_columns)} features", flush=True)
        model, training = train_binary_model(
            prepare_features(grouped.train, feature_columns),
            grouped.train["significant_sdc_target"].astype(int),
            prepare_features(grouped.holdout, feature_columns),
            grouped.holdout["significant_sdc_target"].astype(int),
            config=configuration,
        )
        calibration_probability = model.predict_proba(
            prepare_features(frames["calibration"], feature_columns)
        )[:, 1]
        calibration = calibrate_threshold_max_f1(
            calibration_probability,
            frames["calibration"]["significant_sdc_target"].astype(int),
        )
        threshold = float(calibration["threshold"])
        results.append(
            {
                "configuration": name,
                "statistics": list(statistics),
                "feature_count": len(feature_columns),
                "feature_columns": feature_columns,
                "fit_internal_holdout": {
                    "rows": len(grouped.holdout),
                    "groups": grouped.summary.holdout_groups,
                    "group_overlap": grouped.summary.group_overlap,
                },
                "calibration": calibration,
                "training": training,
                "full": target_metrics(
                    model, frames["test"], feature_columns, threshold
                ),
                "finite": target_metrics(
                    model, frames["test"].loc[finite], feature_columns, threshold
                ),
            }
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "protocol": (
            "current v2 Fit/Calibration/Final Test split; semantic-group "
            "internal Fit holdout; Calibration maximum-F1 threshold"
        ),
        "job": args.job,
        "seed": args.seed,
        "canonical_finite_definition": (
            "exclude rows whose complete canonical 72D vector is all NaN"
        ),
        "test_rows": len(frames["test"]),
        "finite_test_rows": int(finite.sum()),
        "detector_config": asdict(configuration),
        "results": results,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.json_normalize(results, sep=".").to_csv(
        output_dir / "results.csv",
        index=False,
    )
    write_report(output_dir / "report.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
