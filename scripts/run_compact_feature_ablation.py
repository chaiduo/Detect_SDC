#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from detect_sdc.features.jobs import load_feature_job
from run_feature_group_ablation import (
    DATASETS,
    FEATURE_GROUPS,
    FULL_SPEC,
    METRIC_KEYS,
    MODELS,
    _uid_digest,
    _write_json,
    _write_summary,
    detector_config,
    expected_full_metrics,
    fixed_validation_cohort,
    run_configuration,
)


STATISTICS = ("mean", "max", "min")


@dataclass(frozen=True)
class Configuration:
    metric: str
    statistic: str

    @property
    def name(self) -> str:
        return f"{self.metric}_{self.statistic}"

    @property
    def feature_columns(self) -> tuple[str, ...]:
        prefix = f"{self.metric}_{self.statistic}_"
        return tuple(
            column
            for column in FULL_SPEC.feature_columns
            if column.startswith(prefix)
        )


CONFIGURATIONS = tuple(
    Configuration(metric, statistic)
    for metric in FEATURE_GROUPS
    for statistic in STATISTICS
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Run the 4 metric x 3 aggregation-statistic compact feature "
            "ablation."
        )
    )
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


def validate_configurations() -> None:
    full = set(FULL_SPEC.feature_columns)
    selected = []
    for configuration in CONFIGURATIONS:
        columns = configuration.feature_columns
        if len(columns) != 6:
            raise AssertionError(
                f"{configuration.name} should contain 6 features, "
                f"got {len(columns)}"
            )
        selected.extend(columns)
    if len(selected) != len(set(selected)):
        raise AssertionError("Compact configurations overlap")
    if set(selected) != full:
        missing = sorted(full - set(selected))
        extra = sorted(set(selected) - full)
        raise AssertionError(
            f"Compact configurations do not partition Full: "
            f"missing={missing}, extra={extra}"
        )


def run_dataset(
    *,
    root: Path,
    config_path: Path,
    model: str,
    dataset: str,
    output_base: Path,
    overwrite: bool,
) -> list[dict[str, Any]]:
    model_key, _ = MODELS[model]
    feature_job = load_feature_job(
        config_path,
        f"{model_key}_{DATASETS[dataset]}",
        repository_root=root,
    )
    source_fit = pd.read_csv(feature_job.fit_output)
    source_calibration = pd.read_csv(feature_job.calibration_output)
    source_test = pd.read_csv(feature_job.test_output)
    fixed_test = fixed_validation_cohort(source_test)
    target = pd.to_numeric(
        fixed_test["significant_sdc_target"],
        errors="raise",
    ).astype(int)
    full_metrics = expected_full_metrics(feature_job)
    full_f1 = float(full_metrics["f1"])
    xgboost_config = detector_config(config_path, model_key)
    dataset_root = output_base / dataset
    _write_json(
        dataset_root / "experiment_metadata.json",
        {
            "model": model,
            "dataset": dataset,
            "definition": (
                "Each configuration retains one discrepancy metric and one "
                "step-aggregation statistic across all six monitored pairs"
            ),
            "metrics": FEATURE_GROUPS,
            "statistics": STATISTICS,
            "configuration_count": len(CONFIGURATIONS),
            "features_per_configuration": 6,
            "full_feature_count": len(FULL_SPEC.feature_columns),
            "full_metrics": {
                key: full_metrics[key] for key in METRIC_KEYS
            },
            "evaluation_cohort": "fixed_full_non_all_nan",
            "test_rows_before_filter": len(source_test),
            "test_rows": len(fixed_test),
            "positive_samples": int(target.sum()),
            "negative_samples": int(len(target) - target.sum()),
            "sample_uid_sha256": _uid_digest(fixed_test["sample_uid"]),
        },
    )

    rows: list[dict[str, Any]] = []
    for configuration in CONFIGURATIONS:
        print(
            f"[compact-feature-ablation] {model} {dataset} "
            f"{configuration.name} features=6",
            flush=True,
        )
        metrics = run_configuration(
            source_fit=source_fit,
            source_calibration=source_calibration,
            fixed_test=fixed_test,
            configuration=configuration,
            destination=dataset_root / configuration.name,
            config=xgboost_config,
            overwrite=overwrite,
        )
        rows.append(
            {
                "model": model,
                "dataset": dataset,
                "configuration": configuration.name,
                "metric": configuration.metric,
                "statistic": configuration.statistic,
                "feature_count": len(configuration.feature_columns),
                "evaluation_cohort": "fixed_full_non_all_nan",
                "test_rows": len(fixed_test),
                "positive_samples": int(target.sum()),
                "negative_samples": int(len(target) - target.sum()),
                **{key: metrics[key] for key in METRIC_KEYS},
                "full_f1": full_f1,
                "delta_f1_pp": 100.0 * (float(metrics["f1"]) - full_f1),
            }
        )
        _write_summary(dataset_root / "summary.csv", rows)
    return rows


def main() -> int:
    args = parse_args()
    validate_configurations()
    root = args.repository_root.resolve()
    config_path = args.config.resolve()
    models = tuple(MODELS) if args.model == "all" else (args.model,)
    datasets = tuple(DATASETS) if args.dataset == "all" else (args.dataset,)
    all_rows = []
    for model in models:
        model_key, _ = MODELS[model]
        output_base = (
            args.output_root.resolve()
            if args.output_root is not None
            else root / "artifacts/iclr_v2/ablations/compact_features" / model_key
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
