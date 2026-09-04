#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from detect_sdc.config import load_yaml
from detect_sdc.detector.xgboost import (
    XGBoostConfig,
    all_feature_nan_mask,
    run_calibrated_xgboost,
)
from detect_sdc.features.extraction import FeatureSpec
from detect_sdc.features.jobs import load_feature_job


DATASETS = {
    "EarthVQA": "earthvqa",
    "LingoQA": "lingoqa",
    "VQAv2": "vqav2",
}
MODELS = {
    "Qwen2.5-VL-7B": ("qwen25_vl", "Qwen2.5-VL-7B"),
    "InternVL3-8B": ("internvl3", "InternVL3-8B"),
    "LLaVA-1.5-7B": ("llava15", "llava-v1.5-7B"),
}
FEATURE_GROUPS = (
    "cos_sim",
    "mean_diff",
    "std_diff",
    "l2_distance",
)
FULL_PAIRS = (
    (6, 7),
    (22, 23),
    (23, 24),
    (24, 25),
    (25, 26),
    (26, 27),
)
FULL_SPEC = FeatureSpec(
    selected_layer_pairs=FULL_PAIRS,
    distance_pairs=FULL_PAIRS,
    last_k_steps=2,
    finite_only=True,
    step_window="prefix",
)
META_COLUMNS = (
    "orig_id",
    "semantic_group_id",
    "split",
    "sample_uid",
    "injected",
    "run_index",
    "is_sdc",
    "fault_component",
    "fault_layer_index",
    "fault_op_type",
    "fault_bit_categories",
    "total_steps",
    "last_k_steps",
    "num_steps_used",
)
TARGET_COLUMNS = (
    "significance",
    "label",
    "significant_sdc_target",
)
METRIC_KEYS = ("precision", "recall", "f1", "tp", "fp", "fn", "tn")


@dataclass(frozen=True)
class Configuration:
    name: str
    ablation_type: str
    included_groups: tuple[str, ...]

    @property
    def removed_group(self) -> str:
        removed = [
            group for group in FEATURE_GROUPS if group not in self.included_groups
        ]
        return removed[0] if self.ablation_type == "leave_one_out" else ""

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return tuple(
            column
            for column in FULL_SPEC.feature_columns
            if any(
                column.startswith(f"{group}_")
                for group in self.included_groups
            )
        )


CONFIGURATIONS = (
    Configuration("full", "full", FEATURE_GROUPS),
    *(
        Configuration(
            f"without_{removed}",
            "leave_one_out",
            tuple(group for group in FEATURE_GROUPS if group != removed),
        )
        for removed in FEATURE_GROUPS
    ),
    *(
        Configuration(f"only_{group}", "single_group", (group,))
        for group in FEATURE_GROUPS
    ),
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run detector feature-group ablations."
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


def detector_config(config_path: Path, model_key: str) -> XGBoostConfig:
    config = load_yaml(config_path)
    detector = _mapping(config.get("detector"), "detector")
    xgboost = _mapping(detector.get("xgboost"), "detector.xgboost")
    values = dict(_mapping(xgboost.get("common"), "xgboost.common"))
    by_model = _mapping(xgboost.get("by_model"), "xgboost.by_model")
    values.update(_mapping(by_model.get(model_key), model_key))
    return XGBoostConfig.from_mapping(values)


def validate_configurations() -> None:
    full = set(FULL_SPEC.feature_columns)
    grouped: dict[str, set[str]] = {}
    for group in FEATURE_GROUPS:
        columns = {
            column
            for column in FULL_SPEC.feature_columns
            if column.startswith(f"{group}_")
        }
        if len(columns) != len(FULL_PAIRS) * 3:
            raise AssertionError(
                f"{group} should contain 18 features, got {len(columns)}"
            )
        grouped[group] = columns
    if set().union(*grouped.values()) != full:
        raise AssertionError("Feature groups do not cover the full feature set")
    if sum(len(columns) for columns in grouped.values()) != len(full):
        raise AssertionError("Feature groups overlap")
    for configuration in CONFIGURATIONS:
        expected = len(configuration.included_groups) * len(FULL_PAIRS) * 3
        if len(configuration.feature_columns) != expected:
            raise AssertionError(
                f"{configuration.name} should contain {expected} features"
            )


def fixed_validation_cohort(valid: pd.DataFrame) -> pd.DataFrame:
    full_columns = list(FULL_SPEC.feature_columns)
    missing = sorted(set(full_columns) - set(valid))
    if missing:
        raise ValueError(f"Validation data is missing features: {missing}")
    return valid.loc[~all_feature_nan_mask(valid, full_columns)].copy()


def materialize(
    *,
    source_fit: pd.DataFrame,
    source_calibration: pd.DataFrame,
    fixed_test: pd.DataFrame,
    configuration: Configuration,
    destination: Path,
    overwrite: bool,
) -> tuple[Path, Path, Path]:
    fit_path = destination / "train_data/fit.csv"
    calibration_path = destination / "train_data/calibration.csv"
    test_path = destination / "train_data/test_fixed_cohort.csv"
    if (
        fit_path.is_file()
        and calibration_path.is_file()
        and test_path.is_file()
        and not overwrite
    ):
        return fit_path, calibration_path, test_path

    columns = [
        *META_COLUMNS,
        *configuration.feature_columns,
        *TARGET_COLUMNS,
    ]
    missing = sorted(
        (set(columns) - set(source_fit))
        | (set(columns) - set(source_calibration))
        | (set(columns) - set(fixed_test))
    )
    if missing:
        raise ValueError(f"{configuration.name} missing columns: {missing}")
    _write_csv(source_fit.loc[:, columns], fit_path)
    _write_csv(source_calibration.loc[:, columns], calibration_path)
    _write_csv(fixed_test.loc[:, columns], test_path)
    return fit_path, calibration_path, test_path


def expected_full_metrics(feature_job: Any) -> Mapping[str, Any]:
    path = (
        feature_job.fit_output.parent.parent
        / "output/metrics_summary.json"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Main detector summary does not exist: {path}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    if tuple(summary["feature_columns"]) != FULL_SPEC.feature_columns:
        raise AssertionError(
            f"Main detector feature order differs from Full: {path}"
        )
    return summary["metrics"]["test_finite"]["target_significant_sdc"]


def verify_full_reproduction(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    for key in METRIC_KEYS:
        if not np.isclose(
            float(actual[key]),
            float(expected[key]),
            rtol=0.0,
            atol=1e-15,
        ):
            raise AssertionError(
                f"Full configuration does not reproduce {key}: "
                f"{actual[key]} != {expected[key]}"
            )


def run_configuration(
    *,
    source_fit: pd.DataFrame,
    source_calibration: pd.DataFrame,
    fixed_test: pd.DataFrame,
    configuration: Configuration,
    destination: Path,
    config: XGBoostConfig,
    overwrite: bool,
) -> Mapping[str, Any]:
    fit_path, calibration_path, test_path = materialize(
        source_fit=source_fit,
        source_calibration=source_calibration,
        fixed_test=fixed_test,
        configuration=configuration,
        destination=destination,
        overwrite=overwrite,
    )
    summary_path = destination / "output/metrics_summary.json"
    if summary_path.is_file() and not overwrite:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = run_calibrated_xgboost(
            fit_path,
            calibration_path,
            test_path,
            destination / "output",
            group_column="semantic_group_id",
            config=config,
        )
    if tuple(summary["feature_columns"]) != configuration.feature_columns:
        raise AssertionError(
            f"Unexpected feature order for {configuration.name}"
        )
    return summary["metrics"]["test_full"]["target_significant_sdc"]


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
    expected = expected_full_metrics(feature_job)
    xgboost_config = detector_config(config_path, model_key)
    dataset_root = output_base / dataset
    _write_json(
        dataset_root / "fixed_cohort.json",
        {
            "model": model,
            "dataset": dataset,
            "definition": (
                "Main-experiment validation rows with at least one finite "
                "feature under the complete four-group configuration"
            ),
            "test_rows_before_filter": len(source_test),
            "test_rows": len(fixed_test),
            "positive_samples": int(target.sum()),
            "negative_samples": int(len(target) - target.sum()),
            "sample_uid_sha256": _uid_digest(fixed_test["sample_uid"]),
            "feature_groups": FEATURE_GROUPS,
            "full_feature_count": len(FULL_SPEC.feature_columns),
        },
    )

    rows: list[dict[str, Any]] = []
    full_f1: float | None = None
    for configuration in CONFIGURATIONS:
        print(
            f"[feature-ablation] {model} {dataset} {configuration.name} "
            f"features={len(configuration.feature_columns)}",
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
        if configuration.name == "full":
            verify_full_reproduction(metrics, expected)
            full_f1 = float(metrics["f1"])
        if full_f1 is None:
            raise AssertionError("Full configuration must run first")
        rows.append(
            {
                "model": model,
                "dataset": dataset,
                "configuration": configuration.name,
                "ablation_type": configuration.ablation_type,
                "included_groups": json.dumps(
                    configuration.included_groups
                ),
                "removed_group": configuration.removed_group,
                "feature_count": len(configuration.feature_columns),
                "evaluation_cohort": "fixed_full_non_all_nan",
                "test_rows": len(fixed_test),
                "positive_samples": int(target.sum()),
                "negative_samples": int(len(target) - target.sum()),
                **{key: metrics[key] for key in METRIC_KEYS},
                "delta_f1_pp": 100.0 * (float(metrics["f1"]) - full_f1),
            }
        )
        _write_summary(dataset_root / "summary.csv", rows)
    return rows


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _uid_digest(values: pd.Series) -> str:
    payload = "\n".join(sorted(values.astype(str))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a non-empty mapping")
    return value


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
            else root / "artifacts/iclr_v2/ablations/feature_groups" / model_key
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
