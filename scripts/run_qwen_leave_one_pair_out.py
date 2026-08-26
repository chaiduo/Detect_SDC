#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from detect_sdc.config import load_yaml
from detect_sdc.detector.xgboost import (
    XGBoostConfig,
    all_feature_nan_mask,
    run_xgboost,
)
from detect_sdc.features.extraction import FeatureSpec
from detect_sdc.features.jobs import load_feature_job


DATASET_NAMES = {
    "EarthVQA": "earthvqa",
    "LingoQA": "lingoqa",
    "VQAv2": "vqav2",
}
MODEL_NAMES = {
    "Qwen2.5-VL-7B": ("qwen25_vl", "Qwen2.5-VL-7B"),
    "InternVL3-8B": ("internvl3", "InternVL3-8B"),
    "LLaVA-1.5-7B": ("llava15", "llava-v1.5-7B"),
}
META_COLUMNS = (
    "orig_id",
    "sample_uid",
    "total_steps",
    "last_k_steps",
    "num_steps_used",
)
TARGET_COLUMNS = (
    "significance",
    "label",
    "significant_sdc_target",
)


FULL_PAIRS = (
    (6, 7),
    (22, 23),
    (23, 24),
    (24, 25),
    (25, 26),
    (26, 27),
)


@dataclass(frozen=True)
class Configuration:
    name: str
    pairs: tuple[tuple[int, int], ...]

    @property
    def spec(self) -> FeatureSpec:
        return FeatureSpec(
            selected_layer_pairs=self.pairs,
            distance_pairs=self.pairs,
            last_k_steps=50,
            finite_only=True,
        )


CONFIGURATIONS = (
    Configuration("full", FULL_PAIRS),
    *(
        Configuration(
            f"without_p{removed[0]}_{removed[1]}",
            tuple(pair for pair in FULL_PAIRS if pair != removed),
        )
        for removed in FULL_PAIRS
    ),
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run leave-one-layer-pair-out ablations."
    )
    parser.add_argument(
        "--model",
        choices=tuple(MODEL_NAMES),
        default="Qwen2.5-VL-7B",
    )
    parser.add_argument("--dataset", choices=tuple(DATASET_NAMES), required=True)
    parser.add_argument("--repository-root", type=Path, default=root)
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configs/experiments/current.yaml",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def feature_columns(configuration: Configuration) -> list[str]:
    return list(configuration.spec.feature_columns)


def detector_config(
    config_path: Path,
    model_key: str,
) -> XGBoostConfig:
    config = load_yaml(config_path)
    detector = _mapping(config.get("detector"), "detector")
    xgboost = _mapping(detector.get("xgboost"), "detector.xgboost")
    values = dict(_mapping(xgboost.get("common"), "xgboost.common"))
    by_model = _mapping(xgboost.get("by_model"), "xgboost.by_model")
    values.update(_mapping(by_model.get(model_key), model_key))
    return XGBoostConfig.from_mapping(values)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def materialize(
    *,
    source_train: pd.DataFrame,
    fixed_valid: pd.DataFrame,
    configuration: Configuration,
    destination: Path,
    overwrite: bool,
) -> tuple[Path, Path]:
    train_path = destination / "train_data/train.csv"
    valid_path = destination / "train_data/valid_fixed_cohort.csv"
    if train_path.is_file() and valid_path.is_file() and not overwrite:
        return train_path, valid_path

    columns = [
        *META_COLUMNS,
        *feature_columns(configuration),
        *TARGET_COLUMNS,
    ]
    missing = sorted(
        (set(columns) - set(source_train))
        | (set(columns) - set(fixed_valid))
    )
    if missing:
        raise ValueError(f"{configuration.name} missing columns: {missing}")
    write_csv(source_train.loc[:, columns], train_path)
    write_csv(fixed_valid.loc[:, columns], valid_path)
    return train_path, valid_path


def fixed_validation_cohort(valid: pd.DataFrame) -> pd.DataFrame:
    full_columns = feature_columns(CONFIGURATIONS[0])
    missing = sorted(set(full_columns) - set(valid))
    if missing:
        raise ValueError(f"all-pairs validation is missing columns: {missing}")
    mask = ~all_feature_nan_mask(valid, full_columns)
    return valid.loc[mask].copy()


def target_metrics(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    metrics = summary["valid_full_metrics"]
    return metrics["target_significant_sdc"]


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "dataset",
        "configuration",
        "removed_pair",
        "pairs",
        "pair_count",
        "feature_count",
        "evaluation_cohort",
        "validation_rows",
        "positive_samples",
        "negative_samples",
        "precision",
        "recall",
        "f1",
        "tp",
        "fp",
        "fn",
        "tn",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def removed_pair(configuration: Configuration) -> str:
    removed = [pair for pair in FULL_PAIRS if pair not in configuration.pairs]
    return "" if not removed else f"({removed[0][0]},{removed[0][1]})"


def main() -> int:
    args = parse_args()
    repository_root = args.repository_root.resolve()
    config_path = args.config.resolve()
    model_name = args.model
    model_key, model_directory = MODEL_NAMES[model_name]
    dataset = args.dataset
    dataset_key = DATASET_NAMES[dataset]
    output_base = (
        args.output_root.resolve()
        if args.output_root is not None
        else (
            repository_root
            / model_directory
            / "pair_ablation_leave_one_out_20260813"
        )
    )
    output_root = output_base / dataset
    feature_job = load_feature_job(
        config_path,
        f"{model_key}_{dataset_key}",
        repository_root=repository_root,
    )
    source_train = pd.read_csv(feature_job.train_output)
    source_valid = pd.read_csv(feature_job.valid_output)
    fixed_valid = fixed_validation_cohort(source_valid)
    config = detector_config(config_path, model_key)

    positive_samples = int(fixed_valid["significant_sdc_target"].sum())
    cohort = {
        "model": model_name,
        "dataset": dataset,
        "definition": (
            "Rows with at least one finite feature under the complete "
            "six-pair configuration"
        ),
        "validation_rows": int(len(fixed_valid)),
        "positive_samples": positive_samples,
        "negative_samples": int(len(fixed_valid) - positive_samples),
        "sample_uid_sha256": _uid_digest(fixed_valid["sample_uid"]),
    }
    _write_json(output_root / "fixed_cohort.json", cohort)

    rows: list[dict[str, Any]] = []
    for configuration in CONFIGURATIONS:
        destination = output_root / configuration.name
        train_path, valid_path = materialize(
            source_train=source_train,
            fixed_valid=fixed_valid,
            configuration=configuration,
            destination=destination,
            overwrite=args.overwrite,
        )
        summary_path = destination / "output/metrics_summary.json"
        if summary_path.is_file() and not args.overwrite:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            summary = run_xgboost(
                train_path,
                valid_path,
                destination / "output",
                group_column="orig_id",
                config=config,
            )
        metrics = target_metrics(summary)
        rows.append(
            {
                "dataset": dataset,
                "configuration": configuration.name,
                "removed_pair": removed_pair(configuration),
                "pairs": json.dumps(configuration.pairs),
                "pair_count": len(configuration.pairs),
                "feature_count": len(feature_columns(configuration)),
                "evaluation_cohort": "fixed_full_six_pair_non_all_nan",
                "validation_rows": len(fixed_valid),
                "positive_samples": positive_samples,
                "negative_samples": len(fixed_valid) - positive_samples,
                **{
                    key: metrics[key]
                    for key in ("precision", "recall", "f1", "tp", "fp", "fn", "tn")
                },
            }
        )
        write_summary(output_root / "summary.csv", rows)
        print(
            f"[leave-one-out] {model_name} {dataset} {configuration.name} "
            f"F1={float(metrics['f1']):.6f}",
            flush=True,
        )
    return 0


def _uid_digest(values: pd.Series) -> str:
    import hashlib

    payload = "\n".join(sorted(values.astype(str))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a non-empty mapping")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
