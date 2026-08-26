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
from detect_sdc.detector.xgboost import XGBoostConfig, run_xgboost
from detect_sdc.features.extraction import FeatureSpec
from detect_sdc.features.jobs import FeatureJob, execute_feature_job


DATASET_NAMES = {
    "EarthVQA": "earthvqa",
    "LingoQA": "lingoqa",
    "VQAv2": "vqav2",
}
MODEL_DIRECTORY = "Qwen2.5-VL-7B"
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


@dataclass(frozen=True)
class PairConfiguration:
    name: str
    pairs: tuple[tuple[int, int], ...]

    @property
    def feature_spec(self) -> FeatureSpec:
        return FeatureSpec(
            selected_layer_pairs=self.pairs,
            distance_pairs=self.pairs,
            last_k_steps=50,
            finite_only=True,
        )


PAIR_CONFIGURATIONS = (
    PairConfiguration(
        "current",
        ((6, 7), (22, 23), (23, 24), (24, 25), (25, 26), (26, 27)),
    ),
    PairConfiguration(
        "spread_6",
        ((0, 1), (5, 6), (10, 11), (15, 16), (20, 21), (26, 27)),
    ),
    PairConfiguration(
        "late_dense",
        tuple((layer, layer + 1) for layer in range(20, 27)),
    ),
    PairConfiguration(
        "even_adjacent",
        tuple((layer, layer + 1) for layer in range(0, 27, 2)),
    ),
    PairConfiguration(
        "mid_late_dense",
        tuple((layer, layer + 1) for layer in range(12, 27)),
    ),
    PairConfiguration(
        "all_adjacent",
        tuple((layer, layer + 1) for layer in range(27)),
    ),
)
ALL_ADJACENT = PAIR_CONFIGURATIONS[-1]


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Run the Qwen2.5-VL-7B layer-pair ablation with the current "
            "project telemetry and canonical detector training pipeline."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=tuple(DATASET_NAMES),
        required=True,
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=repository_root,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=repository_root / "configs/experiments/current.yaml",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            repository_root
            / MODEL_DIRECTORY
            / "pair_ablation_project_20260813"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild existing feature CSVs and detector outputs.",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Extract and materialize all pair configurations without training.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Development-only input record limit.",
    )
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    return args


def detector_config(config_path: Path) -> XGBoostConfig:
    config = load_yaml(config_path)
    detector = _mapping(config.get("detector"), "detector")
    xgboost = _mapping(detector.get("xgboost"), "detector.xgboost")
    values = dict(_mapping(xgboost.get("common"), "xgboost.common"))
    by_model = _mapping(xgboost.get("by_model"), "xgboost.by_model")
    values.update(_mapping(by_model.get("qwen25_vl"), "qwen25_vl"))
    return XGBoostConfig.from_mapping(values)


def extract_all_pairs(
    *,
    repository_root: Path,
    dataset: str,
    dataset_key: str,
    dataset_root: Path,
    max_samples: int | None,
    overwrite: bool,
) -> tuple[Path, Path]:
    train_path = dataset_root / "all_pairs" / "train.csv"
    valid_path = dataset_root / "all_pairs" / "valid.csv"
    if train_path.is_file() and valid_path.is_file() and not overwrite:
        print(
            f"[ablation] dataset={dataset} reuse all-pair features",
            flush=True,
        )
        return train_path, valid_path

    labels_path = (
        repository_root
        / MODEL_DIRECTORY
        / dataset
        / "json"
        / "labels.jsonl"
    )
    job = FeatureJob(
        name=f"qwen25_vl_{dataset_key}_all_adjacent_ablation",
        model="qwen25_vl",
        dataset=dataset_key,
        uid_namespace=f"qwen25_vl_{dataset_key}",
        input_path=labels_path,
        train_output=train_path,
        valid_output=valid_path,
        group_column="orig_id",
        valid_ratio=0.15,
        random_state=42,
        spec=ALL_ADJACENT.feature_spec,
    )
    summary = execute_feature_job(job, max_samples=max_samples)
    _atomic_write_json(dataset_root / "all_pairs" / "extraction_summary.json", summary)
    return train_path, valid_path


def materialize_configuration(
    *,
    source_train: Path,
    source_valid: Path,
    destination: Path,
    pair_config: PairConfiguration,
    overwrite: bool,
) -> tuple[Path, Path]:
    train_path = destination / "train_data" / "train.csv"
    valid_path = destination / "train_data" / "valid.csv"
    if train_path.is_file() and valid_path.is_file() and not overwrite:
        return train_path, valid_path

    train = pd.read_csv(source_train)
    valid = pd.read_csv(source_valid)
    columns = [
        *META_COLUMNS,
        *pair_config.feature_spec.feature_columns,
        *TARGET_COLUMNS,
    ]
    missing = sorted((set(columns) - set(train)) | (set(columns) - set(valid)))
    if missing:
        raise ValueError(
            f"{pair_config.name} is missing all-pair columns: {missing}"
        )
    _atomic_write_csv(train.loc[:, columns], train_path)
    _atomic_write_csv(valid.loc[:, columns], valid_path)
    return train_path, valid_path


def train_configuration(
    *,
    train_path: Path,
    valid_path: Path,
    destination: Path,
    config: XGBoostConfig,
    overwrite: bool,
) -> Mapping[str, Any]:
    summary_path = destination / "output" / "metrics_summary.json"
    if summary_path.is_file() and not overwrite:
        print(f"[ablation] reuse detector result: {summary_path}", flush=True)
        return json.loads(summary_path.read_text(encoding="utf-8"))
    return run_xgboost(
        train_path,
        valid_path,
        destination / "output",
        group_column="orig_id",
        config=config,
    )


def summarize_result(
    *,
    dataset: str,
    pair_config: PairConfiguration,
    summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for split_name, split_label in (
        ("valid_full_metrics", "full"),
        ("valid_non_all_nan_metrics", "non_all_feature_nan"),
    ):
        metrics = _mapping(summary.get(split_name), split_name)
        target = _mapping(
            metrics.get("target_significant_sdc"),
            f"{split_name}.target_significant_sdc",
        )
        rows.append(
            {
                "dataset": dataset,
                "pair_config": pair_config.name,
                "pairs": json.dumps(pair_config.pairs),
                "pair_count": len(pair_config.pairs),
                "feature_count": len(pair_config.feature_spec.feature_columns),
                "evaluation_slice": split_label,
                "precision": target["precision"],
                "recall": target["recall"],
                "f1": target["f1"],
                "tp": target["tp"],
                "fp": target["fp"],
                "fn": target["fn"],
                "tn": target["tn"],
            }
        )
    return rows


def write_dataset_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "dataset",
        "pair_config",
        "pairs",
        "pair_count",
        "feature_count",
        "evaluation_slice",
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
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
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


def main() -> int:
    args = parse_args()
    repository_root = args.repository_root.resolve()
    config_path = args.config.resolve()
    output_root = args.output_root.resolve()
    dataset = args.dataset
    dataset_key = DATASET_NAMES[dataset]
    dataset_root = output_root / dataset

    source_train, source_valid = extract_all_pairs(
        repository_root=repository_root,
        dataset=dataset,
        dataset_key=dataset_key,
        dataset_root=dataset_root,
        max_samples=args.max_samples,
        overwrite=args.overwrite,
    )

    feature_paths = {}
    for pair_config in PAIR_CONFIGURATIONS:
        feature_paths[pair_config.name] = materialize_configuration(
            source_train=source_train,
            source_valid=source_valid,
            destination=dataset_root / pair_config.name,
            pair_config=pair_config,
            overwrite=args.overwrite,
        )
        print(
            f"[ablation] dataset={dataset} materialized={pair_config.name}",
            flush=True,
        )

    if args.extract_only:
        return 0

    xgboost_config = detector_config(config_path)
    summary_rows: list[dict[str, Any]] = []
    for pair_config in PAIR_CONFIGURATIONS:
        print(
            f"[ablation] dataset={dataset} train={pair_config.name} "
            f"features={len(pair_config.feature_spec.feature_columns)}",
            flush=True,
        )
        train_path, valid_path = feature_paths[pair_config.name]
        summary = train_configuration(
            train_path=train_path,
            valid_path=valid_path,
            destination=dataset_root / pair_config.name,
            config=xgboost_config,
            overwrite=args.overwrite,
        )
        summary_rows.extend(
            summarize_result(
                dataset=dataset,
                pair_config=pair_config,
                summary=summary,
            )
        )
        write_dataset_summary(dataset_root / "summary.csv", summary_rows)

    print(
        f"[ablation] complete dataset={dataset} "
        f"summary={dataset_root / 'summary.csv'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
