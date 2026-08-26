#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from detect_sdc.config import load_yaml
from detect_sdc.detector.xgboost import XGBoostConfig, run_xgboost
from detect_sdc.features.extraction import FeatureSpec
from detect_sdc.features.jobs import FeatureJob, execute_feature_job, load_feature_job


DATASETS = {
    "EarthVQA": "earthvqa",
    "LingoQA": "lingoqa",
    "VQAv2": "vqav2",
}
MODELS = {
    "Qwen2.5-VL-7B": {
        "key": "qwen25_vl",
        "directory": "Qwen2.5-VL-7B",
        "layer_count": 28,
    },
    "InternVL3-8B": {
        "key": "internvl3",
        "directory": "InternVL3-8B",
        "layer_count": 28,
    },
    "LLaVA-1.5-7B": {
        "key": "llava15",
        "directory": "llava-v1.5-7B",
        "layer_count": 32,
    },
}
CURRENT_PAIRS = (
    (6, 7),
    (22, 23),
    (23, 24),
    (24, 25),
    (25, 26),
    (26, 27),
)
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
    def spec(self) -> FeatureSpec:
        return FeatureSpec(
            selected_layer_pairs=self.pairs,
            distance_pairs=self.pairs,
            last_k_steps=50,
            finite_only=True,
        )


def pair_configurations(layer_count: int) -> tuple[PairConfiguration, ...]:
    last_source = layer_count - 2
    if last_source < 26:
        raise ValueError("Current configuration requires at least 28 layers")

    spread_sources = tuple(
        math.floor(index * last_source / 5)
        for index in range(6)
    )
    late_sources = tuple(range(last_source - 6, last_source + 1))
    even_sources = tuple(range(0, last_source + 1, 2))
    mid_late_sources = tuple(range(last_source - 14, last_source + 1))
    all_sources = tuple(range(last_source + 1))
    return (
        PairConfiguration("current", CURRENT_PAIRS),
        PairConfiguration(
            "spread_6",
            tuple((source, source + 1) for source in spread_sources),
        ),
        PairConfiguration(
            "late_dense",
            tuple((source, source + 1) for source in late_sources),
        ),
        PairConfiguration(
            "even_adjacent",
            tuple((source, source + 1) for source in even_sources),
        ),
        PairConfiguration(
            "mid_late_dense",
            tuple((source, source + 1) for source in mid_late_sources),
        ),
        PairConfiguration(
            "all_adjacent",
            tuple((source, source + 1) for source in all_sources),
        ),
    )


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run full-network layer-pair design ablations."
    )
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument("--repository-root", type=Path, default=root)
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configs/experiments/current.yaml",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    return args


def detector_config(config_path: Path, model_key: str) -> XGBoostConfig:
    config = load_yaml(config_path)
    detector = _mapping(config.get("detector"), "detector")
    xgboost = _mapping(detector.get("xgboost"), "detector.xgboost")
    values = dict(_mapping(xgboost.get("common"), "xgboost.common"))
    by_model = _mapping(xgboost.get("by_model"), "xgboost.by_model")
    values.update(_mapping(by_model.get(model_key), model_key))
    return XGBoostConfig.from_mapping(values)


def extract_all_pairs(
    *,
    base_job: FeatureJob,
    all_pairs: PairConfiguration,
    dataset_root: Path,
    max_samples: int | None,
    overwrite: bool,
) -> tuple[Path, Path]:
    train_path = dataset_root / "all_pairs_cache/train.csv"
    valid_path = dataset_root / "all_pairs_cache/valid.csv"
    if train_path.is_file() and valid_path.is_file() and not overwrite:
        return train_path, valid_path

    job = FeatureJob(
        name=f"{base_job.name}_all_adjacent_design_ablation",
        model=base_job.model,
        dataset=base_job.dataset,
        uid_namespace=base_job.uid_namespace,
        input_path=base_job.input_path,
        train_output=train_path,
        valid_output=valid_path,
        group_column=base_job.group_column,
        valid_ratio=base_job.valid_ratio,
        random_state=base_job.random_state,
        spec=all_pairs.spec,
    )
    summary = execute_feature_job(job, max_samples=max_samples)
    _write_json(dataset_root / "all_pairs_cache/extraction_summary.json", summary)
    return train_path, valid_path


def materialize(
    *,
    source_train: pd.DataFrame,
    source_valid: pd.DataFrame,
    configuration: PairConfiguration,
    destination: Path,
    overwrite: bool,
) -> tuple[Path, Path]:
    train_path = destination / "train_data/train.csv"
    valid_path = destination / "train_data/valid.csv"
    if train_path.is_file() and valid_path.is_file() and not overwrite:
        return train_path, valid_path

    columns = [
        *META_COLUMNS,
        *configuration.spec.feature_columns,
        *TARGET_COLUMNS,
    ]
    missing = sorted(
        (set(columns) - set(source_train))
        | (set(columns) - set(source_valid))
    )
    if missing:
        raise ValueError(f"{configuration.name} missing columns: {missing}")
    _write_csv(source_train.loc[:, columns], train_path)
    _write_csv(source_valid.loc[:, columns], valid_path)
    return train_path, valid_path


def train(
    *,
    train_path: Path,
    valid_path: Path,
    destination: Path,
    config: XGBoostConfig,
    overwrite: bool,
) -> Mapping[str, Any]:
    summary_path = destination / "output/metrics_summary.json"
    if summary_path.is_file() and not overwrite:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    return run_xgboost(
        train_path,
        valid_path,
        destination / "output",
        group_column="orig_id",
        config=config,
    )


def summary_rows(
    *,
    model: str,
    dataset: str,
    configuration: PairConfiguration,
    summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for split_name, slice_name in (
        ("valid_full_metrics", "full"),
        ("valid_non_all_nan_metrics", "non_all_feature_nan"),
    ):
        target = summary[split_name]["target_significant_sdc"]
        rows.append(
            {
                "model": model,
                "dataset": dataset,
                "pair_config": configuration.name,
                "pairs": json.dumps(configuration.pairs),
                "pair_count": len(configuration.pairs),
                "feature_count": len(configuration.spec.feature_columns),
                "evaluation_slice": slice_name,
                **{
                    key: target[key]
                    for key in ("precision", "recall", "f1", "tp", "fp", "fn", "tn")
                },
            }
        )
    return rows


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "model",
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
        writer = csv.DictWriter(stream, fieldnames=fields)
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


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a non-empty mapping")
    return value


def main() -> int:
    args = parse_args()
    root = args.repository_root.resolve()
    config_path = args.config.resolve()
    model_info = MODELS[args.model]
    model_key = str(model_info["key"])
    model_directory = str(model_info["directory"])
    layer_count = int(model_info["layer_count"])
    dataset_key = DATASETS[args.dataset]
    output_base = (
        args.output_root.resolve()
        if args.output_root is not None
        else root / model_directory / "pair_ablation_design_20260813"
    )
    dataset_root = output_base / args.dataset
    configurations = pair_configurations(layer_count)
    base_job = load_feature_job(
        config_path,
        f"{model_key}_{dataset_key}",
        repository_root=root,
    )
    train_path, valid_path = extract_all_pairs(
        base_job=base_job,
        all_pairs=configurations[-1],
        dataset_root=dataset_root,
        max_samples=args.max_samples,
        overwrite=args.overwrite,
    )
    source_train = pd.read_csv(train_path)
    source_valid = pd.read_csv(valid_path)

    paths: dict[str, tuple[Path, Path]] = {}
    for configuration in configurations:
        paths[configuration.name] = materialize(
            source_train=source_train,
            source_valid=source_valid,
            configuration=configuration,
            destination=dataset_root / configuration.name,
            overwrite=args.overwrite,
        )
    if args.extract_only:
        return 0

    config = detector_config(config_path, model_key)
    rows: list[dict[str, Any]] = []
    for configuration in configurations:
        print(
            f"[design-ablation] {args.model} {args.dataset} "
            f"{configuration.name} pairs={len(configuration.pairs)}",
            flush=True,
        )
        configuration_train, configuration_valid = paths[configuration.name]
        result = train(
            train_path=configuration_train,
            valid_path=configuration_valid,
            destination=dataset_root / configuration.name,
            config=config,
            overwrite=args.overwrite,
        )
        rows.extend(
            summary_rows(
                model=args.model,
                dataset=args.dataset,
                configuration=configuration,
                summary=result,
            )
        )
        write_summary(dataset_root / "summary.csv", rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
