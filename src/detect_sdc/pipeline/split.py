"""Dataset split manifest generation shared by all model jobs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ..adapters import load_dataset_adapter
from ..config import load_yaml
from ..dataset_splits import (
    DatasetSplitManifest,
    create_split_manifest,
    load_split_manifest,
    write_split_manifest,
)


def run_split_job(
    config_path: str | Path,
    dataset_name: str,
    *,
    repository_root: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    experiment = load_yaml(config_path)
    datasets = _mapping(experiment.get("datasets"), "datasets")
    if dataset_name not in datasets:
        raise KeyError(f"Unknown dataset: {dataset_name}")

    dataset_config_path = _resolve(root, str(datasets[dataset_name]))
    dataset_config = load_yaml(dataset_config_path)
    sampling = _mapping(dataset_config.get("sampling"), "sampling")
    split = _mapping(dataset_config.get("split"), "split")
    max_samples_value = sampling.get("max_samples")
    max_samples = (
        None if max_samples_value is None else int(max_samples_value)
    )
    manifest_path = _resolve(
        root,
        _required_string(split, "manifest"),
    )
    manifest = create_split_manifest(
        dataset_name,
        load_dataset_adapter(dataset_config_path).iter_samples(
            max_samples=max_samples
        ),
        seed=int(split.get("seed", 42)),
        fit_ratio=float(split.get("fit_ratio", 0.7)),
        calibration_ratio=float(split.get("calibration_ratio", 0.15)),
        test_ratio=float(split.get("test_ratio", 0.15)),
    )
    write_split_manifest(manifest, manifest_path, overwrite=overwrite)
    summary = {
        "dataset": dataset_name,
        "output": str(manifest_path),
        **{
            key: value
            for key, value in manifest.to_dict().items()
            if key != "assignments"
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def load_configured_split_manifest(
    dataset_config: Mapping[str, Any],
    *,
    repository_root: str | Path,
) -> DatasetSplitManifest:
    split = _mapping(dataset_config.get("split"), "split")
    manifest_path = _resolve(
        Path(repository_root).resolve(),
        _required_string(split, "manifest"),
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Dataset split manifest does not exist. Run "
            f"`detect-sdc split --dataset {dataset_config.get('name')}`: "
            f"{manifest_path}"
        )
    manifest = load_split_manifest(manifest_path)
    expected_name = str(dataset_config.get("name", ""))
    if manifest.dataset != expected_name:
        raise ValueError(
            f"Split manifest dataset is {manifest.dataset}, "
            f"expected {expected_name}"
        )
    return manifest


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _required_string(mapping: Mapping[str, Any], key: str) -> str:
    value = str(mapping.get(key, "")).strip()
    if not value:
        raise ValueError(f"Configuration requires {key}")
    return value


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path.resolve()
