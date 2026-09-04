"""Configuration-driven feature extraction jobs."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ..config import load_yaml
from ..dataset_splits import load_split_manifest
from ..splitting import validate_identity_columns
from .extraction import (
    FeatureSpec,
    LayerPair,
    SampleSkipped,
    extract_feature_row,
    iter_json_samples,
)


@dataclass(frozen=True)
class FeatureJob:
    name: str
    model: str
    dataset: str
    uid_namespace: str
    input_path: Path
    fit_output: Path
    calibration_output: Path
    test_output: Path
    split_manifest: Path
    group_column: str
    spec: FeatureSpec


@dataclass
class FeatureRowCollector:
    rows: list[dict[str, Any]] = field(default_factory=list)
    duplicate_count: int = 0
    _rows_by_uid: dict[str, dict[str, Any]] = field(default_factory=dict)

    def add(self, row: dict[str, Any]) -> bool:
        sample_uid = str(row["sample_uid"])
        previous = self._rows_by_uid.get(sample_uid)
        if previous is not None:
            if not _feature_rows_equal(previous, row):
                raise ValueError(
                    f"Stable sample UID maps to different feature rows: {sample_uid}"
                )
            self.duplicate_count += 1
            return False

        self._rows_by_uid[sample_uid] = row
        self.rows.append(row)
        return True


def load_feature_job(
    config_path: str | Path,
    job_name: str,
    *,
    repository_root: str | Path,
) -> FeatureJob:
    root = Path(repository_root).resolve()
    config = load_yaml(config_path)
    jobs = _mapping(_mapping(config.get("featurization"), "featurization").get("jobs"), "jobs")
    if job_name not in jobs:
        raise KeyError(f"Unknown feature job: {job_name}")
    job = _mapping(jobs[job_name], f"feature job {job_name}")

    model = str(job.get("model", ""))
    dataset = str(job.get("dataset", ""))
    _validate_matrix_pair(config, model, dataset)

    features = _mapping(config.get("features"), "features")
    selected_pairs = _layer_pairs(features.get("layer_pairs"), "features.layer_pairs")
    distance_pairs = _layer_pairs(features.get("distance_pairs"), "features.distance_pairs")

    datasets = _mapping(config.get("datasets"), "datasets")
    dataset_config_path = _resolve_path(root, str(datasets[dataset]))
    dataset_config = load_yaml(dataset_config_path)
    split = _mapping(dataset_config.get("split"), f"{dataset}.split")

    return FeatureJob(
        name=job_name,
        model=model,
        dataset=dataset,
        uid_namespace=str(job.get("uid_namespace", job_name)),
        input_path=_resolve_path(root, _required_string(job, "input")),
        fit_output=_resolve_path(root, _required_string(job, "fit_output")),
        calibration_output=_resolve_path(
            root,
            _required_string(job, "calibration_output"),
        ),
        test_output=_resolve_path(root, _required_string(job, "test_output")),
        split_manifest=_resolve_path(
            root,
            _required_string(split, "manifest"),
        ),
        group_column=str(
            split.get("group_column", "semantic_group_id")
        ),
        spec=FeatureSpec(
            selected_layer_pairs=selected_pairs,
            distance_pairs=distance_pairs,
            last_k_steps=int(features.get("last_k_steps", 50)),
            finite_only=bool(features.get("finite_only", True)),
            step_window=str(features.get("step_window", "suffix")),
        ),
    )


def run_feature_job(
    config_path: str | Path,
    job_name: str,
    *,
    repository_root: str | Path,
    max_samples: int | None = None,
    fit_output: str | Path | None = None,
    calibration_output: str | Path | None = None,
    test_output: str | Path | None = None,
) -> dict[str, Any]:
    job = load_feature_job(config_path, job_name, repository_root=repository_root)
    return execute_feature_job(
        job,
        max_samples=max_samples,
        fit_output=fit_output,
        calibration_output=calibration_output,
        test_output=test_output,
    )


def execute_feature_job(
    job: FeatureJob,
    *,
    max_samples: int | None = None,
    fit_output: str | Path | None = None,
    calibration_output: str | Path | None = None,
    test_output: str | Path | None = None,
) -> dict[str, Any]:
    import pandas as pd

    if not job.input_path.is_file():
        raise FileNotFoundError(f"Feature input does not exist: {job.input_path}")
    manifest = load_split_manifest(job.split_manifest)
    if manifest.dataset != job.dataset:
        raise ValueError(
            f"Split manifest dataset is {manifest.dataset}, "
            f"expected {job.dataset}"
        )

    collector = FeatureRowCollector()
    skipped: Counter[str] = Counter()
    input_samples = 0
    for sample in iter_json_samples(job.input_path, max_samples=max_samples):
        input_samples += 1
        try:
            row = extract_feature_row(
                sample,
                spec=job.spec,
                uid_namespace=job.uid_namespace,
            )
            assignment = manifest.assignment_for_orig_id(
                str(row["orig_id"])
            )
            if (
                str(row["semantic_group_id"])
                != assignment.semantic_group_id
            ):
                raise ValueError(
                    "Feature semantic_group_id differs from split manifest "
                    f"for {row['orig_id']}"
                )
            if row["split"] not in (None, assignment.split):
                raise ValueError(
                    f"Feature split differs from manifest for {row['orig_id']}"
                )
            row["split"] = assignment.split
        except SampleSkipped as error:
            skipped[error.reason] += 1
            continue
        if not collector.add(row):
            skipped["duplicate_sample"] += 1
            continue
        if len(collector.rows) % 1000 == 0:
            print(
                f"[featurize] job={job.name} extracted={len(collector.rows)} "
                f"input_samples={input_samples}",
                flush=True,
            )

    if not collector.rows:
        raise ValueError(f"Feature job produced no rows: {job.name}")

    columns = [
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
        *job.spec.feature_columns,
        "significance",
        "label",
        "significant_sdc_target",
    ]
    frame = pd.DataFrame(collector.rows, columns=columns)
    validate_identity_columns(
        frame,
        group_column=job.group_column,
        sample_uid_column="sample_uid",
    )
    split_frames = {
        split_name: frame.loc[frame["split"] == split_name].copy()
        for split_name in ("fit", "calibration", "test")
    }
    if sum(len(item) for item in split_frames.values()) != len(frame):
        raise AssertionError("Feature split assignment lost rows")
    split_groups = {
        split_name: set(item[job.group_column].astype(str))
        for split_name, item in split_frames.items()
    }
    if (
        split_groups["fit"] & split_groups["calibration"]
        or split_groups["fit"] & split_groups["test"]
        or split_groups["calibration"] & split_groups["test"]
    ):
        raise AssertionError(
            f"{job.group_column} overlaps across feature splits"
        )

    fit_path = Path(fit_output).resolve() if fit_output else job.fit_output
    calibration_path = (
        Path(calibration_output).resolve()
        if calibration_output
        else job.calibration_output
    )
    test_path = Path(test_output).resolve() if test_output else job.test_output
    _atomic_write_csv(split_frames["fit"], fit_path)
    _atomic_write_csv(split_frames["calibration"], calibration_path)
    _atomic_write_csv(split_frames["test"], test_path)

    summary = {
        "job": job.name,
        "model": job.model,
        "dataset": job.dataset,
        "input": str(job.input_path),
        "input_samples": input_samples,
        "extracted_rows": len(frame),
        "skipped": dict(sorted(skipped.items())),
        "feature_count": len(job.spec.feature_columns),
        "label_counts": _value_counts(frame["label"]),
        "significant_sdc_target_counts": _value_counts(
            frame["significant_sdc_target"]
        ),
        "split_manifest": str(job.split_manifest),
        "split_assignment_sha256": manifest.assignment_sha256,
        "splits": {
            split_name: {
                "rows": len(item),
                "groups": len(split_groups[split_name]),
                "target_counts": _value_counts(
                    item["significant_sdc_target"]
                ),
            }
            for split_name, item in split_frames.items()
        },
        "group_overlap": 0,
        "fit_output": str(fit_path),
        "calibration_output": str(calibration_path),
        "test_output": str(test_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def _atomic_write_csv(frame: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def _value_counts(series: Any) -> dict[str, int]:
    counts = series.value_counts(dropna=False).sort_index().to_dict()
    return {str(key): int(value) for key, value in counts.items()}


def _feature_rows_equal(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> bool:
    if left.keys() != right.keys():
        return False
    for key, left_value in left.items():
        right_value = right[key]
        if isinstance(left_value, float) and isinstance(right_value, float):
            if math.isnan(left_value) and math.isnan(right_value):
                continue
        if left_value != right_value:
            return False
    return True


def _validate_matrix_pair(config: Mapping[str, Any], model: str, dataset: str) -> None:
    matrix = config.get("matrix")
    if not isinstance(matrix, list) or [model, dataset] not in matrix:
        raise ValueError(
            f"Feature job pair is not present in the experiment matrix: "
            f"{model}/{dataset}"
        )


def _layer_pairs(value: Any, name: str) -> tuple[LayerPair, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    pairs = []
    for pair in value:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"{name} entries must be [source, target]: {pair!r}")
        pairs.append((int(pair[0]), int(pair[1])))
    return tuple(pairs)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a non-empty mapping")
    return value


def _required_string(mapping: Mapping[str, Any], key: str) -> str:
    value = str(mapping.get(key, "")).strip()
    if not value:
        raise ValueError(f"Feature job requires {key}")
    return value


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path.resolve()
