"""Freeze reproducibility metadata for the pre-refactor pipelines."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import atomic_write_yaml, load_yaml


DEPENDENCY_NAMES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "xgboost",
    "torch",
    "transformers",
    "PyYAML",
)


def _run_git(repository_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository_root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def git_snapshot(repository_root: Path) -> dict[str, Any]:
    status_lines = _run_git(repository_root, "status", "--short").splitlines()
    return {
        "commit": _run_git(repository_root, "rev-parse", "HEAD"),
        "branch": _run_git(repository_root, "branch", "--show-current"),
        "dirty": bool(status_lines),
        "status": status_lines,
    }


def environment_snapshot() -> dict[str, Any]:
    dependencies: dict[str, str | None] = {}
    for name in DEPENDENCY_NAMES:
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "dependencies": dependencies,
    }


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def hash_and_count_nonempty_lines(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            if line.strip():
                count += 1
    return digest.hexdigest(), count


def _parse_integer(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def inspect_feature_csv(path: Path) -> dict[str, Any]:
    label_counts: Counter[str] = Counter()
    significance_counts: Counter[str] = Counter()
    label_significance_counts: Counter[tuple[int | None, int | None]] = Counter()
    labels: set[int] = set()
    orig_ids: set[str] = set()
    sample_uids: set[str] = set()
    row_count = 0

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        for row in reader:
            row_count += 1
            label = _parse_integer(row.get("label"))
            significance = _parse_integer(row.get("significance"))
            label_counts[str(label)] += 1
            significance_counts[str(significance)] += 1
            label_significance_counts[(label, significance)] += 1
            if label is not None:
                labels.add(label)

            if row.get("orig_id"):
                orig_ids.add(row["orig_id"])
            if row.get("sample_uid"):
                sample_uids.add(row["sample_uid"])

    binary_labels = labels.issubset({0, 1})
    if binary_labels:
        positive_count = sum(
            count
            for (label, significance), count in label_significance_counts.items()
            if label == 1 and significance == 2
        )
    else:
        positive_count = sum(
            count
            for (label, _), count in label_significance_counts.items()
            if label == 2
        )
    target_counts = Counter()
    target_counts.update({"0": row_count - positive_count} if row_count > positive_count else {})
    target_counts.update({"1": positive_count} if positive_count else {})

    return {
        "rows": row_count,
        "columns": columns,
        "column_count": len(columns),
        "unique_orig_ids": len(orig_ids),
        "unique_sample_uids": len(sample_uids),
        "label_counts": dict(sorted(label_counts.items())),
        "significance_counts": dict(sorted(significance_counts.items())),
        "significant_sdc_target_counts": dict(sorted(target_counts.items())),
    }


def inspect_file(path: Path, role: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "role": role,
        "path": str(path),
        "exists": path.exists(),
    }
    if not path.exists():
        return record

    stat = path.stat()
    print(f"[baseline] hashing {role}: {path} ({stat.st_size} bytes)", flush=True)
    if path.suffix == ".jsonl":
        sha256, row_count = hash_and_count_nonempty_lines(path)
    else:
        sha256 = sha256_file(path)
        row_count = None

    record.update(
        {
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": sha256,
        }
    )

    if path.suffix == ".jsonl":
        record["rows"] = row_count
    elif path.suffix == ".csv":
        record["data_summary"] = inspect_feature_csv(path)
    elif role == "metrics" and path.suffix == ".json":
        with path.open("r", encoding="utf-8") as stream:
            record["snapshot"] = json.load(stream)

    return record


def _resolve_path(repository_root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = repository_root / path
    return path.resolve()


def freeze_baseline(
    spec_path: str | Path,
    output_path: str | Path,
    repository_root: str | Path,
) -> dict[str, Any]:
    repository = Path(repository_root).resolve()
    spec_file = Path(spec_path).resolve()
    spec = load_yaml(spec_file)
    experiments = spec.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("Baseline spec must contain a non-empty experiments list")

    frozen_experiments = []
    for experiment in experiments:
        if not isinstance(experiment, dict) or "name" not in experiment:
            raise ValueError("Each baseline experiment must be a mapping with a name")
        files = experiment.get("files", {})
        if not isinstance(files, dict):
            raise ValueError(f"Experiment files must be a mapping: {experiment['name']}")

        frozen_files = {}
        for role, value in files.items():
            frozen_files[role] = inspect_file(_resolve_path(repository, str(value)), role)

        frozen_experiments.append(
            {
                "name": experiment["name"],
                "model": experiment.get("model"),
                "dataset": experiment.get("dataset"),
                "files": frozen_files,
            }
        )

    manifest = {
        "schema_version": 1,
        "baseline_id": spec.get("baseline_id", Path(output_path).parent.name),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Pre-refactor behavioral and data baseline",
        "source_revision": spec.get("source_revision"),
        "capture_revision": git_snapshot(repository),
        "environment": environment_snapshot(),
        "spec": {
            "path": str(spec_file),
            "sha256": sha256_file(spec_file),
        },
        "experiments": frozen_experiments,
    }
    atomic_write_yaml(output_path, manifest)
    return manifest
