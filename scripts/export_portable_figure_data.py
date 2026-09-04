#!/usr/bin/env python3
"""Export compact, portable data required to reproduce paper figures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from plot_fault_quadrant_comparison import (
    build_jobs as build_quadrant_jobs,
)
from plot_significant_sdc_share import (
    build_jobs as build_share_jobs,
)
from plot_significant_sdc_share import (
    collect_counts,
)

SHARE_OUTPUT = Path("figures/significant_sdc_share.csv")
QUADRANT_OUTPUT = Path("figures/fault_quadrant_counts.csv")
MANIFEST_OUTPUT = Path("figures/figure_data_manifest.json")


@dataclass
class QuadrantCounts:
    counts: Counter[tuple[bool, bool]]
    total: int = 0
    non_finite: int = 0
    skipped: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=REPOSITORY_ROOT,
    )
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def write_share_data(root: Path) -> Path:
    jobs = build_share_jobs(root)
    counts = collect_counts(jobs)
    output = root / SHARE_OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "model",
                "dataset",
                "all_sdc",
                "significant_sdc",
                "removed_all_feature_nan",
                "significant_sdc_share_percent",
            ]
        )
        for job in jobs:
            item = counts[job.name]
            if item is None:
                raise FileNotFoundError(f"Missing feature CSVs for {job.name}")
            writer.writerow(
                [
                    job.model,
                    job.dataset,
                    item.all_sdc,
                    item.significant_sdc,
                    item.removed_all_feature_nan,
                    item.percentage,
                ]
            )
    return output


def _quadrant_row(
    *,
    model: str,
    dataset: str,
    threshold: float,
    non_finite_policy: str,
    stats: Any,
) -> list[Any]:
    return [
        model,
        dataset,
        threshold,
        non_finite_policy,
        stats.counts[(False, False)],
        stats.counts[(False, True)],
        stats.counts[(True, False)],
        stats.counts[(True, True)],
        stats.total,
        stats.non_finite,
        stats.skipped,
    ]


def _parse_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def collect_quadrant_variants(
    path: Path,
    thresholds: tuple[float, ...],
) -> dict[tuple[float, str], QuadrantCounts]:
    variants = {
        (threshold, policy): QuadrantCounts(Counter())
        for threshold in thresholds
        for policy in ("exclude", "count_as_large")
    }
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            fault = record.get("fault")
            if not isinstance(fault, dict):
                continue
            before = _parse_float(fault.get("before"))
            after = _parse_float(fault.get("after"))
            if before is None or after is None:
                for item in variants.values():
                    item.skipped += 1
                continue

            significant = record.get("is_sdc") == 1 and record.get("significance") == 2
            finite = math.isfinite(before) and math.isfinite(after)
            for threshold in thresholds:
                excluded = variants[(threshold, "exclude")]
                included = variants[(threshold, "count_as_large")]
                if not finite:
                    excluded.non_finite += 1
                    included.non_finite += 1
                    included.counts[(True, significant)] += 1
                    included.total += 1
                    continue
                large = abs(after - before) > threshold
                excluded.counts[(large, significant)] += 1
                included.counts[(large, significant)] += 1
                excluded.total += 1
                included.total += 1
    return variants


def write_quadrant_data(root: Path, workers: int) -> Path:
    jobs = build_quadrant_jobs(root)
    job_thresholds = [
        ((1.0, 5.0, 10.0) if job.model == "Qwen2.5-VL-7B" and job.dataset == "LingoQA" else (1.0,)) for job in jobs
    ]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                collect_quadrant_variants,
                job.labels_path,
                thresholds,
            )
            for job, thresholds in zip(jobs, job_thresholds)
        ]
        job_variants = [future.result() for future in futures]

    output = root / QUADRANT_OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "model",
                "dataset",
                "threshold",
                "non_finite_policy",
                "small_non_significant",
                "small_significant",
                "large_non_significant",
                "large_significant",
                "total",
                "non_finite",
                "skipped",
            ]
        )
        for job, thresholds, variants in zip(
            jobs,
            job_thresholds,
            job_variants,
        ):
            for threshold in thresholds:
                for policy in ("exclude", "count_as_large"):
                    writer.writerow(
                        _quadrant_row(
                            model=job.model,
                            dataset=job.dataset,
                            threshold=threshold,
                            non_finite_policy=policy,
                            stats=variants[(threshold, policy)],
                        )
                    )
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_sources(root: Path) -> list[Path]:
    patterns = (
        "figures/*.csv",
        "analysis/iclr_v2/**/*.csv",
        "analysis/iclr_v2/**/*.json",
        "compare_experiment/results_v2/summary/*.csv",
    )
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(path for path in root.glob(pattern) if path.is_file())
    return sorted(paths)


def write_manifest(root: Path) -> Path:
    output = root / MANIFEST_OUTPUT
    entries = []
    for path in manifest_sources(root):
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    payload = {
        "schema_version": 1,
        "purpose": (
            "Portable source data for paper tables and figures; "
            "large JSONL and model artifacts are intentionally excluded."
        ),
        "files": entries,
        "total_bytes": sum(item["bytes"] for item in entries),
    }
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def main() -> int:
    args = parse_args()
    root = args.repository_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Repository root does not exist: {root}")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    outputs = [
        write_share_data(root),
        write_quadrant_data(root, args.workers),
    ]
    outputs.append(write_manifest(root))
    for output in outputs:
        print(
            f"{output.relative_to(root)}: {output.stat().st_size} bytes",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
