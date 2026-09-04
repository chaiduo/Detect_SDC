#!/usr/bin/env python3
"""Aggregate per-job detector metrics into paper-ready CSV files."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from detect_sdc.pipeline.jobs import load_pipeline_job

from .config import load_comparison_config


DEFAULT_JOBS = (
    "qwen25_vl_earthvqa",
    "qwen25_vl_lingoqa",
    "qwen25_vl_vqav2",
    "internvl3_earthvqa",
    "internvl3_lingoqa",
    "internvl3_vqav2",
    "llava15_earthvqa",
    "llava15_lingoqa",
    "llava15_vqav2",
)
METRICS = (
    "sdc_recall",
    "significant_sdc_recall",
    "non_significant_fpr",
    "significant_sdc_precision",
    "significant_sdc_f1",
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jobs", nargs="+", default=list(DEFAULT_JOBS)
    )
    parser.add_argument("--repository-root", type=Path, default=root)
    parser.add_argument(
        "--comparison-config",
        type=Path,
        default=root
        / "compare_experiment/configs/detection_comparison.yaml",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.repository_root.resolve()
    config = load_comparison_config(
        args.comparison_config, repository_root=root
    )
    rows = []
    for job_name in args.jobs:
        job = load_pipeline_job(
            config.source_config, job_name, repository_root=root
        )
        path = config.results_root / job_name / "evaluation/metrics.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        for method, method_result in summary["methods"].items():
            for cohort, cohort_result in method_result["cohorts"].items():
                metrics = cohort_result["metrics"]
                rows.append(
                    {
                        "job": job_name,
                        "model": job.model_name,
                        "dataset": job.dataset_name,
                        "method": method,
                        "cohort": cohort,
                        **{name: metrics[name] for name in METRICS},
                        "samples": metrics["samples"],
                    }
                )

    aggregates: dict[tuple[str, str], dict[str, Any]] = {}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["cohort"])].append(row)
    for key, selected in grouped.items():
        aggregates[key] = {
            "method": key[0],
            "cohort": key[1],
            "jobs": len(selected),
            **{
                name: float(
                    np.mean([float(row[name]) for row in selected])
                )
                for name in METRICS
            },
        }

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else config.results_root / "summary"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "detailed_metrics.csv", rows)
    _write_csv(
        output_dir / "macro_average_metrics.csv",
        list(aggregates.values()),
    )
    print(f"[comparison-summary] wrote {output_dir}", flush=True)
    return 0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty summary: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
