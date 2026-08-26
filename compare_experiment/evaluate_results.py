#!/usr/bin/env python3
"""Calibrate comparison detectors and report detection-only metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_comparison_config
from .evaluation import (
    apply_threshold,
    evaluate_detection,
    threshold_at_fpr,
)


METHOD_SCORES = {
    "Ranger-style": "ranger_score",
    "Dr.DNA-style": "drdna_score",
    "SIEVE": "sieve_score",
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--repository-root", type=Path, default=root)
    parser.add_argument(
        "--comparison-config",
        type=Path,
        default=root
        / "compare_experiment/configs/detection_comparison.yaml",
    )
    parser.add_argument("--scores", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--allow-answer-mismatch",
        action="store_true",
        help="Use historical labels even when replayed generation differs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.repository_root.resolve()
    config = load_comparison_config(
        args.comparison_config, repository_root=root
    )
    result_root = root / "compare_experiment/results" / args.job
    scores_path = (args.scores or result_root / "detection_scores.jsonl").resolve()
    output_dir = (args.output_dir or result_root / "evaluation").resolve()
    rows = _read_jsonl(scores_path)
    fault_exact = [
        row
        for row in rows
        if row["fault_before_matches"] and row["fault_after_matches"]
    ]
    eligible = (
        fault_exact
        if args.allow_answer_mismatch
        else [row for row in fault_exact if row["answer_matches_recorded"]]
    )
    calibration_rows = [
        row
        for row in eligible
        if row["split"] == "calibration" and not row["is_sdc"]
    ]
    test_rows = [row for row in eligible if row["split"] == "test"]
    if not calibration_rows or not test_rows:
        raise ValueError("Scores do not contain calibration and test rows")

    summary: dict[str, Any] = {
        "job": args.job,
        "source_rows": len(rows),
        "fault_exact_rows": len(fault_exact),
        "answer_match_rows": sum(
            bool(row["answer_matches_recorded"]) for row in fault_exact
        ),
        "eligible_rows": len(eligible),
        "calibration_non_sdc_rows": len(calibration_rows),
        "test_rows": len(test_rows),
        "methods": {},
    }
    prediction_rows = []
    for method_index, (method, score_column) in enumerate(
        METHOD_SCORES.items()
    ):
        calibration = threshold_at_fpr(
            [float(row[score_column]) for row in calibration_rows],
            target_fpr=config.target_fpr,
        )
        method_summary = {
            "threshold_calibration": calibration.to_dict(),
            "cohorts": {},
            "supplementary_operating_points": {},
        }
        for cohort_name, cohort_rows in (
            ("full", test_rows),
            (
                "finite_only",
                [row for row in test_rows if not row["has_non_finite"]],
            ),
        ):
            detected = apply_threshold(
                [float(row[score_column]) for row in cohort_rows],
                calibration.threshold,
            )
            metrics = evaluate_detection(
                is_sdc=[row["is_sdc"] for row in cohort_rows],
                is_significant_sdc=[
                    row["is_significant_sdc"] for row in cohort_rows
                ],
                detected=detected,
            )
            method_summary["cohorts"][cohort_name] = {
                "metrics": metrics.to_dict(),
                "bootstrap_95_ci": _bootstrap_intervals(
                    cohort_rows,
                    detected,
                    replicates=config.bootstrap_replicates,
                    seed=config.bootstrap_seed + method_index,
                ),
                "per_fault_run": _per_run_metrics(
                    cohort_rows,
                    detected,
                ),
            }
            if cohort_name == "full":
                for row, prediction in zip(cohort_rows, detected):
                    prediction_rows.append(
                        {
                            "sample_uid": row["sample_uid"],
                            "method": method,
                            "score": row[score_column],
                            "threshold": calibration.threshold,
                            "detected": int(prediction),
                            "is_sdc": row["is_sdc"],
                            "is_significant_sdc": row[
                                "is_significant_sdc"
                            ],
                            "run_index": row["run_index"],
                        }
                    )
        for fpr in config.supplementary_fpr_budgets:
            point = threshold_at_fpr(
                [float(row[score_column]) for row in calibration_rows],
                target_fpr=fpr,
            )
            detected = apply_threshold(
                [float(row[score_column]) for row in test_rows],
                point.threshold,
            )
            metrics = evaluate_detection(
                is_sdc=[row["is_sdc"] for row in test_rows],
                is_significant_sdc=[
                    row["is_significant_sdc"] for row in test_rows
                ],
                detected=detected,
            )
            method_summary["supplementary_operating_points"][str(fpr)] = {
                "calibration": point.to_dict(),
                "test_metrics": metrics.to_dict(),
            }
        summary["methods"][method] = method_summary

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(prediction_rows[0]),
        )
        writer.writeheader()
        writer.writerows(prediction_rows)
    print(json.dumps(summary, indent=2, allow_nan=True), flush=True)
    return 0


def _bootstrap_intervals(
    rows: list[dict[str, Any]],
    detected: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, list[float]]:
    if not rows or replicates <= 0:
        return {}
    generator = np.random.default_rng(seed)
    is_sdc = np.asarray([row["is_sdc"] for row in rows], dtype=bool)
    is_significant = np.asarray(
        [row["is_significant_sdc"] for row in rows], dtype=bool
    )
    values: dict[str, list[float]] = {
        "sdc_recall": [],
        "significant_sdc_recall": [],
        "non_sdc_fpr": [],
        "significant_sdc_f1": [],
    }
    for _ in range(replicates):
        indices = generator.integers(0, len(rows), size=len(rows))
        metrics = evaluate_detection(
            is_sdc=is_sdc[indices],
            is_significant_sdc=is_significant[indices],
            detected=detected[indices],
        )
        for name in values:
            values[name].append(float(getattr(metrics, name)))
    return {
        name: [
            float(np.percentile(samples, 2.5)),
            float(np.percentile(samples, 97.5)),
        ]
        for name, samples in values.items()
    }


def _per_run_metrics(
    rows: list[dict[str, Any]],
    detected: np.ndarray,
) -> dict[str, Any]:
    output = {}
    run_indices = sorted({int(row["run_index"]) for row in rows})
    for run_index in run_indices:
        indices = [
            index
            for index, row in enumerate(rows)
            if int(row["run_index"]) == run_index
        ]
        selected = [rows[index] for index in indices]
        metrics = evaluate_detection(
            is_sdc=[row["is_sdc"] for row in selected],
            is_significant_sdc=[
                row["is_significant_sdc"] for row in selected
            ],
            detected=detected[indices],
        )
        output[str(run_index)] = metrics.to_dict()
    return output


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


if __name__ == "__main__":
    raise SystemExit(main())
