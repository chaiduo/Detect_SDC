#!/usr/bin/env python3
"""Calibrate and evaluate all detectors from one shared fault campaign."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from detect_sdc.detector.xgboost import (
    add_significant_sdc_target,
    calibrate_threshold_max_f1,
    prepare_features,
)
from detect_sdc.features.jobs import load_feature_job
from detect_sdc.pipeline.jobs import load_pipeline_job

from .config import load_comparison_config
from .evaluation import apply_threshold, evaluate_detection, threshold_at_fpr


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
        default=root / "compare_experiment/configs/detection_comparison.yaml",
    )
    parser.add_argument("--records", type=Path, default=None)
    parser.add_argument("--detector-summary", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.repository_root.resolve()
    config = load_comparison_config(
        args.comparison_config,
        repository_root=root,
    )
    pipeline_job = load_pipeline_job(
        config.source_config,
        args.job,
        repository_root=root,
    )
    feature_job = load_feature_job(
        config.source_config,
        args.job,
        repository_root=root,
    )
    result_root = config.results_root / args.job
    records_path = (args.records or pipeline_job.paths.labeled_output).resolve()
    summary_path = (
        args.detector_summary
        or feature_job.fit_output.parent.parent / "output/metrics_summary.json"
    ).resolve()
    output_dir = (args.output_dir or result_root / "evaluation").resolve()

    records = _read_records(records_path)
    detector_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    calibration = _build_rows(
        pd.read_csv(feature_job.calibration_output),
        records,
        detector_summary,
    )
    test = _build_rows(
        pd.read_csv(feature_job.test_output),
        records,
        detector_summary,
    )
    calibration_rows = list(calibration)
    calibration_negative_rows = [
        row for row in calibration if not row["is_significant_sdc"]
    ]
    test_rows = list(test)
    if not calibration_rows or not calibration_negative_rows or not test_rows:
        raise ValueError("Comparison requires calibration and test rows")

    summary: dict[str, Any] = {
        "job": args.job,
        "protocol": "significant_sdc_fit_calibration_final_test",
        "records": str(records_path),
        "detector_summary": str(summary_path),
        "calibration_objective": "maximize significant-SDC F1",
        "negative_definition": "significant_sdc_target == 0",
        "calibration_rows": len(calibration_rows),
        "calibration_non_significant_rows": len(calibration_negative_rows),
        "test_rows": len(test_rows),
        "methods": {},
    }
    prediction_rows = []
    for method_index, (method, score_column) in enumerate(METHOD_SCORES.items()):
        threshold = calibrate_threshold_max_f1(
            [float(row[score_column]) for row in calibration_rows],
            [int(row["is_significant_sdc"]) for row in calibration_rows],
        )
        selected_threshold = float(threshold["threshold"])
        method_summary = {
            "threshold_calibration": threshold,
            "cohorts": {},
            "supplementary_operating_points": {},
        }
        for cohort_name, cohort_rows in (
            ("full", test_rows),
            (
                "finite_only",
                [row for row in test_rows if not row["all_feature_nan"]],
            ),
        ):
            detected = apply_threshold(
                [float(row[score_column]) for row in cohort_rows],
                selected_threshold,
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
                "bootstrap_95_ci": _cluster_bootstrap_intervals(
                    cohort_rows,
                    detected,
                    replicates=config.bootstrap_replicates,
                    seed=config.bootstrap_seed + method_index,
                ),
                "per_fault_run": _per_run_metrics(cohort_rows, detected),
            }
            if cohort_name == "full":
                for row, prediction in zip(cohort_rows, detected):
                    prediction_rows.append(
                        {
                            "sample_uid": row["sample_uid"],
                            "semantic_group_id": row["semantic_group_id"],
                            "method": method,
                            "score": row[score_column],
                            "threshold": selected_threshold,
                            "detected": int(prediction),
                            "is_sdc": int(row["is_sdc"]),
                            "is_significant_sdc": int(
                                row["is_significant_sdc"]
                            ),
                            "run_index": row["run_index"],
                        }
                    )
        for fpr in config.supplementary_fpr_budgets:
            point = threshold_at_fpr(
                [
                    float(row[score_column])
                    for row in calibration_negative_rows
                ],
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
        writer = csv.DictWriter(stream, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    print(json.dumps(summary, indent=2, allow_nan=True), flush=True)
    return 0


def _read_records(path: Path) -> dict[str, dict[str, Any]]:
    records = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            uid = str(row["sample_uid"])
            if uid in records:
                raise ValueError(f"Duplicate sample_uid at line {line_number}: {uid}")
            records[uid] = row
    return records


def _build_rows(
    feature_frame: pd.DataFrame,
    records: dict[str, dict[str, Any]],
    detector_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    frame = add_significant_sdc_target(feature_frame)
    feature_columns = list(detector_summary["feature_columns"])
    prepared_features = prepare_features(frame, feature_columns)
    all_feature_nan = prepared_features.isna().all(axis=1).to_numpy()
    booster = xgb.Booster()
    booster.load_model(detector_summary["model_path"])
    sieve_scores = booster.inplace_predict(
        prepared_features,
        validate_features=False,
    )
    output = []
    for (_, feature), sieve_score, feature_nan in zip(
        frame.iterrows(),
        sieve_scores,
        all_feature_nan,
    ):
        uid = str(feature["sample_uid"])
        try:
            record = records[uid]
        except KeyError as error:
            raise ValueError(f"Feature row is missing fault record: {uid}") from error
        for score_name in ("ranger_score", "drdna_score", "has_non_finite"):
            if score_name not in record:
                raise ValueError(
                    f"Fault record {uid} is missing unified score: {score_name}"
                )
        output.append(
            {
                "sample_uid": uid,
                "semantic_group_id": str(feature["semantic_group_id"]),
                "injected": bool(int(feature["injected"])),
                "is_sdc": bool(int(feature["is_sdc"])),
                "is_significant_sdc": bool(
                    int(feature["significant_sdc_target"])
                ),
                "run_index": feature["run_index"],
                "has_non_finite": bool(record["has_non_finite"]),
                "all_feature_nan": bool(feature_nan),
                "ranger_score": float(record["ranger_score"]),
                "drdna_score": float(record["drdna_score"]),
                "sieve_score": float(sieve_score),
            }
        )
    return output


def _cluster_bootstrap_intervals(
    rows: list[dict[str, Any]],
    detected: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, list[float]]:
    if not rows or replicates <= 0:
        return {}
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(str(row["semantic_group_id"]), []).append(index)
    group_ids = sorted(groups)
    generator = np.random.default_rng(seed)
    is_sdc = np.asarray([row["is_sdc"] for row in rows], dtype=bool)
    is_significant = np.asarray(
        [row["is_significant_sdc"] for row in rows], dtype=bool
    )
    values: dict[str, list[float]] = {
        "sdc_recall": [],
        "significant_sdc_recall": [],
        "non_significant_fpr": [],
        "significant_sdc_f1": [],
    }
    for _ in range(replicates):
        sampled_groups = generator.choice(
            group_ids,
            size=len(group_ids),
            replace=True,
        )
        indices = np.asarray(
            [index for group in sampled_groups for index in groups[group]],
            dtype=int,
        )
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
    indices_by_run: dict[int, list[int]] = {}
    for index, row in enumerate(rows):
        if not row["injected"]:
            continue
        if pd.isna(row["run_index"]):
            raise ValueError(
                f"Injected row is missing run_index: {row['sample_uid']}"
            )
        indices_by_run.setdefault(int(row["run_index"]), []).append(index)
    for run_index, indices in sorted(indices_by_run.items()):
        selected = [rows[index] for index in indices]
        output[str(run_index)] = evaluate_detection(
            is_sdc=[row["is_sdc"] for row in selected],
            is_significant_sdc=[
                row["is_significant_sdc"] for row in selected
            ],
            detected=detected[indices],
        ).to_dict()
    return output


if __name__ == "__main__":
    raise SystemExit(main())
