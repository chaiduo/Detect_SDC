#!/usr/bin/env python3
"""Evaluate frozen detector predictions by injected-value deviation bucket."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from detect_sdc.config import load_yaml
from detect_sdc.detector.xgboost import prepare_features


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
METHODS = ("Ranger-style", "Dr.DNA-style", "SIEVE")
BUCKETS = (
    "zero",
    "(0,1]",
    "(1,1e6]",
    "(1e6,1e12]",
    "(1e12,1e18]",
    "(1e18,1e24]",
    "(1e24,1e30]",
    "(1e30,1e36]",
    ">1e36",
    "non_finite",
)
BOUNDS = (1.0, 1e6, 1e12, 1e18, 1e24, 1e30, 1e36)


@dataclass(frozen=True)
class JobPaths:
    job: str
    model: str
    dataset: str
    labels: Path
    detector_summary: Path
    detector_predictions: Path
    comparison_predictions: Path


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen Final Test predictions in fixed absolute "
            "fault-deviation buckets."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configs/experiments/current.yaml",
    )
    parser.add_argument(
        "--comparison-root",
        type=Path,
        default=root / "compare_experiment/results_v2",
    )
    parser.add_argument(
        "--jobs",
        nargs="+",
        default=list(DEFAULT_JOBS),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "analysis/deviation_bucket_evaluation_20260831",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip jobs whose final prediction artifacts are incomplete.",
    )
    return parser.parse_args()


def deviation_bucket(before: Any, after: Any) -> str:
    before_value = float(before)
    after_value = float(after)
    if not math.isfinite(before_value) or not math.isfinite(after_value):
        return "non_finite"
    delta = abs(after_value - before_value)
    if not math.isfinite(delta):
        return "non_finite"
    if delta == 0.0:
        return "zero"
    for bound, name in zip(BOUNDS, BUCKETS[1:-2]):
        if delta <= bound:
            return name
    return ">1e36"


def binary_metrics(
    significant: Iterable[bool | int],
    detected: Iterable[bool | int],
) -> dict[str, Any]:
    truth = np.asarray(list(significant), dtype=bool)
    prediction = np.asarray(list(detected), dtype=bool)
    if truth.shape != prediction.shape or truth.ndim != 1:
        raise ValueError("Targets and predictions must be matching vectors")
    positive = int(truth.sum())
    negative = int(len(truth) - positive)
    true_positive = int(np.sum(truth & prediction))
    false_positive = int(np.sum(~truth & prediction))
    false_negative = positive - true_positive
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, positive)
    f1 = (
        0.0
        if precision + recall == 0.0
        else 2.0 * precision * recall / (precision + recall)
    )
    return {
        "samples": int(len(truth)),
        "significant_samples": positive,
        "non_significant_samples": negative,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "fpr": _ratio(false_positive, negative),
        "significant_rate": _ratio(positive, len(truth)),
    }


def cluster_bootstrap(
    frame: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> dict[str, list[float] | None]:
    if frame.empty or int(frame["is_significant_sdc"].sum()) == 0:
        return {"recall": None, "precision": None, "f1": None, "fpr": None}
    grouped = []
    for _, group in frame.groupby("bootstrap_group", sort=True):
        truth = group["is_significant_sdc"].to_numpy(dtype=bool)
        prediction = group["detected"].to_numpy(dtype=bool)
        grouped.append(
            np.asarray(
                [
                    np.sum(truth & prediction),
                    np.sum(~truth & prediction),
                    np.sum(truth & ~prediction),
                    np.sum(~truth & ~prediction),
                ],
                dtype=np.int64,
            )
        )
    counts = np.stack(grouped)
    generator = np.random.default_rng(seed)
    sampled = generator.integers(
        0,
        len(counts),
        size=(replicates, len(counts)),
    )
    totals = counts[sampled].sum(axis=1)
    tp, fp, fn, tn = (totals[:, index] for index in range(4))
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        f1 = 2 * tp / (2 * tp + fp + fn)
        fpr = fp / (fp + tn)
    return {
        name: _percentile_interval(values)
        for name, values in (
            ("recall", recall),
            ("precision", precision),
            ("f1", f1),
            ("fpr", fpr),
        )
    }


def evaluate_frame(
    frame: pd.DataFrame,
    *,
    job: str,
    model: str,
    dataset: str,
    cohort: str,
    method: str,
    bucket: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    metrics = binary_metrics(
        frame["is_significant_sdc"],
        frame["detected"],
    )
    intervals = cluster_bootstrap(
        frame,
        replicates=replicates,
        seed=seed,
    )
    positive_samples = int(metrics["significant_samples"])
    return {
        "job": job,
        "model": model,
        "dataset": dataset,
        "cohort": cohort,
        "method": method,
        "bucket": bucket,
        **metrics,
        **{
            f"{name}_ci_low": None if interval is None else interval[0]
            for name, interval in intervals.items()
        },
        **{
            f"{name}_ci_high": None if interval is None else interval[1]
            for name, interval in intervals.items()
        },
        "evidence": (
            "no_positive"
            if positive_samples == 0
            else "underpowered"
            if positive_samples < 30
            else "adequate"
        ),
    }


def load_job_frame(paths: JobPaths) -> pd.DataFrame:
    summary = json.loads(paths.detector_summary.read_text(encoding="utf-8"))
    feature_columns = list(summary["feature_columns"])
    feature_frame = pd.read_csv(
        paths.detector_predictions,
        usecols=["sample_uid", "injected", *feature_columns],
    )
    feature_frame = feature_frame.loc[
        feature_frame["injected"].astype(int).eq(1)
    ].copy()
    numeric_features = prepare_features(feature_frame, feature_columns)
    feature_frame["all_feature_nan"] = numeric_features.isna().all(axis=1)
    finite_by_uid = dict(
        zip(
            feature_frame["sample_uid"].astype(str),
            ~feature_frame["all_feature_nan"],
        )
    )
    target_uids = set(finite_by_uid)
    bucket_by_uid: dict[str, str] = {}
    with paths.labels.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            uid = str(record["sample_uid"])
            if uid not in target_uids:
                continue
            fault = record.get("fault")
            if not isinstance(fault, dict):
                raise ValueError(
                    f"Missing fault metadata at {paths.labels}:{line_number}"
                )
            try:
                bucket_by_uid[uid] = deviation_bucket(
                    fault["before"],
                    fault["after"],
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid fault values at {paths.labels}:{line_number}"
                ) from error
    missing = sorted(target_uids - set(bucket_by_uid))
    if missing:
        raise ValueError(
            f"{paths.job}: {len(missing)} Final Test rows lack fault metadata"
        )

    predictions = pd.read_csv(paths.comparison_predictions)
    predictions = predictions.loc[
        predictions["sample_uid"].astype(str).isin(target_uids)
    ].copy()
    expected = len(target_uids)
    for method in METHODS:
        method_rows = predictions.loc[predictions["method"].eq(method)]
        if len(method_rows) != expected:
            raise ValueError(
                f"{paths.job}/{method}: expected {expected} injected rows, "
                f"got {len(method_rows)}"
            )
    predictions["sample_uid"] = predictions["sample_uid"].astype(str)
    predictions["finite_features"] = predictions["sample_uid"].map(
        finite_by_uid
    )
    predictions["bucket"] = predictions["sample_uid"].map(bucket_by_uid)
    predictions["bootstrap_group"] = (
        paths.job + ":" + predictions["semantic_group_id"].astype(str)
    )
    predictions["job"] = paths.job
    predictions["model"] = paths.model
    predictions["dataset"] = paths.dataset
    return predictions


def resolve_jobs(
    *,
    root: Path,
    config_path: Path,
    comparison_root: Path,
    requested_jobs: Iterable[str],
    skip_missing: bool,
) -> tuple[list[JobPaths], list[str]]:
    config = load_yaml(config_path)
    jobs = config["execution"]["jobs"]
    resolved = []
    skipped = []
    for job in requested_jobs:
        if job not in jobs:
            raise ValueError(f"Unknown job: {job}")
        item = jobs[job]
        labels = _resolve(root, item["labeled_output"])
        output = _resolve(
            root,
            config["featurization"]["jobs"][job]["fit_output"],
        ).parent.parent / "output"
        paths = JobPaths(
            job=job,
            model=str(item["model"]),
            dataset=str(item["dataset"]),
            labels=labels,
            detector_summary=output / "metrics_summary.json",
            detector_predictions=(
                output / "significant_sdc_binary_test_full_predictions.csv"
            ),
            comparison_predictions=(
                comparison_root / job / "evaluation/predictions.csv"
            ),
        )
        missing = [
            path
            for path in (
                paths.labels,
                paths.detector_summary,
                paths.detector_predictions,
                paths.comparison_predictions,
            )
            if not path.is_file()
        ]
        if missing and skip_missing:
            skipped.append(job)
            continue
        if missing:
            raise FileNotFoundError(
                f"{job} is incomplete: {[str(path) for path in missing]}"
            )
        resolved.append(paths)
    if not resolved:
        raise ValueError("No completed jobs were selected")
    return resolved, skipped


def evaluate(
    frames: list[pd.DataFrame],
    *,
    replicates: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows = []
    scopes = [
        (str(frame["job"].iloc[0]), frame)
        for frame in frames
    ]
    scopes.append(("all_completed", pd.concat(frames, ignore_index=True)))
    for scope_index, (job, frame) in enumerate(scopes):
        model = (
            str(frame["model"].iloc[0])
            if job != "all_completed"
            else "all"
        )
        dataset = (
            str(frame["dataset"].iloc[0])
            if job != "all_completed"
            else "all"
        )
        for cohort_name, cohort in (
            ("full_injected", frame),
            ("finite_injected", frame.loc[frame["finite_features"]]),
        ):
            for method_index, method in enumerate(METHODS):
                method_rows = cohort.loc[cohort["method"].eq(method)]
                for bucket_index, bucket in enumerate(BUCKETS):
                    selected = method_rows.loc[method_rows["bucket"].eq(bucket)]
                    rows.append(
                        evaluate_frame(
                            selected,
                            job=job,
                            model=model,
                            dataset=dataset,
                            cohort=cohort_name,
                            method=method,
                            bucket=bucket,
                            replicates=replicates,
                            seed=(
                                seed
                                + scope_index * 10_000
                                + method_index * 100
                                + bucket_index
                            ),
                        )
                    )
    return rows


def write_report(
    rows: list[dict[str, Any]],
    *,
    jobs: list[JobPaths],
    skipped: list[str],
    output_dir: Path,
    replicates: int,
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    detailed = output_dir / "bucket_metrics.csv"
    with detailed.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "protocol": {
            "training": "one detector per job on the complete Fit split",
            "threshold": "maximize F1 on the complete Calibration split",
            "evaluation": "frozen predictions partitioned only on Final Test",
            "negative_definition": "significant_sdc_target == 0",
            "finite_definition": "not all 72 SIEVE features are NaN",
            "deviation": "absolute(after - before)",
            "buckets": list(BUCKETS),
            "bootstrap_unit": "job:semantic_group_id",
            "bootstrap_replicates": replicates,
            "seed": seed,
        },
        "included_jobs": [item.job for item in jobs],
        "skipped_jobs": skipped,
        "rows": rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_markdown(rows, jobs=jobs, skipped=skipped, output_dir=output_dir)
    _plot_metrics(rows, output_dir=output_dir)
    _plot_counts(rows, output_dir=output_dir)


def _write_markdown(
    rows: list[dict[str, Any]],
    *,
    jobs: list[JobPaths],
    skipped: list[str],
    output_dir: Path,
) -> None:
    aggregate = [
        row
        for row in rows
        if row["job"] == "all_completed"
        and row["cohort"] == "full_injected"
    ]
    lines = [
        "# Deviation-bucket evaluation",
        "",
        "## Protocol",
        "",
        "- Train one detector per model/dataset job on the complete Fit split.",
        "- Select one threshold by maximizing Significant-SDC F1 on the complete Calibration split.",
        "- Freeze the detector and threshold before partitioning Final Test.",
        "- Define deviation as `abs(after - before)`; NaN/Inf is a separate bucket.",
        "- Full-injected contains every injected Final Test row.",
        "- Finite-injected excludes only rows whose 72 SIEVE features are all NaN.",
        "- Buckets are diagnostic strata unavailable to the deployed detector.",
        "",
        f"Included jobs: {', '.join(item.job for item in jobs)}.",
        f"Skipped incomplete jobs: {', '.join(skipped) if skipped else 'none'}.",
        "",
        "## Aggregate Full-injected results",
        "",
        "| Bucket | Samples | Significant | Sig. rate | Ranger F1 | Dr.DNA F1 | SIEVE F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    by_bucket = {
        bucket: {
            row["method"]: row
            for row in aggregate
            if row["bucket"] == bucket
        }
        for bucket in BUCKETS
    }
    for bucket in BUCKETS:
        methods = by_bucket[bucket]
        reference = methods["SIEVE"]
        lines.append(
            "| {bucket} | {samples:,} | {positive:,} | {rate} | {ranger} | "
            "{drdna} | {sieve} |".format(
                bucket=bucket,
                samples=reference["samples"],
                positive=reference["significant_samples"],
                rate=_percent(reference["significant_rate"]),
                ranger=_metric_cell(methods["Ranger-style"]),
                drdna=_metric_cell(methods["Dr.DNA-style"]),
                sieve=_metric_cell(methods["SIEVE"]),
            )
        )

    sieve_rows = [
        row for row in aggregate if row["method"] == "SIEVE"
    ]
    small = _combine_metric_rows(
        row
        for row in sieve_rows
        if row["bucket"] in {"zero", "(0,1]", "(1,1e6]"}
    )
    extreme = _combine_metric_rows(
        row
        for row in sieve_rows
        if row["bucket"] in {">1e36", "non_finite"}
    )
    total_significant = sum(
        int(row["significant_samples"]) for row in sieve_rows
    )
    lines.extend(
        [
            "",
            "## Main findings",
            "",
            (
                f"- `{extreme['significant_samples']:,}/{total_significant:,}` "
                f"({100.0 * _ratio(extreme['significant_samples'], total_significant):.2f}%) "
                "Significant-SDC rows have `abs(after-before) > 1e36` or "
                "a non-finite injected value."
            ),
            (
                f"- At `abs(after-before) <= 1e6`, SIEVE sees "
                f"{small['significant_samples']:,} positives among "
                f"{small['samples']:,} rows and reaches "
                f"Recall={_percent(small['recall'])}, "
                f"Precision={_percent(small['precision'])}, "
                f"F1={_percent(small['f1'])}."
            ),
            (
                f"- In the extreme buckets, SIEVE reaches "
                f"Recall={_percent(extreme['recall'])}, "
                f"Precision={_percent(extreme['precision'])}, "
                f"F1={_percent(extreme['f1'])}."
            ),
            "",
            "## Interpretation rules",
            "",
            "- `underpowered` means fewer than 30 Significant-SDC samples in the bucket.",
            "- `no_positive` means Recall and F1 are not evidential for that bucket.",
            "- Do not compare raw bucket precision without considering bucket prevalence.",
            "- Do not train or select thresholds from Final Test bucket membership.",
            "",
            "Detailed per-job Full/Finite metrics and cluster-bootstrap intervals are in `bucket_metrics.csv`.",
        ]
    )
    (output_dir / "report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _plot_metrics(rows: list[dict[str, Any]], *, output_dir: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), sharex=True)
    colors = {
        "Ranger-style": "#228be6",
        "Dr.DNA-style": "#f59f00",
        "SIEVE": "#e64980",
    }
    markers = {"Ranger-style": "o", "Dr.DNA-style": "s", "SIEVE": "^"}
    for axis, cohort in zip(axes, ("full_injected", "finite_injected")):
        selected = [
            row
            for row in rows
            if row["job"] == "all_completed" and row["cohort"] == cohort
        ]
        for method in METHODS:
            method_rows = {
                row["bucket"]: row
                for row in selected
                if row["method"] == method
            }
            axis.plot(
                range(len(BUCKETS)),
                [
                    (
                        np.nan
                        if method_rows[bucket]["significant_samples"] == 0
                        else 100.0 * method_rows[bucket]["f1"]
                    )
                    for bucket in BUCKETS
                ],
                marker=markers[method],
                linewidth=1.8,
                markersize=4.5,
                color=colors[method],
                label=method,
            )
        axis.set_ylabel("Significant-SDC F1 (%)")
        axis.set_title(cohort.replace("_", " ").title())
        axis.set_ylim(0, 105)
        axis.grid(axis="y", linewidth=0.5, alpha=0.35)
    axes[0].legend(frameon=False, ncol=3, loc="lower right")
    axes[-1].set_xticks(range(len(BUCKETS)))
    axes[-1].set_xticklabels(BUCKETS, rotation=35, ha="right")
    axes[-1].set_xlabel("Absolute injected-value deviation bucket")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(
            output_dir / f"bucket_f1.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


def _plot_counts(rows: list[dict[str, Any]], *, output_dir: Path) -> None:
    selected = {
        row["bucket"]: row
        for row in rows
        if row["job"] == "all_completed"
        and row["cohort"] == "full_injected"
        and row["method"] == "SIEVE"
    }
    positive = [selected[bucket]["significant_samples"] for bucket in BUCKETS]
    negative = [
        selected[bucket]["non_significant_samples"] for bucket in BUCKETS
    ]
    figure, axis = plt.subplots(figsize=(10.5, 4.2))
    x = np.arange(len(BUCKETS))
    axis.bar(x, negative, color="#74c0fc", label="Non-significant")
    axis.bar(
        x,
        positive,
        bottom=negative,
        color="#fa5252",
        label="Significant SDC",
    )
    axis.set_yscale("symlog", linthresh=1)
    axis.set_ylabel("Final Test injected rows (symlog)")
    axis.set_xticks(x)
    axis.set_xticklabels(BUCKETS, rotation=35, ha="right")
    axis.set_xlabel("Absolute injected-value deviation bucket")
    axis.legend(frameon=False, ncol=2)
    axis.grid(axis="y", linewidth=0.5, alpha=0.35)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(
            output_dir / f"bucket_counts.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


def _percent(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _metric_cell(row: dict[str, Any]) -> str:
    if row["significant_samples"] == 0:
        return "N/A"
    suffix = " *" if row["evidence"] == "underpowered" else ""
    return f"{100.0 * float(row['f1']):.2f}%{suffix}"


def _combine_metric_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    selected = list(rows)
    true_positive = sum(int(row["true_positive"]) for row in selected)
    false_positive = sum(int(row["false_positive"]) for row in selected)
    false_negative = sum(int(row["false_negative"]) for row in selected)
    significant_samples = sum(
        int(row["significant_samples"]) for row in selected
    )
    non_significant_samples = sum(
        int(row["non_significant_samples"]) for row in selected
    )
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    return {
        "samples": significant_samples + non_significant_samples,
        "significant_samples": significant_samples,
        "recall": recall,
        "precision": precision,
        "f1": (
            0.0
            if precision + recall == 0.0
            else 2.0 * precision * recall / (precision + recall)
        ),
    }


def _percentile_interval(values: np.ndarray) -> list[float] | None:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return None
    return [
        float(np.percentile(finite, 2.5)),
        float(np.percentile(finite, 97.5)),
    ]


def _ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else float(numerator / denominator)


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def main() -> int:
    args = parse_args()
    if args.bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")
    root = Path(__file__).resolve().parents[1]
    config_path = args.config.resolve()
    comparison_root = args.comparison_root.resolve()
    jobs, skipped = resolve_jobs(
        root=root,
        config_path=config_path,
        comparison_root=comparison_root,
        requested_jobs=args.jobs,
        skip_missing=args.skip_missing,
    )
    frames = []
    for paths in jobs:
        print(f"[deviation-buckets] loading {paths.job}", flush=True)
        frames.append(load_job_frame(paths))
    rows = evaluate(
        frames,
        replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    write_report(
        rows,
        jobs=jobs,
        skipped=skipped,
        output_dir=args.output_dir.resolve(),
        replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    print(f"[deviation-buckets] wrote {args.output_dir.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
