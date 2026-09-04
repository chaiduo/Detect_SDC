#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pandas as pd

from detect_sdc.config import load_yaml
from detect_sdc.detector.xgboost import (
    XGBoostConfig,
    add_significant_sdc_target,
    all_feature_nan_mask,
    binary_metrics,
    calibrate_threshold_max_f1,
    get_feature_columns,
    prepare_features,
    train_binary_model,
)
from detect_sdc.features.jobs import load_feature_job
from detect_sdc.splitting import split_by_group


PAIR_PATTERN = re.compile(r"_p(\d+)_(\d+)$")
TRANSFER_PAIRS = (
    (6, 7),
    (22, 23),
    (25, 26),
    (26, 27),
)
DISPLAY_NAMES = {
    "qwen25_vl_earthvqa": "Q-E",
    "qwen25_vl_lingoqa": "Q-L",
    "qwen25_vl_vqav2": "Q-V",
    "llava15_earthvqa": "L-E",
    "llava15_lingoqa": "L-L",
    "llava15_vqav2": "L-V",
    "internvl3_earthvqa": "I-E",
    "internvl3_lingoqa": "I-L",
    "internvl3_vqav2": "I-V",
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen 48D detector in a 9x9 transfer matrix."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configs/experiments/current.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "analysis/detector_transfer_48d",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=8)
    return parser.parse_args()


def detector_config(
    config: dict[str, Any],
    model: str,
    *,
    seed: int,
    n_jobs: int,
) -> XGBoostConfig:
    common = dict(config["detector"]["xgboost"]["common"])
    common.update(config["detector"]["xgboost"]["by_model"].get(model, {}))
    return replace(
        XGBoostConfig.from_mapping(common),
        random_state=seed,
        n_jobs=n_jobs,
        verbose=False,
    )


def columns_for_pairs(
    feature_columns: list[str],
    pairs: tuple[tuple[int, int], ...],
) -> list[str]:
    available = {
        (int(match.group(1)), int(match.group(2)))
        for column in feature_columns
        if (match := PAIR_PATTERN.search(column))
    }
    missing = sorted(set(pairs) - available)
    if missing:
        raise ValueError(f"Feature data is missing layer pairs: {missing}")
    suffixes = tuple(f"_p{source}_{target}" for source, target in pairs)
    selected = [
        column for column in feature_columns if column.endswith(suffixes)
    ]
    expected = 12 * len(pairs)
    if len(selected) != expected:
        raise ValueError(f"Expected {expected} features, found {len(selected)}")
    return selected


def relationship(source: Any, target: Any) -> str:
    if source.name == target.name:
        return "in_domain"
    if source.model == target.model:
        return "same_model_cross_dataset"
    if source.dataset == target.dataset:
        return "cross_model_same_dataset"
    return "cross_model_cross_dataset"


def evaluate(
    model: Any,
    frame: pd.DataFrame,
    feature_columns: list[str],
    threshold: float,
) -> dict[str, Any]:
    probability = model.predict_proba(
        prepare_features(frame, feature_columns)
    )
    prediction = (probability[:, 1] > threshold).astype(int)
    target = binary_metrics(
        frame["significant_sdc_target"].astype(int),
        prediction,
        probability,
    )["target_significant_sdc"]
    return {
        key: target[key]
        for key in (
            "support",
            "pred_total",
            "tp",
            "fp",
            "fn",
            "tn",
            "precision",
            "recall",
            "false_positive_rate",
            "f1",
        )
    }


def matrix(
    results: pd.DataFrame,
    *,
    cohort: str,
    metric: str,
    jobs: tuple[str, ...],
) -> pd.DataFrame:
    selected = results.loc[results["cohort"].eq(cohort)]
    output = selected.pivot(
        index="source_job",
        columns="target_job",
        values=metric,
    )
    output = output.loc[list(jobs), list(jobs)]
    output.index = [DISPLAY_NAMES[name] for name in output.index]
    output.columns = [DISPLAY_NAMES[name] for name in output.columns]
    return output


def markdown_matrix(frame: pd.DataFrame) -> list[str]:
    header = "| Source \\ Target | " + " | ".join(frame.columns) + " |"
    separator = "|---|" + "|".join("---:" for _ in frame.columns) + "|"
    rows = [header, separator]
    for name, values in frame.iterrows():
        cells = " | ".join(f"{value:.2%}" for value in values)
        rows.append(f"| {name} | {cells} |")
    return rows


def write_report(
    path: Path,
    *,
    jobs: tuple[str, ...],
    full_f1: pd.DataFrame,
    finite_f1: pd.DataFrame,
    relation_summary: pd.DataFrame,
    source_summary: pd.DataFrame,
) -> None:
    lines = [
        "# 48D Detector Transfer Matrix",
        "",
        "## Protocol",
        "",
        "- Fault policy: random for source and target.",
        "- Source Fit trains the XGBoost detector.",
        "- Source Calibration selects the maximum-F1 threshold.",
        "- The model and numerical threshold are frozen on every target.",
        "- Each target uses features from its own clean-trained Mapping.",
        "- Target Fit and Calibration labels are not used for an off-diagonal cell.",
        "- Finite uses the target's canonical 72D non-all-NaN cohort.",
        "",
        "Abbreviations: Q=Qwen2.5-VL, L=LLaVA-1.5, I=InternVL3; "
        "E=EarthVQA, L=LingoQA, V=VQAv2.",
        "",
        "## Full F1",
        "",
        *markdown_matrix(full_f1),
        "",
        "## Finite F1",
        "",
        *markdown_matrix(finite_f1),
        "",
        "## Transfer-Type Macro Metrics",
        "",
        "| Cohort | Relationship | Cells | Precision | Recall | F1 | FPR |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in relation_summary.itertuples(index=False):
        lines.append(
            f"| {row.cohort} | {row.relationship} | {row.cells} | "
            f"{row.precision:.2%} | {row.recall:.2%} | {row.f1:.2%} | "
            f"{row.fpr:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Off-Diagonal Performance by Source",
            "",
            "| Source | Full F1 | Finite F1 | Full FPR | Finite FPR |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in source_summary.itertuples(index=False):
        lines.append(
            f"| {DISPLAY_NAMES[row.source_job]} | {row.full_f1:.2%} | "
            f"{row.finite_f1:.2%} | {row.full_fpr:.2%} | "
            f"{row.finite_fpr:.2%} |"
        )
    best_source = source_summary.iloc[0]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "- Overall off-diagonal macro F1 is "
                f"{source_summary['full_f1'].mean():.2%} Full and "
                f"{source_summary['finite_f1'].mean():.2%} Finite."
            ),
            (
                "- The strongest source is "
                f"{best_source['source_job']}: "
                f"{best_source['full_f1']:.2%} Full F1, "
                f"{best_source['finite_f1']:.2%} Finite F1, and "
                f"{best_source['full_fpr']:.2%} FPR over its eight "
                "off-diagonal targets."
            ),
            (
                "- Transfer performance varies substantially by source. "
                "The result supports source-dependent transfer rather than "
                "universal zero-shot detector invariance."
            ),
        ]
    )
    lines.extend(
        [
            "",
            "## Source/Target Key",
            "",
            "| Key | Task |",
            "|---|---|",
        ]
    )
    for job in jobs:
        lines.append(f"| {DISPLAY_NAMES[job]} | {job} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = args.config.resolve()
    config = load_yaml(config_path)
    jobs = tuple(config["featurization"]["jobs"])
    unknown_names = sorted(set(jobs) - set(DISPLAY_NAMES))
    if unknown_names:
        raise ValueError(f"Missing display names for jobs: {unknown_names}")

    job_configs: dict[str, Any] = {}
    frames: dict[str, dict[str, pd.DataFrame]] = {}
    canonical_features: dict[str, list[str]] = {}
    transfer_features: list[str] | None = None
    for job_name in jobs:
        job = load_feature_job(config_path, job_name, repository_root=root)
        job_configs[job_name] = job
        frames[job_name] = {
            split: add_significant_sdc_target(pd.read_csv(path))
            for split, path in (
                ("fit", job.fit_output),
                ("calibration", job.calibration_output),
                ("test", job.test_output),
            )
        }
        canonical = get_feature_columns(frames[job_name]["fit"])
        selected = columns_for_pairs(canonical, TRANSFER_PAIRS)
        canonical_features[job_name] = canonical
        if transfer_features is None:
            transfer_features = selected
        elif selected != transfer_features:
            raise ValueError(
                f"48D feature order differs for transfer job {job_name}"
            )
    assert transfer_features is not None

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    for source_index, source_name in enumerate(jobs, start=1):
        print(
            f"[source {source_index}/{len(jobs)}] {source_name}",
            flush=True,
        )
        source = job_configs[source_name]
        source_frames = frames[source_name]
        model_config = detector_config(
            config,
            source.model,
            seed=args.seed,
            n_jobs=args.n_jobs,
        )
        grouped = split_by_group(
            source_frames["fit"],
            group_column=source.group_column,
            holdout_ratio=model_config.test_ratio,
            random_state=args.seed,
        )
        model, training = train_binary_model(
            prepare_features(grouped.train, transfer_features),
            grouped.train["significant_sdc_target"].astype(int),
            prepare_features(grouped.holdout, transfer_features),
            grouped.holdout["significant_sdc_target"].astype(int),
            config=model_config,
        )
        calibration_probability = model.predict_proba(
            prepare_features(
                source_frames["calibration"],
                transfer_features,
            )
        )[:, 1]
        calibration = calibrate_threshold_max_f1(
            calibration_probability,
            source_frames["calibration"]["significant_sdc_target"].astype(int),
        )
        threshold = float(calibration["threshold"])
        source_rows.append(
            {
                "source_job": source_name,
                "source_model": source.model,
                "source_dataset": source.dataset,
                "threshold": threshold,
                "calibration_precision": calibration["calibration_precision"],
                "calibration_recall": calibration["calibration_recall"],
                "calibration_f1": calibration["calibration_f1"],
                "calibration_fpr": calibration["calibration_fpr"],
                "best_iteration": training["best_iteration"],
            }
        )

        for target_name in jobs:
            target = job_configs[target_name]
            test = frames[target_name]["test"]
            finite = ~all_feature_nan_mask(
                test,
                canonical_features[target_name],
            )
            for cohort, cohort_frame in (
                ("full", test),
                ("finite", test.loc[finite]),
            ):
                metrics = evaluate(
                    model,
                    cohort_frame,
                    transfer_features,
                    threshold,
                )
                result_rows.append(
                    {
                        "source_job": source_name,
                        "source_model": source.model,
                        "source_dataset": source.dataset,
                        "target_job": target_name,
                        "target_model": target.model,
                        "target_dataset": target.dataset,
                        "relationship": relationship(source, target),
                        "cohort": cohort,
                        "threshold": threshold,
                        **metrics,
                    }
                )

    results = pd.DataFrame(result_rows)
    sources = pd.DataFrame(source_rows)
    results.to_csv(output_dir / "transfer_metrics.csv", index=False)
    sources.to_csv(output_dir / "source_models.csv", index=False)

    matrices: dict[str, pd.DataFrame] = {}
    for cohort in ("full", "finite"):
        for metric in ("f1", "precision", "recall", "false_positive_rate"):
            item = matrix(
                results,
                cohort=cohort,
                metric=metric,
                jobs=jobs,
            )
            matrices[f"{cohort}_{metric}"] = item
            item.to_csv(output_dir / f"{cohort}_{metric}_matrix.csv")

    relation_summary = (
        results.groupby(["cohort", "relationship"], as_index=False)
        .agg(
            cells=("f1", "size"),
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            f1=("f1", "mean"),
            fpr=("false_positive_rate", "mean"),
        )
        .sort_values(["cohort", "relationship"])
    )
    relation_summary.to_csv(output_dir / "relationship_summary.csv", index=False)

    off_diagonal = results.loc[results["relationship"].ne("in_domain")]
    source_summary = (
        off_diagonal.pivot_table(
            index="source_job",
            columns="cohort",
            values=["f1", "false_positive_rate"],
            aggfunc="mean",
        )
        .reset_index()
    )
    source_summary.columns = [
        "source_job",
        "finite_f1",
        "full_f1",
        "finite_fpr",
        "full_fpr",
    ]
    source_summary = source_summary.sort_values(
        ["full_f1", "finite_f1"],
        ascending=False,
    )
    source_summary.to_csv(output_dir / "source_summary.csv", index=False)

    summary = {
        "protocol": "source Fit train, source Calibration threshold, frozen target Test",
        "fault_policy": "random",
        "feature_pairs": [list(pair) for pair in TRANSFER_PAIRS],
        "feature_count": len(transfer_features),
        "seed": args.seed,
        "jobs": list(jobs),
        "cell_count": len(jobs) ** 2,
        "cohort_cell_count": len(results),
        "detector_config": {
            job_name: asdict(
                detector_config(
                    config,
                    job_configs[job_name].model,
                    seed=args.seed,
                    n_jobs=args.n_jobs,
                )
            )
            for job_name in jobs
        },
        "relationship_summary": relation_summary.to_dict(orient="records"),
        "best_off_diagonal_source_full": source_summary.iloc[0].to_dict(),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(
        output_dir / "report.md",
        jobs=jobs,
        full_f1=matrices["full_f1"],
        finite_f1=matrices["finite_f1"],
        relation_summary=relation_summary,
        source_summary=source_summary,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
