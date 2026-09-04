#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pandas as pd

from detect_sdc.config import load_yaml
from detect_sdc.detector.xgboost import (
    XGBoostConfig,
    add_significant_sdc_target,
    calibrate_threshold_max_f1,
    get_feature_columns,
    prepare_features,
    train_binary_model,
)
from detect_sdc.features.jobs import load_feature_job
from detect_sdc.splitting import split_by_group


PAIR_PATTERN = re.compile(r"_p(\d+)_(\d+)$")
BASELINE_PAIRS = (
    (6, 7),
    (22, 23),
    (23, 24),
    (24, 25),
    (25, 26),
    (26, 27),
)
COMPACT_PAIRS = (
    (6, 7),
    (22, 23),
    (25, 26),
    (26, 27),
)
CONFIGURATIONS = {
    "72D": BASELINE_PAIRS,
    "48D": COMPACT_PAIRS,
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Repeated Fit-only validation of the shared 48D layer pairs."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configs/experiments/current.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "analysis/layer_pair_48d_stability",
    )
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=20)
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


def feature_columns_for_pairs(
    feature_columns: list[str],
    pairs: tuple[tuple[int, int], ...],
) -> list[str]:
    selected_pairs = {
        (int(match.group(1)), int(match.group(2)))
        for column in feature_columns
        if (match := PAIR_PATTERN.search(column))
    }
    missing_pairs = sorted(set(pairs) - selected_pairs)
    if missing_pairs:
        raise ValueError(f"Feature data is missing layer pairs: {missing_pairs}")

    suffixes = tuple(f"_p{source}_{target}" for source, target in pairs)
    selected = [
        column for column in feature_columns if column.endswith(suffixes)
    ]
    expected = 12 * len(pairs)
    if len(selected) != expected:
        raise ValueError(f"Expected {expected} features, found {len(selected)}")
    return selected


def confidence_interval(values: pd.Series) -> tuple[float, float]:
    count = len(values)
    mean = float(values.mean())
    if count < 2:
        return mean, mean
    standard_error = float(values.std(ddof=1)) / math.sqrt(count)
    margin = 1.96 * standard_error
    return mean - margin, mean + margin


def metric_summary(values: pd.Series) -> dict[str, float | int]:
    ci_low, ci_high = confidence_interval(values)
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean_95_ci": [ci_low, ci_high],
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_report(
    path: Path,
    *,
    summary: dict[str, Any],
    task_summary: pd.DataFrame,
) -> None:
    overall = summary["macro"]
    delta = overall["paired_delta_48D_minus_72D"]
    lines = [
        "# 48D Fit-Holdout Stability",
        "",
        "## Protocol",
        "",
        f"- Repeats: {summary['repeats']} paired random seeds.",
        f"- Seeds: {summary['seeds'][0]}-{summary['seeds'][-1]}.",
        "- Data used for selection: Fit only.",
        "- Split unit: semantic group.",
        "- 48D and 72D use identical train/holdout groups for each seed.",
        "- Each holdout threshold maximizes Significant-SDC F1.",
        "- Calibration and Final Test are not read by this experiment.",
        "",
        "## Nine-Task Macro F1",
        "",
        "| Configuration | Mean | Std | Min | Max |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("72D", "48D"):
        item = overall[name]
        lines.append(
            f"| {name} | {item['mean']:.2%} | {item['std']:.2%} | "
            f"{item['min']:.2%} | {item['max']:.2%} |"
        )
    lines.extend(
        [
            "",
            "Paired 48D - 72D difference:",
            "",
            f"- Mean: {delta['mean'] * 100:+.3f} pp",
            f"- Standard deviation: {delta['std'] * 100:.3f} pp",
            (
                "- 95% CI of mean: "
                f"[{delta['mean_95_ci'][0] * 100:+.3f}, "
                f"{delta['mean_95_ci'][1] * 100:+.3f}] pp"
            ),
            f"- 48D wins: {summary['48d_macro_wins']}/{summary['repeats']} seeds",
            "",
            "## Per-Task Validation F1",
            "",
            "| Task | 72D Mean | 48D Mean | Paired Change | 48D Wins |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in task_summary.itertuples(index=False):
        lines.append(
            f"| {row.job} | {row.mean_72d:.2%} | {row.mean_48d:.2%} | "
            f"{row.mean_delta * 100:+.3f} pp | "
            f"{row.wins_48d}/{summary['repeats']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
        ]
    )
    if delta["mean_95_ci"][0] > 0:
        lines.append(
            "The paired 95% confidence interval is above zero, supporting "
            "a stable validation improvement from 48D."
        )
    elif delta["mean_95_ci"][1] < 0:
        lines.append(
            "The paired 95% confidence interval is below zero, indicating "
            "that 48D is consistently worse than 72D."
        )
    else:
        lines.append(
            "The paired 95% confidence interval crosses zero. The result "
            "supports 48D as a compact configuration with comparable "
            "validation performance, but not as significantly better than "
            "72D."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.repeats <= 1:
        raise ValueError("--repeats must be greater than one")

    root = Path(__file__).resolve().parents[1]
    config_path = args.config.resolve()
    config = load_yaml(config_path)
    jobs = tuple(config["featurization"]["jobs"])
    seeds = tuple(range(args.seed_start, args.seed_start + args.repeats))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frames: dict[str, pd.DataFrame] = {}
    job_configs: dict[str, Any] = {}
    features: dict[str, dict[str, list[str]]] = {}
    for job_name in jobs:
        job = load_feature_job(config_path, job_name, repository_root=root)
        frame = add_significant_sdc_target(pd.read_csv(job.fit_output))
        canonical = get_feature_columns(frame)
        frames[job_name] = frame
        job_configs[job_name] = job
        features[job_name] = {
            name: feature_columns_for_pairs(canonical, pairs)
            for name, pairs in CONFIGURATIONS.items()
        }

    rows: list[dict[str, Any]] = []
    for seed_index, seed in enumerate(seeds, start=1):
        for job_index, job_name in enumerate(jobs, start=1):
            print(
                f"[seed {seed_index}/{len(seeds)}={seed}] "
                f"[job {job_index}/{len(jobs)}={job_name}]",
                flush=True,
            )
            frame = frames[job_name]
            job = job_configs[job_name]
            model_config = detector_config(
                config,
                job.model,
                seed=seed,
                n_jobs=args.n_jobs,
            )
            grouped = split_by_group(
                frame,
                group_column=job.group_column,
                holdout_ratio=model_config.test_ratio,
                random_state=seed,
            )
            for name in ("72D", "48D"):
                selected_features = features[job_name][name]
                model, training = train_binary_model(
                    prepare_features(grouped.train, selected_features),
                    grouped.train["significant_sdc_target"].astype(int),
                    prepare_features(grouped.holdout, selected_features),
                    grouped.holdout["significant_sdc_target"].astype(int),
                    config=model_config,
                )
                probability = model.predict_proba(
                    prepare_features(grouped.holdout, selected_features)
                )[:, 1]
                threshold = calibrate_threshold_max_f1(
                    probability,
                    grouped.holdout["significant_sdc_target"].astype(int),
                )
                rows.append(
                    {
                        "seed": seed,
                        "job": job_name,
                        "model": job.model,
                        "dataset": job.dataset,
                        "configuration": name,
                        "feature_count": len(selected_features),
                        "train_rows": len(grouped.train),
                        "holdout_rows": len(grouped.holdout),
                        "train_groups": grouped.summary.train_groups,
                        "holdout_groups": grouped.summary.holdout_groups,
                        "group_overlap": grouped.summary.group_overlap,
                        "precision": threshold["calibration_precision"],
                        "recall": threshold["calibration_recall"],
                        "f1": threshold["calibration_f1"],
                        "threshold": threshold["threshold"],
                        "best_iteration": training["best_iteration"],
                    }
                )
            pd.DataFrame(rows).to_csv(output_dir / "repetitions.csv", index=False)

    results = pd.DataFrame(rows)
    macro = (
        results.groupby(["seed", "configuration"], as_index=False)["f1"]
        .mean()
        .pivot(index="seed", columns="configuration", values="f1")
        .reset_index()
    )
    macro["delta_48D_minus_72D"] = macro["48D"] - macro["72D"]
    macro.to_csv(output_dir / "seed_macro.csv", index=False)

    task_rows: list[dict[str, Any]] = []
    for job_name, group in results.groupby("job"):
        paired = group.pivot(index="seed", columns="configuration", values="f1")
        delta = paired["48D"] - paired["72D"]
        task_rows.append(
            {
                "job": job_name,
                "mean_72d": paired["72D"].mean(),
                "std_72d": paired["72D"].std(ddof=1),
                "mean_48d": paired["48D"].mean(),
                "std_48d": paired["48D"].std(ddof=1),
                "mean_delta": delta.mean(),
                "std_delta": delta.std(ddof=1),
                "wins_48d": int((delta > 0).sum()),
                "ties": int((delta == 0).sum()),
            }
        )
    task_summary = pd.DataFrame(task_rows).sort_values("job")
    task_summary.to_csv(output_dir / "task_summary.csv", index=False)

    summary = {
        "protocol": "paired repeated semantic-group holdout on Fit only",
        "config": str(config_path),
        "jobs": list(jobs),
        "seeds": list(seeds),
        "repeats": len(seeds),
        "configurations": {
            name: {
                "pairs": [list(pair) for pair in pairs],
                "feature_count": 12 * len(pairs),
            }
            for name, pairs in CONFIGURATIONS.items()
        },
        "macro": {
            "72D": metric_summary(macro["72D"]),
            "48D": metric_summary(macro["48D"]),
            "paired_delta_48D_minus_72D": metric_summary(
                macro["delta_48D_minus_72D"]
            ),
        },
        "48d_macro_wins": int(
            (macro["delta_48D_minus_72D"] > 0).sum()
        ),
        "detector_config_example": asdict(
            detector_config(
                config,
                job_configs[jobs[0]].model,
                seed=seeds[0],
                n_jobs=args.n_jobs,
            )
        ),
    }
    write_json(output_dir / "summary.json", summary)
    write_report(
        output_dir / "report.md",
        summary=summary,
        task_summary=task_summary,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
