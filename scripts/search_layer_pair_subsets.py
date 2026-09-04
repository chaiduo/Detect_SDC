#!/usr/bin/env python3

from __future__ import annotations

import argparse
import itertools
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


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Search all non-empty subsets of the current layer pairs."
    )
    parser.add_argument("--job", required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configs/experiments/current.yaml",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root / "analysis/layer_pair_subset_search",
    )
    parser.add_argument("--n-jobs", type=int, default=4)
    return parser.parse_args()


def detector_config(config_path: Path, model: str, n_jobs: int) -> XGBoostConfig:
    config = load_yaml(config_path)
    common = dict(config["detector"]["xgboost"]["common"])
    common.update(config["detector"]["xgboost"]["by_model"].get(model, {}))
    return replace(
        XGBoostConfig.from_mapping(common),
        n_jobs=n_jobs,
        verbose=False,
    )


def layer_pairs(feature_columns: list[str]) -> tuple[tuple[int, int], ...]:
    pairs = {
        (int(match.group(1)), int(match.group(2)))
        for column in feature_columns
        if (match := PAIR_PATTERN.search(column))
    }
    if not pairs:
        raise ValueError("No layer-pair feature columns found")
    return tuple(sorted(pairs))


def columns_for_pairs(
    feature_columns: list[str],
    pairs: tuple[tuple[int, int], ...],
) -> list[str]:
    suffixes = tuple(f"_p{source}_{target}" for source, target in pairs)
    selected = [
        column for column in feature_columns if column.endswith(suffixes)
    ]
    if len(selected) != 12 * len(pairs):
        raise ValueError(
            f"Expected {12 * len(pairs)} features for {pairs}, got {len(selected)}"
        )
    return selected


def target_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    target = metrics["target_significant_sdc"]
    return {
        key: target[key]
        for key in ("precision", "recall", "f1", "tp", "fp", "fn", "tn")
    }


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
    return target_metrics(
        binary_metrics(
            frame["significant_sdc_target"].astype(int),
            prediction,
            probability,
        )
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = args.config.resolve()
    job = load_feature_job(
        config_path,
        args.job,
        repository_root=root,
    )
    frames = {
        "fit": add_significant_sdc_target(pd.read_csv(job.fit_output)),
        "calibration": add_significant_sdc_target(
            pd.read_csv(job.calibration_output)
        ),
        "test": add_significant_sdc_target(pd.read_csv(job.test_output)),
    }
    canonical_features = get_feature_columns(frames["fit"])
    pairs = layer_pairs(canonical_features)
    canonical_finite = ~all_feature_nan_mask(
        frames["test"],
        canonical_features,
    )
    grouped = split_by_group(
        frames["fit"],
        group_column=job.group_column,
        holdout_ratio=detector_config(
            config_path, job.model, args.n_jobs
        ).test_ratio,
        random_state=detector_config(
            config_path, job.model, args.n_jobs
        ).random_state,
    )
    config = detector_config(config_path, job.model, args.n_jobs)
    candidates = [
        subset
        for count in range(1, len(pairs) + 1)
        for subset in itertools.combinations(pairs, count)
    ]
    rows: list[dict[str, Any]] = []
    destination = args.output_root.resolve() / args.job
    destination.mkdir(parents=True, exist_ok=True)

    for index, subset in enumerate(candidates, start=1):
        features = columns_for_pairs(canonical_features, subset)
        print(
            f"[{args.job}] {index}/{len(candidates)} pairs={subset}",
            flush=True,
        )
        model, training = train_binary_model(
            prepare_features(grouped.train, features),
            grouped.train["significant_sdc_target"].astype(int),
            prepare_features(grouped.holdout, features),
            grouped.holdout["significant_sdc_target"].astype(int),
            config=config,
        )
        holdout_probability = model.predict_proba(
            prepare_features(grouped.holdout, features)
        )[:, 1]
        selection = calibrate_threshold_max_f1(
            holdout_probability,
            grouped.holdout["significant_sdc_target"].astype(int),
        )
        calibration_probability = model.predict_proba(
            prepare_features(frames["calibration"], features)
        )[:, 1]
        threshold = calibrate_threshold_max_f1(
            calibration_probability,
            frames["calibration"]["significant_sdc_target"].astype(int),
        )
        full = evaluate(
            model,
            frames["test"],
            features,
            float(threshold["threshold"]),
        )
        finite = evaluate(
            model,
            frames["test"].loc[canonical_finite],
            features,
            float(threshold["threshold"]),
        )
        rows.append(
            {
                "job": args.job,
                "model": job.model,
                "dataset": job.dataset,
                "pairs": json.dumps(subset),
                "pair_key": "+".join(
                    f"p{source}_{target}" for source, target in subset
                ),
                "pair_count": len(subset),
                "feature_count": len(features),
                "fit_holdout_f1": selection["calibration_f1"],
                "fit_holdout_precision": selection["calibration_precision"],
                "fit_holdout_recall": selection["calibration_recall"],
                "calibration_threshold": threshold["threshold"],
                "calibration_f1": threshold["calibration_f1"],
                **{f"full_{key}": value for key, value in full.items()},
                **{f"finite_{key}": value for key, value in finite.items()},
                "best_iteration": training["best_iteration"],
            }
        )
        pd.DataFrame(rows).to_csv(destination / "candidates.csv", index=False)

    ranked = sorted(
        rows,
        key=lambda row: (
            -row["fit_holdout_f1"],
            row["pair_count"],
            row["pair_key"],
        ),
    )
    current_key = "+".join(
        f"p{source}_{target}" for source, target in pairs
    )
    current = next(row for row in rows if row["pair_key"] == current_key)
    summary = {
        "job": args.job,
        "selection_protocol": (
            "maximize F1 on the Fit-internal group holdout; use Calibration "
            "only for the operating threshold; evaluate Final Test once"
        ),
        "canonical_finite_definition": (
            "exclude rows whose complete current 72-feature vector is all NaN"
        ),
        "candidate_count": len(rows),
        "current": current,
        "selected": ranked[0],
        "detector_config": asdict(config),
    }
    write_json(destination / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
