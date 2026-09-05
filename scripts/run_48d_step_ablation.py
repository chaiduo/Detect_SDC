#!/usr/bin/env python3

"""Run current-v2 prefix-step ablations with the paper's 48D layer pairs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from detect_sdc.config import load_yaml
from detect_sdc.detector.xgboost import (
    XGBoostConfig,
    all_feature_nan_mask,
    run_calibrated_xgboost,
)
from detect_sdc.features.extraction import (
    FeatureSpec,
    SampleSkipped,
    extract_feature_row,
    iter_json_samples,
)
from detect_sdc.features.jobs import load_feature_job
from detect_sdc.splitting import validate_identity_columns


K_VALUES = (2, 4, 8, 16)
PAIRS = ((6, 7), (22, 23), (25, 26), (26, 27))
METRICS = ("cos_sim", "mean_diff", "std_diff", "l2_distance")
STATISTICS = ("mean", "max", "min")
DATASETS = ("earthvqa", "lingoqa", "vqav2")
MODELS = ("qwen25_vl", "llava15", "internvl3")
META_COLUMNS = (
    "orig_id", "semantic_group_id", "split", "sample_uid", "injected",
    "run_index", "is_sdc", "fault_component", "fault_layer_index",
    "fault_op_type", "fault_bit_categories", "total_steps", "last_k_steps",
    "num_steps_used",
)
TARGET_COLUMNS = ("significance", "label", "significant_sdc_target")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=root / "configs/experiments/current.yaml",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=root / "analysis/step_ablation_48d",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help=(
            "Optional root containing <job>/labels.jsonl inputs, for example "
            "analysis/telemetry_50."
        ),
    )
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def detector_config(
    config: dict[str, Any], model: str, n_jobs: int
) -> XGBoostConfig:
    values = dict(config["detector"]["xgboost"]["common"])
    values.update(config["detector"]["xgboost"]["by_model"].get(model, {}))
    return replace(
        XGBoostConfig.from_mapping(values),
        n_jobs=n_jobs,
        verbose=False,
    )


def specs() -> dict[int, FeatureSpec]:
    return {
        k: FeatureSpec(
            selected_layer_pairs=PAIRS,
            distance_pairs=PAIRS,
            last_k_steps=k,
            finite_only=True,
            step_window="prefix",
        )
        for k in K_VALUES
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def extract_job(
    job: Any,
    output_root: Path,
    feature_specs: dict[int, FeatureSpec],
    overwrite: bool,
) -> None:
    destinations = {
        k: output_root / f"k_{k}" / "features.csv" for k in K_VALUES
    }
    if all(path.is_file() for path in destinations.values()) and not overwrite:
        return

    columns = [
        *META_COLUMNS,
        *feature_specs[K_VALUES[-1]].feature_columns,
        *TARGET_COLUMNS,
    ]
    writers: dict[int, csv.DictWriter] = {}
    temporary: dict[int, Path] = {}
    skipped: Counter[str] = Counter()
    seen: set[str] = set()
    input_rows = 0
    extracted_rows = 0

    with ExitStack() as stack:
        for k, destination in destinations.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            temp = destination.with_suffix(".tmp")
            temporary[k] = temp
            stream = stack.enter_context(temp.open("w", encoding="utf-8", newline=""))
            writers[k] = csv.DictWriter(stream, fieldnames=columns)
            writers[k].writeheader()

        for sample in iter_json_samples(job.input_path):
            input_rows += 1
            rows: dict[int, dict[str, Any]] = {}
            try:
                for k, feature_spec in feature_specs.items():
                    rows[k] = extract_feature_row(
                        sample,
                        spec=feature_spec,
                        uid_namespace=job.uid_namespace,
                    )
            except SampleSkipped as error:
                skipped[error.reason] += 1
                continue
            uid = str(rows[K_VALUES[0]]["sample_uid"])
            if uid in seen:
                skipped["duplicate_sample"] += 1
                continue
            seen.add(uid)
            for k in K_VALUES:
                writers[k].writerow(rows[k])
            extracted_rows += 1
            if extracted_rows % 1000 == 0:
                print(f"[extract] {job.name}: {extracted_rows}", flush=True)

    for k, temp in temporary.items():
        temp.replace(destinations[k])
    write_json(
        output_root / "extraction_summary.json",
        {
            "job": job.name,
            "input_rows": input_rows,
            "extracted_rows": extracted_rows,
            "skipped": dict(skipped),
            "layer_pairs": [list(pair) for pair in PAIRS],
            "k_values": list(K_VALUES),
            "feature_count": 48,
        },
    )


def validate_step_coverage(
    output_root: Path,
) -> None:
    """Reject a K sweep when the labeled telemetry was collected only for K=2."""
    coverage: dict[int, int] = {}
    for k in K_VALUES:
        frame = pd.read_csv(
            output_root / f"k_{k}" / "features.csv",
            usecols=["num_steps_used"],
        )
        coverage[k] = int(frame["num_steps_used"].max())
    if max(coverage[k] for k in K_VALUES if k > 2) <= 2:
        raise RuntimeError(
            "Current labeled telemetry contains at most two decoding steps; "
            "K=4/8/16 cannot be evaluated. Re-collect the v2 campaign with "
            "telemetry_max_steps >= 16 before running this sweep. "
            f"Observed max num_steps_used by K: {coverage}"
        )


def prepare_job(
    job: Any,
    output_root: Path,
    feature_specs: dict[int, FeatureSpec],
    overwrite: bool,
) -> dict[str, Any]:
    fixed_path = output_root / "fixed_k2_finite.json"
    split_paths = [
        output_root / f"k_{k}" / f"{split}.csv"
        for k in K_VALUES
        for split in ("fit", "calibration", "test")
    ]
    if fixed_path.is_file() and all(path.is_file() for path in split_paths) and not overwrite:
        return json.loads(fixed_path.read_text(encoding="utf-8"))

    baseline = pd.read_csv(output_root / "k_2/features.csv")
    validate_identity_columns(
        baseline,
        group_column=job.group_column,
        sample_uid_column="sample_uid",
    )
    baseline_test = baseline.loc[baseline["split"] == "test"].copy()
    k2_features = list(feature_specs[2].feature_columns)
    fixed_test = baseline_test.loc[
        ~all_feature_nan_mask(baseline_test, k2_features)
    ].copy()
    fixed_uids = set(fixed_test["sample_uid"].astype(str))

    for k in K_VALUES:
        frame = pd.read_csv(output_root / f"k_{k}/features.csv")
        test = frame.loc[frame["sample_uid"].astype(str).isin(fixed_uids)].copy()
        if len(test) != len(fixed_uids):
            raise ValueError(f"{job.name} K={k} changed the fixed K=2 cohort")
        write_csv(frame.loc[frame["split"] == "fit"], output_root / f"k_{k}/fit.csv")
        write_csv(
            frame.loc[frame["split"] == "calibration"],
            output_root / f"k_{k}/calibration.csv",
        )
        write_csv(test, output_root / f"k_{k}/test.csv")

    result = {
        "definition": "Final Test rows with at least one finite K=2 48D feature",
        "test_rows_before_filter": len(baseline_test),
        "test_rows": len(fixed_test),
        "positive_samples": int(fixed_test["significant_sdc_target"].sum()),
        "negative_samples": int(
            len(fixed_test) - fixed_test["significant_sdc_target"].sum()
        ),
        "split_manifest": str(job.split_manifest),
    }
    write_json(fixed_path, result)
    return result


def train_job(
    job: Any,
    output_root: Path,
    config: dict[str, Any],
    n_jobs: int,
    overwrite: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for k in K_VALUES:
        destination = output_root / f"k_{k}"
        model_output = destination / "output"
        summary_path = model_output / "metrics_summary.json"
        if summary_path.is_file() and not overwrite:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            print(f"[train] {job.name} K={k}", flush=True)
            summary = run_calibrated_xgboost(
                destination / "fit.csv",
                destination / "calibration.csv",
                destination / "test.csv",
                model_output,
                group_column=job.group_column,
                config=detector_config(config, job.model, n_jobs),
            )
            write_json(summary_path, summary)
        metrics = summary["metrics"]["test_full"]["target_significant_sdc"]
        test = pd.read_csv(
            destination / "test.csv",
            usecols=["total_steps", "num_steps_used"],
        )
        rows.append(
            {
                "job": job.name,
                "model": job.model,
                "dataset": job.dataset,
                "k": k,
                "feature_count": 48,
                "evaluation_cohort": "fixed_k2_finite",
                "test_rows": len(test),
                "mean_steps_used": float(test["num_steps_used"].mean()),
                "median_steps_used": float(test["num_steps_used"].median()),
                "p95_steps_used": float(np.percentile(test["num_steps_used"], 95)),
                **{
                    key: metrics[key]
                    for key in ("precision", "recall", "f1", "tp", "fp", "fn", "tn")
                },
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = args.config.resolve()
    config = load_yaml(config_path)
    feature_specs = specs()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []

    for model in MODELS:
        for dataset in DATASETS:
            name = f"{model}_{dataset}"
            job = load_feature_job(config_path, name, repository_root=root)
            if args.input_root is not None:
                input_path = (
                    args.input_root.resolve() / name / "labels.jsonl"
                )
                if not input_path.is_file():
                    raise FileNotFoundError(
                        f"Telemetry label input does not exist: {input_path}"
                    )
                job = replace(job, input_path=input_path)
            destination = output_dir / name
            extract_job(job, destination, feature_specs, args.overwrite)
            validate_step_coverage(destination)
            cohort = prepare_job(job, destination, feature_specs, args.overwrite)
            rows = train_job(
                job, destination, config, args.n_jobs, args.overwrite
            )
            all_rows.extend(rows)
            write_json(destination / "cohort.json", cohort)

    detailed = pd.DataFrame(all_rows)
    write_csv(detailed, output_dir / "detailed_results.csv")
    aggregate = (
        detailed.groupby("k", as_index=False)
        .agg(
            macro_precision=("precision", "mean"),
            macro_recall=("recall", "mean"),
            macro_f1=("f1", "mean"),
            macro_fpr=("fp", "sum"),
            mean_steps_used=("mean_steps_used", "mean"),
        )
    )
    negative = detailed.groupby("k")[["fp", "tn"]].sum()
    aggregate["macro_fpr"] = negative["fp"].to_numpy() / (
        negative["fp"] + negative["tn"]
    ).to_numpy()
    write_csv(aggregate, output_dir / "aggregate_results.csv")
    write_json(
        output_dir / "summary.json",
        {
            "protocol": "current v2; 48D prefix-step ablation",
            "k_values": list(K_VALUES),
            "layer_pairs": [list(pair) for pair in PAIRS],
            "feature_count": 48,
            "jobs": [f"{model}_{dataset}" for model in MODELS for dataset in DATASETS],
            "cohort": "fixed K=2 non-all-NaN 48D Final Test cohort",
        },
    )
    print(aggregate.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
