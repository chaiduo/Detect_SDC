#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from detect_sdc.config import load_yaml
from detect_sdc.detector.xgboost import (
    XGBoostConfig,
    all_feature_nan_mask,
    run_xgboost,
)
from detect_sdc.features.extraction import (
    FeatureSpec,
    SampleSkipped,
    extract_feature_row,
    iter_json_samples,
)
from detect_sdc.features.jobs import load_feature_job
from detect_sdc.splitting import validate_identity_columns


K_VALUES = (1, 2, 4, 8, 12, 16, 24, 32, 50)
DATASETS = {
    "EarthVQA": "earthvqa",
    "LingoQA": "lingoqa",
    "VQAv2": "vqav2",
}
MODELS = {
    "Qwen2.5-VL-7B": ("qwen25_vl", "Qwen2.5-VL-7B"),
    "InternVL3-8B": ("internvl3", "InternVL3-8B"),
    "LLaVA-1.5-7B": ("llava15", "llava-v1.5-7B"),
}
META_COLUMNS = (
    "orig_id",
    "sample_uid",
    "total_steps",
    "last_k_steps",
    "num_steps_used",
)
TARGET_COLUMNS = (
    "significance",
    "label",
    "significant_sdc_target",
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run prefix-step online detection latency ablations."
    )
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument("--repository-root", type=Path, default=root)
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configs/experiments/current.yaml",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--overwrite-features", action="store_true")
    parser.add_argument("--overwrite-detectors", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    return args


def detector_config(config_path: Path, model_key: str) -> XGBoostConfig:
    config = load_yaml(config_path)
    detector = _mapping(config.get("detector"), "detector")
    xgboost = _mapping(detector.get("xgboost"), "detector.xgboost")
    values = dict(_mapping(xgboost.get("common"), "xgboost.common"))
    values.update(
        _mapping(
            _mapping(xgboost.get("by_model"), "xgboost.by_model").get(model_key),
            model_key,
        )
    )
    return XGBoostConfig.from_mapping(values)


def prefix_specs(base_spec: FeatureSpec) -> dict[int, FeatureSpec]:
    return {
        k: FeatureSpec(
            selected_layer_pairs=base_spec.selected_layer_pairs,
            distance_pairs=base_spec.distance_pairs,
            last_k_steps=k,
            finite_only=base_spec.finite_only,
            step_window="prefix",
        )
        for k in K_VALUES
    }


def feature_columns(spec: FeatureSpec) -> list[str]:
    return list(spec.feature_columns)


def extract_prefix_features(
    *,
    input_path: Path,
    uid_namespace: str,
    specs: Mapping[int, FeatureSpec],
    output_root: Path,
    max_samples: int | None,
    overwrite: bool,
) -> dict[str, Any]:
    destinations = {
        k: output_root / f"k_{k}/features/all.csv"
        for k in K_VALUES
    }
    if all(path.is_file() for path in destinations.values()) and not overwrite:
        return json.loads(
            (output_root / "extraction_summary.json").read_text(encoding="utf-8")
        )

    for path in destinations.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_suffix(f"{path.suffix}.tmp").unlink(missing_ok=True)

    columns = [
        *META_COLUMNS,
        *feature_columns(specs[K_VALUES[-1]]),
        *TARGET_COLUMNS,
    ]
    input_samples = 0
    extracted_rows = 0
    skipped: Counter[str] = Counter()
    seen_uids: set[str] = set()
    temporary_paths: dict[int, Path] = {}

    try:
        with ExitStack() as stack:
            writers: dict[int, csv.DictWriter] = {}
            for k, destination in destinations.items():
                temporary = destination.with_suffix(f"{destination.suffix}.tmp")
                temporary_paths[k] = temporary
                stream = stack.enter_context(
                    temporary.open("w", encoding="utf-8", newline="")
                )
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writers[k] = writer

            for sample in iter_json_samples(input_path, max_samples=max_samples):
                input_samples += 1
                rows: dict[int, dict[str, Any]] = {}
                try:
                    for k in K_VALUES:
                        rows[k] = extract_feature_row(
                            sample,
                            spec=specs[k],
                            uid_namespace=uid_namespace,
                        )
                except SampleSkipped as error:
                    skipped[error.reason] += 1
                    continue

                sample_uid = str(rows[K_VALUES[-1]]["sample_uid"])
                if sample_uid in seen_uids:
                    skipped["duplicate_sample"] += 1
                    continue
                seen_uids.add(sample_uid)
                for k in K_VALUES:
                    writers[k].writerow(rows[k])
                extracted_rows += 1
                if extracted_rows % 1000 == 0:
                    print(
                        f"[prefix-extract] rows={extracted_rows} "
                        f"input={input_samples}",
                        flush=True,
                    )

        for k, temporary in temporary_paths.items():
            temporary.replace(destinations[k])
    except BaseException:
        for temporary in temporary_paths.values():
            temporary.unlink(missing_ok=True)
        raise

    summary = {
        "input": str(input_path),
        "input_samples": input_samples,
        "extracted_rows": extracted_rows,
        "skipped": dict(sorted(skipped.items())),
        "k_values": list(K_VALUES),
        "step_window": "prefix",
        "feature_count": len(feature_columns(specs[K_VALUES[-1]])),
    }
    _write_json(output_root / "extraction_summary.json", summary)
    return summary


def prepare_splits(
    *,
    output_root: Path,
    base_job: Any,
    specs: Mapping[int, FeatureSpec],
    overwrite: bool,
) -> dict[str, Any]:
    manifest_path = output_root / "fixed_cohort.json"
    split_paths = [
        output_root / f"k_{k}/train_data/train.csv"
        for k in K_VALUES
    ] + [
        output_root / f"k_{k}/train_data/valid_fixed_k50.csv"
        for k in K_VALUES
    ]
    if manifest_path.is_file() and all(path.is_file() for path in split_paths) and not overwrite:
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    baseline = pd.read_csv(output_root / "k_50/features/all.csv")
    validate_identity_columns(
        baseline,
        group_column=base_job.group_column,
        sample_uid_column="sample_uid",
    )
    source_train = pd.read_csv(
        base_job.train_output,
        usecols=["sample_uid"],
    )
    source_valid = pd.read_csv(
        base_job.valid_output,
        usecols=["sample_uid"],
    )
    train_uids = set(source_train["sample_uid"].astype(str))
    source_valid_uids = set(source_valid["sample_uid"].astype(str))
    baseline_uids = set(baseline["sample_uid"].astype(str))
    if train_uids & source_valid_uids:
        raise ValueError("Main train and validation sample UIDs overlap")
    if train_uids | source_valid_uids != baseline_uids:
        raise ValueError("Prefix features do not match the main experiment split")

    baseline_uid = baseline["sample_uid"].astype(str)
    baseline_train = baseline.loc[baseline_uid.isin(train_uids)].copy()
    baseline_valid = baseline.loc[
        baseline_uid.isin(source_valid_uids)
    ].copy()
    baseline_columns = feature_columns(specs[50])
    fixed_valid = baseline_valid.loc[
        ~all_feature_nan_mask(baseline_valid, baseline_columns)
    ].copy()
    valid_uids = set(fixed_valid["sample_uid"].astype(str))

    for k in K_VALUES:
        frame = pd.read_csv(output_root / f"k_{k}/features/all.csv")
        uids = frame["sample_uid"].astype(str)
        train = frame.loc[uids.isin(train_uids)].copy()
        valid = frame.loc[uids.isin(valid_uids)].copy()
        if len(train) != len(train_uids) or len(valid) != len(valid_uids):
            raise ValueError(f"K={k} does not match the K=50 sample cohort")
        _write_csv(train, output_root / f"k_{k}/train_data/train.csv")
        _write_csv(
            valid,
            output_root / f"k_{k}/train_data/valid_fixed_k50.csv",
        )

    target = pd.to_numeric(
        fixed_valid["significant_sdc_target"],
        errors="raise",
    ).astype(int)
    manifest = {
        "definition": (
            "Main-experiment validation rows with at least one finite "
            "K=50 detector feature"
        ),
        "step_window": "prefix",
        "train_rows": len(baseline_train),
        "validation_rows_before_filter": len(baseline_valid),
        "validation_rows": len(fixed_valid),
        "positive_samples": int(target.sum()),
        "negative_samples": int(len(target) - target.sum()),
        "sample_uid_sha256": _uid_digest(fixed_valid["sample_uid"]),
        "split_source": "main_experiment_feature_csv",
        "train_sample_uid_sha256": _uid_digest(
            baseline_train["sample_uid"]
        ),
    }
    _write_json(manifest_path, manifest)
    return manifest


def train_detectors(
    *,
    model: str,
    dataset: str,
    output_root: Path,
    config: XGBoostConfig,
    overwrite: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for k in K_VALUES:
        root = output_root / f"k_{k}"
        summary_path = root / "output/metrics_summary.json"
        if summary_path.is_file() and not overwrite:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            print(
                f"[prefix-train] {model} {dataset} K={k}",
                flush=True,
            )
            summary = run_xgboost(
                root / "train_data/train.csv",
                root / "train_data/valid_fixed_k50.csv",
                root / "output",
                group_column="orig_id",
                config=config,
            )
        metrics = summary["valid_full_metrics"]["target_significant_sdc"]
        valid = pd.read_csv(
            root / "train_data/valid_fixed_k50.csv",
            usecols=["total_steps", "num_steps_used"],
        )
        rows.append(
            {
                "model": model,
                "dataset": dataset,
                "k": k,
                "step_window": "prefix",
                "evaluation_cohort": "fixed_k50_non_all_nan",
                "validation_rows": len(valid),
                "mean_steps_used": float(valid["num_steps_used"].mean()),
                "median_steps_used": float(valid["num_steps_used"].median()),
                "p95_steps_used": float(
                    np.percentile(valid["num_steps_used"], 95)
                ),
                "finished_before_k_share": float(
                    (valid["total_steps"] < k).mean()
                ),
                **{
                    key: metrics[key]
                    for key in (
                        "precision",
                        "recall",
                        "f1",
                        "tp",
                        "fp",
                        "fn",
                        "tn",
                    )
                },
            }
        )
        _write_summary(output_root / "summary.csv", rows)
    return rows


def verify_baseline(
    *,
    rows: list[dict[str, Any]],
    main_summary_path: Path,
) -> None:
    main = json.loads(main_summary_path.read_text(encoding="utf-8"))
    expected = main["model_summary"]["valid_non_all_nan_metrics"][
        "target_significant_sdc"
    ]
    actual = next(row for row in rows if row["k"] == 50)
    fields = ("precision", "recall", "f1", "tp", "fp", "fn", "tn")
    mismatches = {
        field: (actual[field], expected[field])
        for field in fields
        if not np.isclose(
            float(actual[field]),
            float(expected[field]),
            rtol=0.0,
            atol=1e-15,
        )
    }
    if mismatches:
        raise AssertionError(f"K=50 does not reproduce main experiment: {mismatches}")


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _uid_digest(values: pd.Series) -> str:
    payload = "\n".join(sorted(values.astype(str))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a non-empty mapping")
    return value


def main() -> int:
    args = parse_args()
    root = args.repository_root.resolve()
    config_path = args.config.resolve()
    model_key, model_directory = MODELS[args.model]
    dataset_key = DATASETS[args.dataset]
    output_base = (
        args.output_root.resolve()
        if args.output_root is not None
        else root / model_directory / "online_step_ablation_20260814"
    )
    output_root = output_base / args.dataset
    job = load_feature_job(
        config_path,
        f"{model_key}_{dataset_key}",
        repository_root=root,
    )
    specs = prefix_specs(job.spec)
    extract_prefix_features(
        input_path=job.input_path,
        uid_namespace=job.uid_namespace,
        specs=specs,
        output_root=output_root,
        max_samples=args.max_samples,
        overwrite=args.overwrite_features,
    )
    prepare_splits(
        output_root=output_root,
        base_job=job,
        specs=specs,
        overwrite=args.overwrite_features,
    )
    rows = train_detectors(
        model=args.model,
        dataset=args.dataset,
        output_root=output_root,
        config=detector_config(config_path, model_key),
        overwrite=args.overwrite_detectors,
    )
    if args.max_samples is None:
        verify_baseline(
            rows=rows,
            main_summary_path=root / model_directory / args.dataset
            / "output/metrics_summary.json",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
