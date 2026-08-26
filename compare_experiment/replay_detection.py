#!/usr/bin/env python3
"""Replay saved faults once and score all comparison detectors."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from detect_sdc.adapters import load_dataset_adapter, load_model_adapter
from detect_sdc.config import load_yaml
from detect_sdc.fault_injector import FaultInjector
from detect_sdc.features.jobs import load_feature_job
from detect_sdc.pipeline.injection import load_mapping_model
from detect_sdc.pipeline.jobs import load_pipeline_job

from .artifacts import load_profiles
from .cohorts import load_comparison_cohorts
from .config import load_comparison_config
from .monitor import OnlineActivationMonitor
from .sieve import (
    SieveTraceScorer,
    detector_config,
    train_comparison_detector,
)


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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--profiles", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.repository_root.resolve()
    comparison = load_comparison_config(
        args.comparison_config, repository_root=root
    )
    pipeline_job = load_pipeline_job(
        comparison.source_config, args.job, repository_root=root
    )
    feature_job = load_feature_job(
        comparison.source_config, args.job, repository_root=root
    )
    cohorts = load_comparison_cohorts(
        feature_job.train_output,
        feature_job.valid_output,
        calibration_ratio=comparison.calibration_ratio,
        random_seed=comparison.split_seed,
    )
    result_root = root / "compare_experiment/results" / args.job
    manifest_path = (args.manifest or result_root / "replay_manifest.jsonl").resolve()
    profile_path = (args.profiles or result_root / "profiles.json").resolve()
    output_path = (args.output or result_root / "detection_scores.jsonl").resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Replay manifest missing: {manifest_path}")
    if not profile_path.is_file():
        raise FileNotFoundError(f"Comparison profiles missing: {profile_path}")
    records = _read_jsonl(manifest_path, args.max_records)
    completed = set()
    if output_path.exists() and not args.overwrite:
        completed = {
            str(row["sample_uid"]) for row in _read_jsonl(output_path, None)
        }
    elif args.overwrite and output_path.exists():
        output_path.unlink()
    pending = [
        row for row in records if str(row["sample_uid"]) not in completed
    ]
    if not pending:
        print(f"[comparison-replay] nothing pending: {output_path}")
        return 0

    range_profile, drdna_profile, _ = load_profiles(profile_path)
    k2_train = _online_k2_train_csv(pipeline_job.paths.labeled_output)
    xgb_config = detector_config(
        load_yaml(comparison.source_config),
        pipeline_job.model_name,
    )
    detector, feature_columns = train_comparison_detector(
        k2_train,
        fit_orig_ids=cohorts.fit_orig_ids,
        output_dir=result_root / "sieve_detector",
        config=xgb_config,
    )
    mapping_model = load_mapping_model(
        pipeline_job.injection_config,
        pipeline_job.paths.mapping_model,
        device=args.device,
    )
    sieve = SieveTraceScorer(
        predictor=mapping_model,
        detector=detector,
        detector_feature_columns=feature_columns,
        layer_pairs=comparison.layer_pairs,
        projection_dim=pipeline_job.projection_dim,
        projection_seed=pipeline_job.profiler_seed,
        max_steps=comparison.max_steps,
        device=args.device,
    )

    dataset = load_dataset_adapter(pipeline_job.dataset_config_path)
    needed = {str(row["orig_id"]) for row in pending}
    samples = {
        str(sample.orig_id): sample
        for sample in dataset.iter_samples(max_samples=pipeline_job.max_samples)
        if str(sample.orig_id) in needed
    }
    missing = needed - set(samples)
    if missing:
        raise ValueError(
            f"Dataset is missing {len(missing)} manifest orig_ids; "
            f"examples={sorted(missing)[:5]}"
        )

    adapter = load_model_adapter(pipeline_job.model_config_path)
    monitor = None
    injector = None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        adapter.load(args.device)
        monitor = OnlineActivationMonitor(
            adapter.model,
            monitored_layers=comparison.monitored_layers,
            max_steps=comparison.max_steps,
        )
        injector = FaultInjector(adapter.model, mode="activation")
        with output_path.open("a", encoding="utf-8") as stream, torch.no_grad():
            for index, record in enumerate(pending, start=1):
                fault = record["fault"]
                injector.reset()
                injector.set_inject_info(
                    idx=int(fault["idx"]),
                    module_name=str(fault["module"]),
                    inject_step=int(fault["forward"]),
                    bit_positions=[
                        int(bit) for bit in fault["bit_positions"]
                    ],
                )
                injector.inject()
                injector.register_step_hooks()
                monitor.register()
                try:
                    monitor.start_sample()
                    sample = samples[str(record["orig_id"])]
                    answer = adapter.generate(
                        sample.question,
                        sample.image,
                        max_new_tokens=pipeline_job.max_new_tokens,
                    )
                    trace = monitor.finish_sample()
                    actual_fault = dict(injector.fault_info or {})
                finally:
                    monitor.unregister()
                    injector.unregister_hooks()
                unobservable = not trace.steps
                if unobservable:
                    ranger_score = float("-inf")
                    drdna_score = float("-inf")
                    sieve_score = float("-inf")
                else:
                    ranger_score = range_profile.score(trace)
                    drdna_score = drdna_profile.score(trace)
                    sieve_score = sieve.score(trace)
                result = {
                    **record,
                    "replay_answer": answer,
                    "answer_matches_recorded": (
                        answer == record["recorded_fault_answer"]
                    ),
                    "fault_before_matches": _same_number(
                        actual_fault.get("before"), fault.get("before")
                    ),
                    "fault_after_matches": _same_number(
                        actual_fault.get("after"), fault.get("after")
                    ),
                    "steps_observed": len(trace.steps),
                    "unobservable_short_output": unobservable,
                    "has_non_finite": trace.has_non_finite(),
                    "ranger_score": ranger_score,
                    "drdna_score": drdna_score,
                    "sieve_score": sieve_score,
                }
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                stream.flush()
                if index % 20 == 0 or index == len(pending):
                    print(
                        f"[comparison-replay] {args.job} "
                        f"{index}/{len(pending)}",
                        flush=True,
                    )
    finally:
        if monitor is not None:
            monitor.unregister()
        if injector is not None:
            injector.unregister_hooks()
        adapter.close()
    return 0


def _online_k2_train_csv(labels_path: Path) -> Path:
    dataset_root = labels_path.parents[1]
    model_root = labels_path.parents[2]
    path = (
        model_root
        / "online_step_ablation_20260814"
        / dataset_root.name
        / "k_2/train_data/train.csv"
    )
    if not path.is_file():
        raise FileNotFoundError(f"K=2 training CSV missing: {path}")
    return path


def _read_jsonl(
    path: Path,
    limit: int | None,
) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _same_number(actual: Any, expected: Any) -> bool:
    if actual is None or expected is None:
        return actual is expected
    actual_value = float(actual)
    expected_value = float(expected)
    if math.isnan(actual_value) and math.isnan(expected_value):
        return True
    return actual_value == expected_value


if __name__ == "__main__":
    raise SystemExit(main())
