#!/usr/bin/env python3
"""Build fault-free Ranger-style and Dr.DNA-style profiles."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import torch

from detect_sdc.adapters import load_dataset_adapter, load_model_adapter
from detect_sdc.features.jobs import load_feature_job
from detect_sdc.pipeline.jobs import load_pipeline_job

from .artifacts import save_profiles
from .cohorts import deterministic_subset, load_comparison_cohorts
from .config import load_comparison_config
from .monitor import OnlineActivationMonitor
from .profiles import DrDNAProfiler, RangeProfiler


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--repository-root", type=Path, default=root)
    parser.add_argument(
        "--comparison-config",
        type=Path,
        default=(
            root
            / "compare_experiment/configs/detection_comparison.yaml"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-profile-samples", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.repository_root.resolve()
    comparison = load_comparison_config(
        args.comparison_config,
        repository_root=root,
    )
    pipeline_job = load_pipeline_job(
        comparison.source_config,
        args.job,
        repository_root=root,
    )
    feature_job = load_feature_job(
        comparison.source_config,
        args.job,
        repository_root=root,
    )
    cohorts = load_comparison_cohorts(
        feature_job.train_output,
        feature_job.valid_output,
        calibration_ratio=comparison.calibration_ratio,
        random_seed=comparison.split_seed,
    )
    configured_limit = comparison.maximum_profile_samples
    requested_limit = args.max_profile_samples
    profile_limit = (
        requested_limit
        if requested_limit is not None
        else configured_limit
    )
    paper_fraction_count = max(
        1,
        math.ceil(
            len(cohorts.fit_orig_ids)
            * comparison.ranger_profile_fraction
        ),
    )
    if profile_limit is not None:
        paper_fraction_count = min(paper_fraction_count, profile_limit)
    selected = frozenset(
        deterministic_subset(
            cohorts.fit_orig_ids,
            limit=paper_fraction_count,
            random_seed=comparison.split_seed,
        )
    )

    output = (
        args.output.resolve()
        if args.output is not None
        else (
            root
            / "compare_experiment/results"
            / args.job
            / "profiles.json"
        )
    )
    if output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Profile already exists; pass --overwrite: {output}"
        )

    dataset = load_dataset_adapter(pipeline_job.dataset_config_path)
    adapter = load_model_adapter(pipeline_job.model_config_path)
    monitor = None
    range_profiler = RangeProfiler(
        monitored_layers=comparison.monitored_layers,
        max_steps=comparison.max_steps,
    )
    drdna_profiler = DrDNAProfiler(
        monitored_layers=comparison.monitored_layers,
        max_steps=comparison.max_steps,
        config=comparison.drdna,
    )
    processed = 0
    skipped_short = 0
    try:
        adapter.load(args.device)
        monitor = OnlineActivationMonitor(
            adapter.model,
            monitored_layers=comparison.monitored_layers,
            max_steps=comparison.max_steps,
        )
        monitor.register()
        with torch.no_grad():
            for sample in dataset.iter_samples(
                max_samples=pipeline_job.max_samples
            ):
                if str(sample.orig_id) not in selected:
                    continue
                monitor.start_sample()
                adapter.generate(
                    sample.question,
                    sample.image,
                    max_new_tokens=pipeline_job.max_new_tokens,
                )
                trace = monitor.finish_sample()
                if not trace.steps:
                    skipped_short += 1
                    continue
                range_profiler.add_trace(trace)
                drdna_profiler.add_trace(trace)
                processed += 1
                if processed % 50 == 0:
                    print(
                        f"[comparison-profile] {args.job} "
                        f"{processed}/{len(selected)}",
                        flush=True,
                    )
    finally:
        if monitor is not None:
            monitor.unregister()
        adapter.close()

    if processed != len(selected):
        missing = len(selected) - processed - skipped_short
        raise RuntimeError(
            f"Profile cohort incomplete: selected={len(selected)} "
            f"processed={processed} short={skipped_short} missing={missing}"
        )
    uid_digest = hashlib.sha256(
        "\n".join(sorted(selected)).encode("utf-8")
    ).hexdigest()
    save_profiles(
        output,
        range_profile=range_profiler.finalize(),
        drdna_profile=drdna_profiler.finalize(),
        metadata={
            "job": args.job,
            "model": pipeline_job.model_name,
            "dataset": pipeline_job.dataset_name,
            "profile_samples": processed,
            "skipped_short_outputs": skipped_short,
            "profile_orig_id_sha256": uid_digest,
            "fit_orig_ids": len(cohorts.fit_orig_ids),
            "calibration_orig_ids": len(
                cohorts.calibration_orig_ids
            ),
            "test_orig_ids": len(cohorts.test_orig_ids),
            "source_config": str(comparison.source_config),
        },
    )
    print(f"[comparison-profile] wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
