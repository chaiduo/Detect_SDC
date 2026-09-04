#!/usr/bin/env python3
"""Run one fault campaign and attach all comparison-method signals."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from detect_sdc.pipeline.injection import run_injection_job

from .artifacts import load_profiles
from .config import load_comparison_config
from .monitor import OnlineActivationMonitor


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
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--telemetry-max-steps", type=int, default=None)
    parser.add_argument("--profiles", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume-from-run", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.repository_root.resolve()
    comparison = load_comparison_config(
        args.comparison_config,
        repository_root=root,
    )
    result_root = comparison.results_root / args.job
    profile_path = (
        args.profiles or result_root / "profiles.json"
    ).resolve()
    range_profile, drdna_profile, _ = load_profiles(profile_path)

    def monitor_factory(model: Any) -> OnlineActivationMonitor:
        return OnlineActivationMonitor(
            model,
            monitored_layers=comparison.monitored_layers,
            max_steps=comparison.max_steps,
        )

    def score(trace: Any) -> dict[str, Any]:
        unobservable = not trace.steps
        if unobservable:
            ranger_score = drdna_score = float("-inf")
        else:
            ranger_score = range_profile.score(trace)
            drdna_score = drdna_profile.score(trace)
        return {
            "steps_observed": len(trace.steps),
            "unobservable_short_output": unobservable,
            "has_non_finite": trace.has_non_finite(),
            "ranger_score": ranger_score,
            "drdna_score": drdna_score,
        }

    run_injection_job(
        comparison.source_config,
        args.job,
        repository_root=root,
        device=args.device,
        max_samples=args.max_samples,
        output_path=args.output,
        overwrite=args.overwrite,
        resume_from_run=args.resume_from_run,
        telemetry_max_steps=args.telemetry_max_steps,
        auxiliary_monitor_factory=monitor_factory,
        auxiliary_scorer=score,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
