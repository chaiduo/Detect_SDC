"""Unified stage dispatcher for configured experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..detector import run_detector_job
from ..features.jobs import load_feature_job, run_feature_job
from ..labeling import run_label_job
from . import PipelineStage
from .injection import run_injection_job
from .jobs import load_pipeline_job
from .mapping import run_mapping_job, run_mapping_training_job
from .profile import run_profile_job


def run_stage(
    config_path: str | Path,
    job_name: str,
    stage: str | PipelineStage,
    *,
    repository_root: str | Path,
    device: str = "cuda:0",
    max_samples: int | None = None,
    batch_size: int = 64,
    overwrite: bool = False,
    resume_injection_from_run: int | None = None,
    telemetry_max_steps: int | None = None,
    injection_output_path: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    selected = PipelineStage(stage)
    if (
        resume_injection_from_run is not None
        and selected != PipelineStage.INJECT
    ):
        raise ValueError(
            "resume_injection_from_run is only valid for the inject stage"
        )
    job = load_pipeline_job(
        config_path,
        job_name,
        repository_root=repository_root,
    )

    if selected == PipelineStage.TRAIN_MAPPING:
        summary = _dry_run_summary(job.name, selected) if dry_run else run_mapping_training_job(
            config_path,
            job.name,
            repository_root=repository_root,
            device=device,
            overwrite=overwrite,
        )
    elif selected == PipelineStage.PROFILE:
        summary = _dry_run_summary(job.name, selected) if dry_run else _run_profile(
            config_path,
            job,
            repository_root,
            device,
            max_samples,
            overwrite,
        )
    elif selected == PipelineStage.COLLECT_MAPPING:
        summary = _dry_run_summary(job.name, selected) if dry_run else run_mapping_job(
            config_path,
            job.name,
            repository_root=repository_root,
            device=device,
            max_samples=max_samples,
            overwrite=overwrite,
        )
    elif selected == PipelineStage.INJECT:
        summary = _dry_run_summary(job.name, selected) if dry_run else run_injection_job(
            config_path,
            job.name,
            repository_root=repository_root,
            device=device,
            max_samples=max_samples,
            overwrite=overwrite,
            resume_from_run=resume_injection_from_run,
            telemetry_max_steps=telemetry_max_steps,
            output_path=injection_output_path,
        )
    elif selected == PipelineStage.LABEL:
        summary = _dry_run_summary(job.name, selected) if dry_run else run_label_job(
            config_path,
            job.name,
            repository_root=repository_root,
            device=device,
            batch_size=batch_size,
            overwrite=overwrite,
        )
    elif selected == PipelineStage.FEATURIZE:
        summary = _dry_run_summary(job.name, selected) if dry_run else _run_featurize(
            config_path,
            job,
            repository_root,
            max_samples,
            overwrite,
        )
    elif selected == PipelineStage.TRAIN_DETECTOR:
        summary = _dry_run_summary(job.name, selected) if dry_run else _run_detector(
            config_path,
            job,
            repository_root,
            overwrite,
        )
    elif selected == PipelineStage.REPORT:
        summary = _load_report(job)
    else:
        raise ValueError(f"Unsupported stage: {selected.value}")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def _run_profile(
    config_path: str | Path,
    job: Any,
    repository_root: str | Path,
    device: str,
    max_samples: int | None,
    overwrite: bool,
) -> dict[str, Any]:
    _require_output_policy([job.paths.profile_output], overwrite)
    return run_profile_job(
        config_path,
        job.name,
        repository_root=repository_root,
        device=device,
        max_samples=max_samples,
    )


def _run_featurize(
    config_path: str | Path,
    job: Any,
    repository_root: str | Path,
    max_samples: int | None,
    overwrite: bool,
) -> dict[str, Any]:
    feature_job = load_feature_job(
        config_path,
        job.name,
        repository_root=repository_root,
    )
    _require_output_policy(
        [
            feature_job.fit_output,
            feature_job.calibration_output,
            feature_job.test_output,
        ],
        overwrite,
    )
    return run_feature_job(
        config_path,
        job.name,
        repository_root=repository_root,
        max_samples=max_samples,
    )


def _run_detector(
    config_path: str | Path,
    job: Any,
    repository_root: str | Path,
    overwrite: bool,
) -> dict[str, Any]:
    output = job.paths.labeled_output.parent.parent / "output/metrics_summary.json"
    _require_output_policy([output], overwrite)
    return run_detector_job(
        config_path,
        job.name,
        repository_root=repository_root,
    )


def _load_report(job: Any) -> dict[str, Any]:
    path = job.paths.labeled_output.parent.parent / "output/metrics_summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"Detector report does not exist: {path}")
    with path.open("r", encoding="utf-8") as stream:
        metrics = json.load(stream)
    return {"job": job.name, "stage": PipelineStage.REPORT.value, "metrics": metrics}


def _require_output_policy(paths: list[Path], overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Stage outputs already exist; pass overwrite=True: {existing}"
        )


def _dry_run_summary(job_name: str, stage: PipelineStage) -> dict[str, Any]:
    return {"job": job_name, "stage": stage.value, "dry_run": True}
