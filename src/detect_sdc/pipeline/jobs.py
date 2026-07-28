"""Experiment job resolution shared by pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..config import load_yaml


@dataclass(frozen=True)
class StagePaths:
    profile_output: Path
    mapping_data: Path
    mapping_model: Path
    injected_output: Path
    labeled_output: Path


@dataclass(frozen=True)
class PipelineJob:
    name: str
    model_name: str
    dataset_name: str
    model_config_path: Path
    dataset_config_path: Path
    model_config: Mapping[str, Any]
    dataset_config: Mapping[str, Any]
    paths: StagePaths
    projection_dim: int
    projection_method: str
    profiler_seed: int
    mapping_training_config: Mapping[str, Any]
    injection_config: Mapping[str, Any]

    @property
    def max_samples(self) -> int | None:
        sampling = self.dataset_config.get("sampling", {})
        if not isinstance(sampling, Mapping):
            raise ValueError("dataset sampling must be a mapping")
        value = sampling.get("max_samples")
        return None if value is None else int(value)

    @property
    def max_new_tokens(self) -> int:
        generation = self.model_config.get("generation", {})
        if not isinstance(generation, Mapping):
            raise ValueError("model generation must be a mapping")
        return int(generation.get("max_new_tokens", 50))


def load_pipeline_job(
    config_path: str | Path,
    job_name: str,
    *,
    repository_root: str | Path,
) -> PipelineJob:
    root = Path(repository_root).resolve()
    experiment = load_yaml(config_path)
    execution = _mapping(experiment.get("execution"), "execution")
    jobs = _mapping(execution.get("jobs"), "execution.jobs")
    if job_name not in jobs:
        raise KeyError(f"Unknown pipeline job: {job_name}")
    job = _mapping(jobs[job_name], f"pipeline job {job_name}")

    model_name = _required_string(job, "model")
    dataset_name = _required_string(job, "dataset")
    if [model_name, dataset_name] not in experiment.get("matrix", []):
        raise ValueError(
            f"Pipeline job pair is not present in matrix: "
            f"{model_name}/{dataset_name}"
        )

    models = _mapping(experiment.get("models"), "models")
    datasets = _mapping(experiment.get("datasets"), "datasets")
    model_path = _resolve(root, str(models[model_name]))
    dataset_path = _resolve(root, str(datasets[dataset_name]))
    model_config = load_yaml(model_path)
    instrumentation = dict(
        _mapping(model_config.get("instrumentation"), "model instrumentation")
    )
    job_instrumentation = job.get("instrumentation", {})
    if not isinstance(job_instrumentation, Mapping):
        raise ValueError("pipeline job instrumentation must be a mapping")
    instrumentation.update(job_instrumentation)
    injection = dict(
        _mapping(model_config.get("injection"), "model injection")
    )
    job_injection = job.get("injection", {})
    if not isinstance(job_injection, Mapping):
        raise ValueError("pipeline job injection must be a mapping")
    injection.update(job_injection)
    mapping_training = dict(
        _mapping(
            model_config.get("mapping_training"),
            "model mapping_training",
        )
    )
    job_mapping_training = job.get("mapping_training", {})
    if not isinstance(job_mapping_training, Mapping):
        raise ValueError("pipeline job mapping_training must be a mapping")
    mapping_training.update(job_mapping_training)

    return PipelineJob(
        name=job_name,
        model_name=model_name,
        dataset_name=dataset_name,
        model_config_path=model_path,
        dataset_config_path=dataset_path,
        model_config=model_config,
        dataset_config=load_yaml(dataset_path),
        paths=StagePaths(
            profile_output=_resolve(root, _required_string(job, "profile_output")),
            mapping_data=_resolve(root, _required_string(job, "mapping_data")),
            mapping_model=_resolve(root, _required_string(job, "mapping_model")),
            injected_output=_resolve(root, _required_string(job, "injected_output")),
            labeled_output=_resolve(root, _required_string(job, "labeled_output")),
        ),
        projection_dim=int(instrumentation.get("projection_dim", 64)),
        projection_method=str(
            instrumentation.get("projection_method", "project")
        ),
        profiler_seed=int(instrumentation.get("seed", 42)),
        mapping_training_config=mapping_training,
        injection_config=injection,
    )


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a non-empty mapping")
    return value


def _required_string(mapping: Mapping[str, Any], key: str) -> str:
    value = str(mapping.get(key, "")).strip()
    if not value:
        raise ValueError(f"Pipeline job requires {key}")
    return value


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path.resolve()
