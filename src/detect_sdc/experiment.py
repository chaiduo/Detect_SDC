"""Typed validation for experiment matrix configuration."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .adapters.registry import import_symbol
from .config import load_yaml
from .pipeline import PipelineStage


@dataclass(frozen=True)
class ExperimentMatrix:
    name: str
    models: Mapping[str, str]
    datasets: Mapping[str, str]
    matrix: tuple[tuple[str, str], ...]
    stages: tuple[PipelineStage, ...]

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ExperimentMatrix":
        name = str(data.get("name", "")).strip()
        if not name:
            raise ValueError("Experiment configuration requires a name")

        models = _string_mapping(data.get("models"), "models")
        datasets = _string_mapping(data.get("datasets"), "datasets")

        raw_matrix = data.get("matrix")
        if not isinstance(raw_matrix, list) or not raw_matrix:
            raise ValueError("Experiment matrix must be a non-empty list")

        matrix = []
        for entry in raw_matrix:
            if not isinstance(entry, list) or len(entry) != 2:
                raise ValueError(f"Matrix entry must be [model, dataset]: {entry!r}")
            model_name, dataset_name = map(str, entry)
            if model_name not in models:
                raise ValueError(f"Unknown model in matrix: {model_name}")
            if dataset_name not in datasets:
                raise ValueError(f"Unknown dataset in matrix: {dataset_name}")
            matrix.append((model_name, dataset_name))
        if len(set(matrix)) != len(matrix):
            raise ValueError("Experiment matrix contains duplicate pairs")

        pipeline = data.get("pipeline")
        if not isinstance(pipeline, Mapping):
            raise ValueError("Experiment configuration requires a pipeline mapping")
        raw_stages = pipeline.get("stages")
        if not isinstance(raw_stages, list) or not raw_stages:
            raise ValueError("Pipeline stages must be a non-empty list")
        stages = tuple(PipelineStage(str(stage)) for stage in raw_stages)
        if len(set(stages)) != len(stages):
            raise ValueError("Pipeline stages contain duplicates")

        return cls(
            name=name,
            models=models,
            datasets=datasets,
            matrix=tuple(matrix),
            stages=stages,
        )

    def validate_references(self, repository_root: str | Path) -> None:
        root = Path(repository_root)
        missing = []
        for path in (*self.models.values(), *self.datasets.values()):
            resolved = Path(path)
            if not resolved.is_absolute():
                resolved = root / resolved
            if not resolved.is_file():
                missing.append(str(resolved))
        if missing:
            raise FileNotFoundError(f"Missing referenced configuration files: {missing}")


def load_experiment(path: str | Path) -> ExperimentMatrix:
    return ExperimentMatrix.from_mapping(load_yaml(path))


def validate_experiment_configuration(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Validate matrix, adapters, jobs, paths, and mapping contracts."""

    from .pipeline.jobs import load_pipeline_job

    root = Path(repository_root).resolve()
    config_path = Path(path).resolve()
    experiment = load_experiment(config_path)
    experiment.validate_references(root)
    raw = load_yaml(config_path)
    execution = _required_mapping(raw.get("execution"), "execution")
    jobs = _required_mapping(execution.get("jobs"), "execution.jobs")

    expected_pairs = set(experiment.matrix)
    jobs_by_pair: dict[tuple[str, str], list[str]] = {
        pair: [] for pair in expected_pairs
    }
    loaded_jobs = []
    for job_name in sorted(jobs):
        job = load_pipeline_job(
            config_path,
            str(job_name),
            repository_root=root,
        )
        pair = (job.model_name, job.dataset_name)
        jobs_by_pair.setdefault(pair, []).append(job.name)
        _validate_pipeline_job(job)
        loaded_jobs.append(job)

    extras = set(jobs_by_pair) - expected_pairs
    if extras:
        raise ValueError(
            f"Execution jobs contain pairs outside the matrix: {sorted(extras)}"
        )
    invalid_counts = {
        f"{model}/{dataset}": names
        for (model, dataset), names in jobs_by_pair.items()
        if len(names) != 1
    }
    if invalid_counts:
        raise ValueError(
            "Every matrix pair requires exactly one execution job: "
            f"{invalid_counts}"
        )
    _validate_unique_outputs(loaded_jobs)
    return {
        "name": experiment.name,
        "pairs": len(experiment.matrix),
        "stages": len(experiment.stages),
        "jobs": len(loaded_jobs),
    }


def _string_mapping(value: Any, name: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a non-empty mapping")
    return {str(key): str(item) for key, item in value.items()}


def _validate_pipeline_job(job: Any) -> None:
    _validate_adapter(job.model_config, "model")
    _validate_adapter(job.dataset_config, "dataset")
    _validate_input_paths(job)

    if job.max_samples is not None and job.max_samples <= 0:
        raise ValueError(f"{job.name}: sampling.max_samples must be positive")
    if job.max_new_tokens <= 0:
        raise ValueError(f"{job.name}: generation.max_new_tokens must be positive")
    if job.projection_dim <= 0:
        raise ValueError(f"{job.name}: projection_dim must be positive")
    if job.projection_method not in {"project", "max", "min", "mean"}:
        raise ValueError(
            f"{job.name}: unsupported projection_method "
            f"{job.projection_method!r}"
        )

    instrumentation = _required_mapping(
        job.model_config.get("instrumentation"),
        f"{job.name}: instrumentation",
    )
    layer_count = _positive_int(
        instrumentation.get("layer_count"),
        f"{job.name}: instrumentation.layer_count",
    )
    injection = job.injection_config
    mapping_path = _required_text(
        injection,
        "mapping_class",
        f"{job.name}: injection",
    )
    mapping_class = import_symbol(mapping_path)
    if not inspect.isclass(mapping_class):
        raise TypeError(f"{job.name}: mapping_class must resolve to a class")
    mapping_kwargs = _required_mapping(
        injection.get("mapping_kwargs"),
        f"{job.name}: injection.mapping_kwargs",
    )
    _bind_signature(
        mapping_class,
        mapping_kwargs,
        f"{job.name}: injection.mapping_kwargs",
    )
    x_dim = _positive_int(
        mapping_kwargs.get("x_dim"),
        f"{job.name}: injection.mapping_kwargs.x_dim",
    )
    num_layers = _positive_int(
        mapping_kwargs.get("num_layers"),
        f"{job.name}: injection.mapping_kwargs.num_layers",
    )
    if x_dim != job.projection_dim:
        raise ValueError(
            f"{job.name}: mapping x_dim ({x_dim}) must equal "
            f"projection_dim ({job.projection_dim})"
        )
    if num_layers != layer_count:
        raise ValueError(
            f"{job.name}: mapping num_layers ({num_layers}) must equal "
            f"instrumentation.layer_count ({layer_count})"
        )

    fault_runs = _nonnegative_int(
        injection.get("fault_runs"),
        f"{job.name}: injection.fault_runs",
    )
    retained_runs = _nonnegative_int(
        injection.get("retain_all_fault_runs"),
        f"{job.name}: injection.retain_all_fault_runs",
    )
    if retained_runs > fault_runs:
        raise ValueError(
            f"{job.name}: retain_all_fault_runs cannot exceed fault_runs"
        )
    _positive_int(
        injection.get("num_bits"),
        f"{job.name}: injection.num_bits",
    )
    _integer(injection.get("seed"), f"{job.name}: injection.seed")

    training = job.mapping_training_config
    trainer_path = _required_text(
        training,
        "trainer",
        f"{job.name}: mapping_training",
    )
    trainer = import_symbol(trainer_path)
    if not callable(trainer):
        raise TypeError(f"{job.name}: mapping trainer must be callable")
    trainer_kwargs = _required_mapping(
        training.get("kwargs"),
        f"{job.name}: mapping_training.kwargs",
    )
    _validate_training_values(job.name, trainer_kwargs)
    _bind_signature(
        trainer,
        {
            "jsonl_path": "mapping.jsonl",
            "save_best_path": "mapping.pt",
            "device": "cpu",
            "model_kwargs": dict(mapping_kwargs),
            **trainer_kwargs,
        },
        f"{job.name}: mapping_training.kwargs",
    )
    _validate_stage_paths(job)


def _validate_adapter(config: Mapping[str, Any], kind: str) -> None:
    dotted_path = _required_text(config, "adapter", f"{kind} config")
    adapter_class = import_symbol(dotted_path)
    if not inspect.isclass(adapter_class):
        raise TypeError(f"{kind} adapter must resolve to a class")
    if not callable(getattr(adapter_class, "from_config", None)):
        raise TypeError(f"{dotted_path} must define from_config(config)")


def _validate_input_paths(job: Any) -> None:
    values = {
        "model_path": job.model_config.get("model_path"),
        "source_path": job.model_config.get("source_path"),
    }
    dataset_paths = _required_mapping(
        job.dataset_config.get("paths"),
        f"{job.name}: dataset paths",
    )
    values.update(
        {f"dataset.paths.{key}": value for key, value in dataset_paths.items()}
    )
    missing = [
        f"{name}={value}"
        for name, value in values.items()
        if value is not None and not Path(str(value)).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"{job.name}: configured input paths do not exist: {missing}"
        )


def _validate_training_values(
    job_name: str,
    kwargs: Mapping[str, Any],
) -> None:
    for key in ("batch_size", "epochs", "early_stop_patience"):
        _positive_int(kwargs.get(key), f"{job_name}: mapping_training.{key}")
    _nonnegative_int(
        kwargs.get("num_workers"),
        f"{job_name}: mapping_training.num_workers",
    )
    for key in ("scheduler_patience",):
        _nonnegative_int(
            kwargs.get(key),
            f"{job_name}: mapping_training.{key}",
        )
    strategy = str(kwargs.get("split_strategy", ""))
    if strategy not in {"sequential", "random", "partition"}:
        raise ValueError(
            f"{job_name}: unsupported mapping split_strategy {strategy!r}"
        )
    for key in ("valid_ratio", "test_ratio", "test_ratio_in_train"):
        value = kwargs.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"{job_name}: mapping_training.{key} must be numeric")
        if not 0.0 < float(value) < 1.0:
            raise ValueError(
                f"{job_name}: mapping_training.{key} must be between 0 and 1"
            )
    cosine_weight = kwargs.get("cosine_weight")
    if (
        not isinstance(cosine_weight, (int, float))
        or isinstance(cosine_weight, bool)
        or float(cosine_weight) < 0
    ):
        raise ValueError(
            f"{job_name}: mapping_training.cosine_weight must be non-negative"
        )
    for key in ("lr", "weight_decay", "scheduler_factor", "min_lr"):
        value = kwargs.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"{job_name}: mapping_training.{key} must be numeric")
        if float(value) < 0:
            raise ValueError(
                f"{job_name}: mapping_training.{key} must be non-negative"
            )
    for key in (
        "scheduler_enabled",
        "pin_memory",
        "persistent_workers",
    ):
        if not isinstance(kwargs.get(key), bool):
            raise TypeError(
                f"{job_name}: mapping_training.{key} must be boolean"
            )
    if kwargs.get("final_metrics") not in {"loss", "detailed"}:
        raise ValueError(
            f"{job_name}: mapping_training.final_metrics must be loss or detailed"
        )
    _integer(kwargs.get("seed"), f"{job_name}: mapping_training.seed")


def _validate_stage_paths(job: Any) -> None:
    expected_names = {
        "profile_output": "profile.json",
        "mapping_data": "mapping.jsonl",
        "injected_output": "injection.jsonl",
        "labeled_output": "labels.jsonl",
    }
    for name, expected_name in expected_names.items():
        path = getattr(job.paths, name)
        if path.name != expected_name:
            raise ValueError(
                f"{job.name}: {name} must use canonical filename "
                f"{expected_name}, got {path}"
            )
    if job.paths.mapping_model.suffix != ".pt":
        raise ValueError(
            f"{job.name}: mapping_model must use .pt, "
            f"got {job.paths.mapping_model}"
        )


def _validate_unique_outputs(jobs: list[Any]) -> None:
    owners: dict[Path, str] = {}
    for job in jobs:
        for name in (
            "profile_output",
            "mapping_data",
            "mapping_model",
            "injected_output",
            "labeled_output",
        ):
            path = getattr(job.paths, name)
            owner = owners.setdefault(path, f"{job.name}.{name}")
            if owner != f"{job.name}.{name}":
                raise ValueError(
                    f"Pipeline outputs collide: {owner} and {job.name}.{name}"
                )


def _bind_signature(
    target: Any,
    kwargs: Mapping[str, Any],
    name: str,
) -> None:
    try:
        inspect.signature(target).bind(**dict(kwargs))
    except TypeError as error:
        raise TypeError(f"{name} does not match callable signature: {error}") from error


def _required_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a non-empty mapping")
    return value


def _required_text(
    mapping: Mapping[str, Any],
    key: str,
    name: str,
) -> str:
    value = str(mapping.get(key, "")).strip()
    if not value:
        raise ValueError(f"{name} requires {key}")
    return value


def _integer(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    return value


def _positive_int(value: Any, name: str) -> int:
    parsed = _integer(value, name)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _nonnegative_int(value: Any, name: str) -> int:
    parsed = _integer(value, name)
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative")
    return parsed
