"""Typed validation for experiment matrix configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

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

        pipeline = data.get("pipeline")
        if not isinstance(pipeline, Mapping):
            raise ValueError("Experiment configuration requires a pipeline mapping")
        raw_stages = pipeline.get("stages")
        if not isinstance(raw_stages, list) or not raw_stages:
            raise ValueError("Pipeline stages must be a non-empty list")
        stages = tuple(PipelineStage(str(stage)) for stage in raw_stages)

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


def _string_mapping(value: Any, name: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a non-empty mapping")
    return {str(key): str(item) for key, item in value.items()}

