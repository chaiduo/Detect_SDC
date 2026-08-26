"""Configuration model for method-comparison experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from detect_sdc.config import load_yaml

from .profiles import DrDNAConfig


@dataclass(frozen=True)
class ComparisonConfig:
    source_config: Path
    layer_pairs: tuple[tuple[int, int], ...]
    max_steps: int
    calibration_ratio: float
    split_seed: int
    ranger_profile_fraction: float
    maximum_profile_samples: int | None
    drdna: DrDNAConfig
    target_fpr: float
    supplementary_fpr_budgets: tuple[float, ...]
    maximum_sdc: int | None
    maximum_non_sdc: int | None
    bootstrap_replicates: int
    bootstrap_seed: int

    @property
    def monitored_layers(self) -> tuple[int, ...]:
        return tuple(
            sorted({layer for pair in self.layer_pairs for layer in pair})
        )


def load_comparison_config(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> ComparisonConfig:
    source = Path(path).resolve()
    root = Path(repository_root).resolve()
    value = load_yaml(source)
    experiment = _mapping(value.get("experiment"), "experiment")
    monitoring = _mapping(value.get("monitoring"), "monitoring")
    cohorts = _mapping(value.get("cohorts"), "cohorts")
    drdna = _mapping(value.get("drdna"), "drdna")
    calibration = _mapping(value.get("calibration"), "calibration")
    evaluation = _mapping(value.get("evaluation"), "evaluation")

    source_config = Path(str(experiment["source_config"]))
    if not source_config.is_absolute():
        source_config = root / source_config
    pairs = tuple(
        (int(pair[0]), int(pair[1]))
        for pair in monitoring["layer_pairs"]
    )
    if not pairs or len(pairs) != len(set(pairs)):
        raise ValueError("monitoring.layer_pairs must be non-empty and unique")

    return ComparisonConfig(
        source_config=source_config.resolve(),
        layer_pairs=pairs,
        max_steps=int(monitoring["max_steps"]),
        calibration_ratio=float(
            cohorts["calibration_ratio_within_train"]
        ),
        split_seed=int(cohorts["split_seed"]),
        ranger_profile_fraction=float(
            cohorts["ranger_profile_fraction_of_fit"]
        ),
        maximum_profile_samples=_optional_int(
            cohorts.get("maximum_profile_samples")
        ),
        drdna=DrDNAConfig(**drdna),
        target_fpr=float(calibration["target_non_sdc_fpr"]),
        supplementary_fpr_budgets=tuple(
            float(item)
            for item in calibration["supplementary_fpr_budgets"]
        ),
        maximum_sdc=_optional_int(
            evaluation.get("maximum_sdc_per_model_dataset")
        ),
        maximum_non_sdc=_optional_int(
            evaluation.get("maximum_non_sdc_per_model_dataset")
        ),
        bootstrap_replicates=int(evaluation["bootstrap_replicates"]),
        bootstrap_seed=int(evaluation["bootstrap_seed"]),
    )


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)
