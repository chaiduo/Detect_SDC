"""Ranger-style and Dr.DNA-style profiles and detection scores."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .monitor import ActivationTrace


ProfileKey = tuple[int, int]


def _key(step: int, layer: int) -> str:
    return f"s{step}_l{layer}"


def _parse_key(value: str) -> ProfileKey:
    step, layer = value.split("_", maxsplit=1)
    return int(step[1:]), int(layer[1:])


@dataclass(frozen=True)
class RangeProfile:
    """Per-step, per-layer activation bounds for Ranger-style detection."""

    monitored_layers: tuple[int, ...]
    max_steps: int
    lower: Mapping[ProfileKey, float]
    upper: Mapping[ProfileKey, float]

    def score(self, trace: ActivationTrace) -> float:
        """Return the largest normalized excursion outside profiled bounds."""
        if trace.has_non_finite():
            return float("inf")
        score = 0.0
        observed = 0
        for step in trace.steps:
            for layer in self.monitored_layers:
                key = (step, layer)
                if key not in self.lower or layer not in trace.vectors[step]:
                    continue
                values = trace.vector(step, layer)
                lower = float(self.lower[key])
                upper = float(self.upper[key])
                scale = max(upper - lower, abs(lower), abs(upper), 1e-12)
                excursion = max(
                    float(np.max(lower - values)) / scale,
                    float(np.max(values - upper)) / scale,
                    0.0,
                )
                score = max(score, excursion)
                observed += 1
        if observed == 0:
            raise ValueError("Trace has no entries covered by the range profile")
        return score

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "ranger_style",
            "monitored_layers": list(self.monitored_layers),
            "max_steps": self.max_steps,
            "lower": {_key(*key): value for key, value in self.lower.items()},
            "upper": {_key(*key): value for key, value in self.upper.items()},
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RangeProfile":
        return cls(
            monitored_layers=tuple(
                int(layer) for layer in value["monitored_layers"]
            ),
            max_steps=int(value["max_steps"]),
            lower={
                _parse_key(key): float(bound)
                for key, bound in value["lower"].items()
            },
            upper={
                _parse_key(key): float(bound)
                for key, bound in value["upper"].items()
            },
        )


class RangeProfiler:
    def __init__(
        self,
        *,
        monitored_layers: Sequence[int],
        max_steps: int,
    ) -> None:
        self.monitored_layers = tuple(
            sorted({int(layer) for layer in monitored_layers})
        )
        if not self.monitored_layers:
            raise ValueError("monitored_layers must not be empty")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.max_steps = int(max_steps)
        self._lower: dict[ProfileKey, float] = {}
        self._upper: dict[ProfileKey, float] = {}

    def add_trace(self, trace: ActivationTrace) -> None:
        for step in trace.steps:
            if step >= self.max_steps:
                continue
            for layer in self.monitored_layers:
                if layer not in trace.vectors[step]:
                    continue
                values = trace.vector(step, layer)
                finite = values[np.isfinite(values)]
                if not finite.size:
                    continue
                key = (step, layer)
                self._lower[key] = min(
                    self._lower.get(key, float("inf")),
                    float(finite.min()),
                )
                self._upper[key] = max(
                    self._upper.get(key, float("-inf")),
                    float(finite.max()),
                )

    def finalize(self) -> RangeProfile:
        if not self._lower:
            raise ValueError("Cannot finalize an empty range profile")
        if set(self._lower) != set(self._upper):
            raise AssertionError("Range profile bounds are inconsistent")
        return RangeProfile(
            monitored_layers=self.monitored_layers,
            max_steps=self.max_steps,
            lower=dict(self._lower),
            upper=dict(self._upper),
        )


@dataclass(frozen=True)
class DrDNAConfig:
    cohort_size: int = 64
    bins: int = 10
    strike_count: int = 3
    random_seed: int = 42
    lambda_individual: float = 1.0
    lambda_layer: float = 1.0
    lambda_extreme: float = 1.0

    def __post_init__(self) -> None:
        if self.cohort_size <= 0:
            raise ValueError("cohort_size must be positive")
        if self.bins <= 1:
            raise ValueError("bins must be greater than one")
        if self.strike_count <= 0:
            raise ValueError("strike_count must be positive")
        if min(
            self.lambda_individual,
            self.lambda_layer,
            self.lambda_extreme,
        ) < 0:
            raise ValueError("Dr.DNA score weights must be non-negative")


@dataclass(frozen=True)
class DrDNALayerProfile:
    site_indices: np.ndarray
    individual_edges: np.ndarray
    individual_counts: np.ndarray
    layer_edges: np.ndarray
    layer_counts: np.ndarray
    maximum_site_position: int
    minimum_site_position: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "site_indices": self.site_indices.tolist(),
            "individual_edges": self.individual_edges.tolist(),
            "individual_counts": self.individual_counts.tolist(),
            "layer_edges": self.layer_edges.tolist(),
            "layer_counts": self.layer_counts.tolist(),
            "maximum_site_position": self.maximum_site_position,
            "minimum_site_position": self.minimum_site_position,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DrDNALayerProfile":
        return cls(
            site_indices=np.asarray(value["site_indices"], dtype=np.int64),
            individual_edges=np.asarray(
                value["individual_edges"], dtype=np.float64
            ),
            individual_counts=np.asarray(
                value["individual_counts"], dtype=np.float64
            ),
            layer_edges=np.asarray(value["layer_edges"], dtype=np.float64),
            layer_counts=np.asarray(value["layer_counts"], dtype=np.float64),
            maximum_site_position=int(value["maximum_site_position"]),
            minimum_site_position=int(value["minimum_site_position"]),
        )


@dataclass(frozen=True)
class DrDNAProfile:
    monitored_layers: tuple[int, ...]
    max_steps: int
    config: DrDNAConfig
    layers: Mapping[ProfileKey, DrDNALayerProfile]
    baseline_cumulative: Mapping[ProfileKey, float]

    def score(self, trace: ActivationTrace) -> float:
        """Return the maximum three-strike margin over monitored steps."""
        if trace.has_non_finite():
            return float("inf")
        curves = self._cumulative_curves(trace)
        scores = []
        for step, curve in curves.items():
            margins = []
            for layer, cumulative in curve:
                baseline = float(self.baseline_cumulative[(step, layer)])
                margins.append(
                    (cumulative - baseline) / max(abs(baseline), 1e-12)
                )
            if not margins:
                continue
            strikes = min(self.config.strike_count, len(margins))
            scores.extend(
                min(margins[start : start + strikes])
                for start in range(len(margins) - strikes + 1)
            )
        if not scores:
            raise ValueError("Trace has no entries covered by the Dr.DNA profile")
        return float(max(scores))

    def _cumulative_curves(
        self,
        trace: ActivationTrace,
    ) -> dict[int, list[tuple[int, float]]]:
        curves: dict[int, list[tuple[int, float]]] = {}
        for step in trace.steps:
            cumulative = 0.0
            curve = []
            for layer in self.monitored_layers:
                key = (step, layer)
                layer_profile = self.layers.get(key)
                if layer_profile is None or layer not in trace.vectors[step]:
                    continue
                vector = trace.vector(step, layer)
                sampled = vector[layer_profile.site_indices]
                tau1 = _individual_dna_score(sampled, layer_profile)
                tau2 = _layer_dna_score(sampled, layer_profile)
                tau3 = _extreme_neuron_score(sampled, layer_profile)
                cumulative += (
                    self.config.lambda_individual * tau1
                    + self.config.lambda_layer * tau2
                    + self.config.lambda_extreme * tau3
                )
                curve.append((layer, cumulative))
            if curve:
                curves[step] = curve
        return curves

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "drdna_style",
            "monitored_layers": list(self.monitored_layers),
            "max_steps": self.max_steps,
            "config": asdict(self.config),
            "layers": {
                _key(*key): profile.to_dict()
                for key, profile in self.layers.items()
            },
            "baseline_cumulative": {
                _key(*key): value
                for key, value in self.baseline_cumulative.items()
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DrDNAProfile":
        return cls(
            monitored_layers=tuple(
                int(layer) for layer in value["monitored_layers"]
            ),
            max_steps=int(value["max_steps"]),
            config=DrDNAConfig(**value["config"]),
            layers={
                _parse_key(key): DrDNALayerProfile.from_dict(profile)
                for key, profile in value["layers"].items()
            },
            baseline_cumulative={
                _parse_key(key): float(score)
                for key, score in value["baseline_cumulative"].items()
            },
        )


class DrDNAProfiler:
    def __init__(
        self,
        *,
        monitored_layers: Sequence[int],
        max_steps: int,
        config: DrDNAConfig | None = None,
    ) -> None:
        self.monitored_layers = tuple(
            sorted({int(layer) for layer in monitored_layers})
        )
        if not self.monitored_layers:
            raise ValueError("monitored_layers must not be empty")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.max_steps = int(max_steps)
        self.config = config or DrDNAConfig()
        self._site_indices: dict[int, np.ndarray] = {}
        self._traces: list[dict[ProfileKey, np.ndarray]] = []

    def add_trace(self, trace: ActivationTrace) -> None:
        sampled_trace: dict[ProfileKey, np.ndarray] = {}
        for step in trace.steps:
            if step >= self.max_steps:
                continue
            for layer in self.monitored_layers:
                if layer not in trace.vectors[step]:
                    continue
                vector = trace.vector(step, layer).reshape(-1)
                if not np.isfinite(vector).all():
                    raise ValueError(
                        "Dr.DNA profiling requires fault-free finite values"
                    )
                indices = self._indices(layer, len(vector))
                sampled_trace[(step, layer)] = vector[indices].astype(
                    np.float64,
                    copy=True,
                )
        if sampled_trace:
            self._traces.append(sampled_trace)

    def finalize(self) -> DrDNAProfile:
        if not self._traces:
            raise ValueError("Cannot finalize an empty Dr.DNA profile")
        profiles: dict[ProfileKey, DrDNALayerProfile] = {}
        keys = sorted({key for trace in self._traces for key in trace})
        for key in keys:
            samples = [
                trace[key] for trace in self._traces if key in trace
            ]
            matrix = np.stack(samples, axis=0)
            site_count = matrix.shape[1]
            individual_edges = np.empty(
                (site_count, self.config.bins + 1),
                dtype=np.float64,
            )
            individual_counts = np.empty(
                (site_count, self.config.bins),
                dtype=np.float64,
            )
            for site in range(site_count):
                counts, edges = _histogram(
                    matrix[:, site],
                    self.config.bins,
                )
                individual_edges[site] = edges
                individual_counts[site] = counts
            layer_counts, layer_edges = _histogram(
                matrix.reshape(-1),
                self.config.bins,
            )
            means = matrix.mean(axis=0)
            profiles[key] = DrDNALayerProfile(
                site_indices=self._site_indices[key[1]].copy(),
                individual_edges=individual_edges,
                individual_counts=individual_counts,
                layer_edges=layer_edges,
                layer_counts=layer_counts,
                maximum_site_position=int(np.argmax(means)),
                minimum_site_position=int(np.argmin(means)),
            )

        preliminary = DrDNAProfile(
            monitored_layers=self.monitored_layers,
            max_steps=self.max_steps,
            config=self.config,
            layers=profiles,
            baseline_cumulative={},
        )
        cumulative_values: dict[ProfileKey, list[float]] = {
            key: [] for key in profiles
        }
        for sampled_trace in self._traces:
            trace = ActivationTrace(
                vectors=_restore_vectors(sampled_trace, profiles),
                max_steps=self.max_steps,
            )
            for step, curve in preliminary._cumulative_curves(trace).items():
                for layer, cumulative in curve:
                    cumulative_values[(step, layer)].append(cumulative)
        baseline = {
            key: float(np.mean(values))
            for key, values in cumulative_values.items()
            if values
        }
        if set(baseline) != set(profiles):
            raise AssertionError("Incomplete Dr.DNA baseline curve")
        return DrDNAProfile(
            monitored_layers=self.monitored_layers,
            max_steps=self.max_steps,
            config=self.config,
            layers=profiles,
            baseline_cumulative=baseline,
        )

    def _indices(self, layer: int, vector_size: int) -> np.ndarray:
        existing = self._site_indices.get(layer)
        if existing is not None:
            if existing[-1] >= vector_size:
                raise ValueError(
                    f"Layer {layer} activation width changed below "
                    f"profiled site {existing[-1]}"
                )
            return existing
        count = min(self.config.cohort_size, vector_size)
        generator = np.random.default_rng(
            self.config.random_seed + 104_729 * layer
        )
        indices = np.sort(
            generator.choice(vector_size, size=count, replace=False)
        ).astype(np.int64)
        self._site_indices[layer] = indices
        return indices


def _histogram(
    values: np.ndarray,
    bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if minimum == maximum:
        padding = max(abs(minimum) * 1e-6, 1e-6)
        minimum -= padding
        maximum += padding
    counts, edges = np.histogram(
        values,
        bins=bins,
        range=(minimum, maximum),
    )
    return counts.astype(np.float64), edges.astype(np.float64)


def _individual_dna_score(
    values: np.ndarray,
    profile: DrDNALayerProfile,
) -> float:
    scores = []
    for site, value in enumerate(values):
        if not np.isfinite(value):
            scores.append(1.0)
            continue
        edges = profile.individual_edges[site]
        counts = profile.individual_counts[site]
        index = int(np.searchsorted(edges, value, side="right") - 1)
        if value == edges[-1]:
            index = len(counts) - 1
        if index < 0 or index >= len(counts):
            scores.append(1.0)
            continue
        total = float(counts.sum())
        frequency = 0.0 if total <= 0 else float(counts[index] / total)
        scores.append(1.0 - frequency)
    return float(np.mean(scores)) if scores else 0.0


def _layer_dna_score(
    values: np.ndarray,
    profile: DrDNALayerProfile,
) -> float:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return 1.0
    edges = profile.layer_edges
    clipped = np.clip(finite, edges[0], np.nextafter(edges[-1], edges[0]))
    current, _ = np.histogram(clipped, bins=edges)
    baseline = profile.layer_counts
    if current.sum() <= 0 or baseline.sum() <= 0:
        return 1.0
    current_cdf = np.cumsum(current / current.sum())
    baseline_cdf = np.cumsum(baseline / baseline.sum())
    return float(np.mean(np.abs(current_cdf - baseline_cdf)))


def _extreme_neuron_score(
    values: np.ndarray,
    profile: DrDNALayerProfile,
) -> float:
    if not np.isfinite(values).all():
        return 1.0
    maximum_mismatch = int(
        int(np.argmax(values)) != profile.maximum_site_position
    )
    minimum_mismatch = int(
        int(np.argmin(values)) != profile.minimum_site_position
    )
    return 0.5 * (maximum_mismatch + minimum_mismatch)


def _restore_vectors(
    sampled_trace: Mapping[ProfileKey, np.ndarray],
    profiles: Mapping[ProfileKey, DrDNALayerProfile],
) -> dict[int, dict[int, np.ndarray]]:
    vectors: dict[int, dict[int, np.ndarray]] = {}
    for key, sampled in sampled_trace.items():
        step, layer = key
        indices = profiles[key].site_indices
        vector = np.zeros(int(indices[-1]) + 1, dtype=np.float32)
        vector[indices] = sampled
        vectors.setdefault(step, {})[layer] = vector
    return vectors
