"""Canonical conversion from labeled telemetry records to tabular features."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np


LayerPair = tuple[int, int]
BASE_METRICS = ("cos_sim", "mean_diff", "std_diff")
STATISTICS = ("mean", "max", "min")


class SampleSkipped(ValueError):
    """Raised when an input sample cannot provide a supervised feature row."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class FeatureSpec:
    selected_layer_pairs: tuple[LayerPair, ...]
    distance_pairs: tuple[LayerPair, ...]
    last_k_steps: int = 50
    finite_only: bool = True
    step_window: str = "suffix"

    def __post_init__(self) -> None:
        if not self.selected_layer_pairs:
            raise ValueError("selected_layer_pairs must not be empty")
        if not self.distance_pairs:
            raise ValueError("distance_pairs must not be empty")
        if self.last_k_steps <= 0:
            raise ValueError("last_k_steps must be positive")
        if self.step_window not in {"prefix", "suffix"}:
            raise ValueError("step_window must be 'prefix' or 'suffix'")

        for name, pairs in (
            ("selected_layer_pairs", self.selected_layer_pairs),
            ("distance_pairs", self.distance_pairs),
        ):
            if len(set(pairs)) != len(pairs):
                raise ValueError(f"{name} contains duplicate layer pairs")
            if any(src < 0 or tgt < 0 for src, tgt in pairs):
                raise ValueError(f"{name} contains a negative layer index")

    @property
    def feature_columns(self) -> tuple[str, ...]:
        columns = []
        for pair in self.selected_layer_pairs:
            pair_key = _pair_key(pair)
            for metric in BASE_METRICS:
                columns.extend(f"{metric}_{stat}_{pair_key}" for stat in STATISTICS)
        for pair in self.distance_pairs:
            pair_key = _pair_key(pair)
            columns.extend(f"l2_distance_{stat}_{pair_key}" for stat in STATISTICS)
        return tuple(columns)


def build_supervision(sample: Mapping[str, Any]) -> tuple[int, int, int]:
    """Return canonical ternary label, significance, and binary target."""
    if "pred_answer" not in sample or "clean_answer" not in sample:
        raise SampleSkipped("missing_answer")
    if "significance" not in sample:
        raise SampleSkipped("missing_significance")

    try:
        significance = int(sample["significance"])
    except (TypeError, ValueError, OverflowError) as error:
        raise SampleSkipped("invalid_significance") from error

    pred_answer = str(sample["pred_answer"])
    clean_answer = str(sample["clean_answer"])
    answers_identical = pred_answer == clean_answer
    if answers_identical:
        significance = 0

    if significance not in (0, 1, 2):
        raise SampleSkipped("invalid_significance")

    if answers_identical:
        label = 0
    elif significance == 2:
        label = 2
    else:
        label = 1
    target = int(not answers_identical and significance == 2)
    return label, significance, target


def stable_sample_uid(sample: Mapping[str, Any], *, namespace: str) -> str:
    """Build an order-independent identity from the source sample and fault."""
    existing = sample.get("sample_uid")
    if existing is not None and str(existing).strip():
        return str(existing)
    if not namespace.strip():
        raise ValueError("sample UID namespace must not be empty")

    orig_id = _orig_id(sample)
    identity = {
        "namespace": namespace,
        "orig_id": str(orig_id),
        "source_file": sample.get("source_file"),
        "row_idx": sample.get("row_idx"),
        "fault": sample.get("fault"),
    }
    payload = json.dumps(
        identity,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:20]
    return f"{namespace}:{orig_id}:{digest}"


def extract_feature_row(
    sample: Mapping[str, Any],
    *,
    spec: FeatureSpec,
    uid_namespace: str,
) -> dict[str, Any]:
    records = sample.get("mean_std_cos", {}).get("records", [])
    if not isinstance(records, list) or not records:
        raise SampleSkipped("missing_telemetry")

    label, significance, target = build_supervision(sample)
    orig_id = _orig_id(sample)
    step_pair_map = _build_step_pair_map(records)
    all_steps = sorted(step_pair_map)
    if not all_steps:
        raise SampleSkipped("missing_telemetry")
    if spec.step_window == "prefix":
        window_steps = all_steps[: spec.last_k_steps]
    else:
        window_steps = all_steps[-spec.last_k_steps :]

    features: dict[str, float] = {}
    for pair in spec.selected_layer_pairs:
        for metric in BASE_METRICS:
            values = _collect_values(step_pair_map, window_steps, pair, metric)
            features.update(
                _aggregate(
                    values,
                    prefix=f"{metric}_{_pair_key(pair)}",
                    finite_only=spec.finite_only,
                )
            )
    for pair in spec.distance_pairs:
        values = _collect_values(step_pair_map, window_steps, pair, "l2_distance")
        features.update(
            _aggregate(
                values,
                prefix=f"l2_distance_{_pair_key(pair)}",
                finite_only=spec.finite_only,
            )
        )

    return {
        "orig_id": orig_id,
        "sample_uid": stable_sample_uid(sample, namespace=uid_namespace),
        "total_steps": len(all_steps),
        "last_k_steps": spec.last_k_steps,
        "num_steps_used": len(window_steps),
        **features,
        "significance": significance,
        "label": label,
        "significant_sdc_target": target,
    }


def iter_json_samples(
    path: str | Path,
    *,
    max_samples: int | None = None,
) -> Iterator[Mapping[str, Any]]:
    """Stream JSONL records without loading the complete input."""
    if max_samples is not None and max_samples < 0:
        raise ValueError("max_samples must be non-negative or None")
    input_path = Path(path)

    with input_path.open("r", encoding="utf-8-sig") as stream:
        yielded = 0
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            if max_samples is not None and yielded >= max_samples:
                return
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSONL record at {input_path}:{line_number}"
                ) from error
            yield _require_mapping(sample, input_path, line_number)
            yielded += 1


def _orig_id(sample: Mapping[str, Any]) -> Any:
    orig_id = sample.get("orig_id", sample.get("id"))
    if orig_id is None or not str(orig_id).strip():
        raise ValueError("sample is missing a stable orig_id/id")
    return orig_id


def _build_step_pair_map(
    records: Sequence[Mapping[str, Any]],
) -> dict[int, dict[LayerPair, Mapping[str, Any]]]:
    step_pair_map: dict[int, dict[LayerPair, Mapping[str, Any]]] = {}
    for record in records:
        try:
            step = int(record["step"])
            pair = (int(record["src_layer"]), int(record["tgt_layer"]))
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError("telemetry record has invalid step or layer fields") from error
        step_pair_map.setdefault(step, {})[pair] = record
    return step_pair_map


def _collect_values(
    step_pair_map: Mapping[int, Mapping[LayerPair, Mapping[str, Any]]],
    steps: Iterable[int],
    pair: LayerPair,
    metric: str,
) -> list[Any]:
    values = []
    for step in steps:
        record = step_pair_map.get(step, {}).get(pair)
        if record is not None and metric in record:
            values.append(record[metric])
    return values


def _aggregate(
    values: Sequence[Any],
    *,
    prefix: str,
    finite_only: bool,
) -> dict[str, float]:
    if values:
        array = np.asarray(values, dtype=float)
        if finite_only:
            array = array[np.isfinite(array)]
    else:
        array = np.asarray([], dtype=float)

    if not array.size:
        mean = maximum = minimum = np.nan
    else:
        mean = float(np.mean(array))
        maximum = float(np.max(array))
        minimum = float(np.min(array))
    return {
        f"{prefix.replace('_p', '_mean_p')}": mean,
        f"{prefix.replace('_p', '_max_p')}": maximum,
        f"{prefix.replace('_p', '_min_p')}": minimum,
    }


def _pair_key(pair: LayerPair) -> str:
    return f"p{pair[0]}_{pair[1]}"


def _require_mapping(
    value: Any,
    path: Path,
    line_number: int | None = None,
) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    location = f"{path}:{line_number}" if line_number is not None else str(path)
    raise ValueError(f"Expected a JSON object at {location}")
