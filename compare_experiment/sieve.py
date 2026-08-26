"""SIEVE trace scoring and comparison-specific detector training."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import xgboost as xgb

from detect_sdc.detector.xgboost import (
    XGBoostConfig,
    add_significant_sdc_target,
    get_feature_columns,
    prepare_features,
    train_binary_model,
)
from detect_sdc.features.extraction import FeatureSpec
from detect_sdc.splitting import split_by_group

from .monitor import ActivationTrace


class BoosterProbabilityAdapter:
    def __init__(self, booster: xgb.Booster) -> None:
        self.booster = booster

    def positive_probability(self, values: np.ndarray) -> float:
        result = self.booster.inplace_predict(
            values.reshape(1, -1),
            validate_features=False,
        )
        return float(result[0])


def train_comparison_detector(
    train_csv: str | Path,
    *,
    fit_orig_ids: Sequence[str],
    output_dir: str | Path,
    config: XGBoostConfig,
) -> tuple[BoosterProbabilityAdapter, tuple[str, ...]]:
    destination = Path(output_dir)
    model_path = destination / "detector.ubj"
    metadata_path = destination / "detector_metadata.json"
    if model_path.is_file() and metadata_path.is_file():
        booster = xgb.Booster()
        booster.load_model(model_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return BoosterProbabilityAdapter(booster), tuple(
            metadata["feature_columns"]
        )

    frame = add_significant_sdc_target(pd.read_csv(train_csv))
    fit_ids = {str(value) for value in fit_orig_ids}
    frame = frame[frame["orig_id"].astype(str).isin(fit_ids)].copy()
    if frame.empty:
        raise ValueError("No SIEVE training rows match the fit cohort")
    columns = get_feature_columns(frame)
    split = split_by_group(
        frame,
        group_column="orig_id",
        holdout_ratio=config.test_ratio,
        random_state=config.random_state,
    )
    model, training = train_binary_model(
        prepare_features(split.train, columns),
        split.train["significant_sdc_target"].astype(int),
        prepare_features(split.holdout, columns),
        split.holdout["significant_sdc_target"].astype(int),
        config=replace(config, verbose=False),
    )
    booster = model.get_booster()
    best_iteration = training["best_iteration"]
    if best_iteration is not None:
        booster = booster[: int(best_iteration) + 1]
    destination.mkdir(parents=True, exist_ok=True)
    booster.save_model(model_path)
    metadata_path.write_text(
        json.dumps(
            {
                "feature_columns": columns,
                "training": training,
                "config": asdict(config),
                "inference_tree_count": booster.num_boosted_rounds(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return BoosterProbabilityAdapter(booster), tuple(columns)


class SieveTraceScorer:
    def __init__(
        self,
        *,
        predictor: Any,
        detector: BoosterProbabilityAdapter,
        detector_feature_columns: Sequence[str],
        layer_pairs: Sequence[tuple[int, int]],
        projection_dim: int,
        projection_seed: int,
        max_steps: int,
        device: str,
    ) -> None:
        self.predictor = predictor.to(device).eval()
        self.detector = detector
        self.layer_pairs = tuple(
            (int(source), int(target)) for source, target in layer_pairs
        )
        self.monitored_layers = tuple(
            sorted({layer for pair in self.layer_pairs for layer in pair})
        )
        self.projection_dim = int(projection_dim)
        self.projection_seed = int(projection_seed)
        self.max_steps = int(max_steps)
        self.device = torch.device(device)
        self._projection: torch.Tensor | None = None
        self._src_layers = torch.tensor(
            [source for source, _ in self.layer_pairs],
            dtype=torch.long,
            device=self.device,
        )
        self._tgt_layers = torch.tensor(
            [target for _, target in self.layer_pairs],
            dtype=torch.long,
            device=self.device,
        )
        expected = FeatureSpec(
            selected_layer_pairs=self.layer_pairs,
            distance_pairs=self.layer_pairs,
            last_k_steps=self.max_steps,
            finite_only=True,
            step_window="prefix",
        ).feature_columns
        if tuple(detector_feature_columns) != expected:
            raise ValueError(
                "Comparison SIEVE detector feature columns do not match "
                "the online feature order"
            )

    @torch.no_grad()
    def score(self, trace: ActivationTrace) -> float:
        if trace.has_non_finite():
            return float("inf")
        metrics_by_step = []
        position = {
            layer: index
            for index, layer in enumerate(self.monitored_layers)
        }
        for step in trace.steps[: self.max_steps]:
            stacked = np.stack(
                [
                    trace.vector(step, layer)
                    for layer in self.monitored_layers
                ],
                axis=0,
            )
            values = torch.as_tensor(
                stacked,
                dtype=torch.float32,
                device=self.device,
            )
            projection = self._get_projection(values.shape[1])
            projected = values @ projection
            source = torch.stack(
                [
                    projected[position[source_layer]]
                    for source_layer, _ in self.layer_pairs
                ],
                dim=0,
            )
            target = torch.stack(
                [
                    projected[position[target_layer]]
                    for _, target_layer in self.layer_pairs
                ],
                dim=0,
            )
            prediction = self.predictor(
                source,
                self._src_layers,
                self._tgt_layers,
            ).float()
            metrics_by_step.append(_pair_metrics(prediction, target.float()))
        if not metrics_by_step:
            raise ValueError("Trace contains no monitored decode steps")
        features = _aggregate(metrics_by_step)
        vector = features.cpu().numpy().astype(np.float32, copy=False)
        vector = np.where(np.isfinite(vector), vector, np.nan)
        return self.detector.positive_probability(vector)

    def _get_projection(self, input_dim: int) -> torch.Tensor:
        if self._projection is not None:
            if self._projection.shape[0] != input_dim:
                raise ValueError("Activation width changed during scoring")
            return self._projection
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.projection_seed)
        random_matrix = torch.randn(
            input_dim,
            self.projection_dim,
            device=self.device,
            dtype=torch.float32,
            generator=generator,
        )
        self._projection, _ = torch.linalg.qr(random_matrix)
        return self._projection


def _pair_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    difference = prediction - target
    return torch.stack(
        (
            F.cosine_similarity(prediction, target, dim=1, eps=1e-12),
            prediction.mean(dim=1) - target.mean(dim=1),
            prediction.std(dim=1, unbiased=False)
            - target.std(dim=1, unbiased=False),
            torch.linalg.vector_norm(difference, ord=2, dim=1),
        ),
        dim=1,
    )


def _aggregate(metrics_by_step: Sequence[torch.Tensor]) -> torch.Tensor:
    values = torch.stack(tuple(metrics_by_step), dim=0)
    finite = torch.isfinite(values)
    counts = finite.sum(dim=0)
    means = torch.where(finite, values, 0.0).sum(dim=0) / counts.clamp_min(1)
    maxima = torch.where(finite, values, -torch.inf).max(dim=0).values
    minima = torch.where(finite, values, torch.inf).min(dim=0).values
    invalid = counts == 0
    statistics = torch.stack(
        (
            means.masked_fill(invalid, torch.nan),
            maxima.masked_fill(invalid, torch.nan),
            minima.masked_fill(invalid, torch.nan),
        ),
        dim=-1,
    )
    return torch.cat(
        (
            statistics[:, :3, :].reshape(-1),
            statistics[:, 3, :].reshape(-1),
        ),
        dim=0,
    )


def detector_config(
    experiment_config: Mapping[str, Any],
    model_key: str,
) -> XGBoostConfig:
    detector = experiment_config["detector"]["xgboost"]
    values = dict(detector["common"])
    values.update(detector["by_model"][model_key])
    return XGBoostConfig.from_mapping(values)
