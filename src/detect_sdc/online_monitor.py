"""Optimized online feature monitoring for SIEVE deployment."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .features.extraction import FeatureSpec


ONLINE_MODES = ("step_hook", "monitor", "predictor", "sieve")
FEATURE_PROFILES = ("full", "cos_sim_mean")


class OnlineSieveMonitor:
    def __init__(
        self,
        model: Any,
        *,
        mode: str,
        layer_pairs: Sequence[tuple[int, int]],
        projection_dim: int,
        projection_seed: int,
        max_steps: int,
        predictor: Any | None = None,
        detector: Any | None = None,
        detector_feature_columns: Sequence[str] | None = None,
        feature_profile: str = "full",
    ) -> None:
        if mode not in ONLINE_MODES:
            raise ValueError(f"Unsupported online mode: {mode}")
        if feature_profile not in FEATURE_PROFILES:
            raise ValueError(
                f"Unsupported feature profile: {feature_profile}"
            )
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        pairs = tuple((int(src), int(tgt)) for src, tgt in layer_pairs)
        if not pairs or len(set(pairs)) != len(pairs):
            raise ValueError("layer_pairs must be non-empty and unique")
        if mode in {"predictor", "sieve"} and predictor is None:
            raise ValueError(f"{mode} mode requires a predictor")
        if mode == "sieve" and detector is None:
            raise ValueError("sieve mode requires a detector")

        self.model = model
        self.mode = mode
        self.layer_pairs = pairs
        self.monitored_layers = tuple(
            sorted({layer for pair in pairs for layer in pair})
        )
        self.projection_dim = int(projection_dim)
        self.projection_seed = int(projection_seed)
        self.max_steps = int(max_steps)
        self.predictor = predictor
        self.detector = detector
        self.detector_feature_columns = tuple(detector_feature_columns or ())
        self.feature_profile = feature_profile

        self._handles: list[Any] = []
        self._step_handle: Any | None = None
        self._projection: torch.Tensor | None = None
        self._pair_positions: tuple[tuple[int, int], ...] = ()
        self._src_layers: torch.Tensor | None = None
        self._tgt_layers: torch.Tensor | None = None
        self._sample_start: float | None = None
        self._current_vectors: dict[int, torch.Tensor] = {}
        self._step_metrics: list[torch.Tensor] = []
        self.current_step = 0
        self.steps_processed = 0
        self.prefill_complete_seconds: float | None = None
        self.detection_ready_seconds: float | None = None
        self.detection_after_prefill_seconds: float | None = None
        self.detector_probability: float | None = None
        self.detector_prediction: int | None = None
        self.feature_vector: np.ndarray | None = None

    def register(self) -> None:
        if self._step_handle is not None or self._handles:
            raise RuntimeError("OnlineSieveMonitor is already registered")
        if (
            self.mode == "sieve"
            and self.detector_feature_columns
            and self.detector_feature_columns != self.expected_feature_columns()
        ):
            raise ValueError(
                "Detector feature columns do not match online feature order"
            )
        layers = self._get_layers()
        if max(self.monitored_layers) >= len(layers):
            raise ValueError(
                f"Monitored layer exceeds model depth {len(layers)}: "
                f"{self.monitored_layers}"
            )

        if self.mode != "step_hook":
            input_dim = int(
                layers[self.monitored_layers[0]].self_attn.o_proj.in_features
            )
            device = next(self.model.parameters()).device
            generator = torch.Generator(device=device)
            generator.manual_seed(self.projection_seed)
            random_matrix = torch.randn(
                input_dim,
                self.projection_dim,
                device=device,
                dtype=torch.float32,
                generator=generator,
            )
            self._projection, _ = torch.linalg.qr(random_matrix)

            position = {
                layer: index
                for index, layer in enumerate(self.monitored_layers)
            }
            self._pair_positions = tuple(
                (position[src], position[tgt])
                for src, tgt in self.layer_pairs
            )
            for layer_idx in self.monitored_layers:
                handle = layers[layer_idx].self_attn.o_proj.register_forward_hook(
                    self._make_layer_hook(layer_idx)
                )
                self._handles.append(handle)

            self._src_layers = torch.tensor(
                [src for src, _ in self.layer_pairs],
                dtype=torch.long,
                device=device,
            )
            self._tgt_layers = torch.tensor(
                [tgt for _, tgt in self.layer_pairs],
                dtype=torch.long,
                device=device,
            )

        self._step_handle = self.model.lm_head.register_forward_hook(
            self._step_hook
        )

    def unregister(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        if self._step_handle is not None:
            self._step_handle.remove()
            self._step_handle = None
        self._current_vectors.clear()

    def start_sample(self, start_time: float | None = None) -> None:
        self.current_step = 0
        self.steps_processed = 0
        self._current_vectors.clear()
        self._step_metrics.clear()
        self.prefill_complete_seconds = None
        self.detection_ready_seconds = None
        self.detection_after_prefill_seconds = None
        self.detector_probability = None
        self.detector_prediction = None
        self.feature_vector = None
        self._sample_start = (
            time.perf_counter() if start_time is None else float(start_time)
        )

    def finish_sample(self) -> None:
        if (
            self.mode != "step_hook"
            and self.steps_processed < self.max_steps
            and self._current_vectors
        ):
            self._finalize_step()
        if self.detection_ready_seconds is not None:
            return
        if self.mode == "monitor" and self.steps_processed:
            self._mark_ready()
        elif self.mode in {"predictor", "sieve"} and self._step_metrics:
            self._complete_prediction()

    def _step_hook(self, _module: Any, _inputs: Any, output: Any) -> Any:
        if self.current_step == 0 and self._sample_start is not None:
            self.prefill_complete_seconds = (
                time.perf_counter() - self._sample_start
            )
        if (
            self.mode != "step_hook"
            and 0 < self.current_step <= self.max_steps
            and self._current_vectors
        ):
            self._finalize_step()
        self.current_step += 1
        return output

    def _make_layer_hook(self, layer_idx: int) -> Any:
        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            if (
                self.current_step == 0
                or self.current_step > self.max_steps
                or self.steps_processed >= self.max_steps
            ):
                return output
            vector = self._to_vector(output)
            if vector is not None:
                self._current_vectors[layer_idx] = vector
            return output

        return hook

    def _finalize_step(self) -> None:
        if self.steps_processed >= self.max_steps:
            self._current_vectors.clear()
            return
        missing = [
            layer
            for layer in self.monitored_layers
            if layer not in self._current_vectors
        ]
        if missing:
            raise RuntimeError(
                f"Online monitor is missing layers at step "
                f"{self.current_step}: {missing}"
            )
        if self._projection is None:
            raise RuntimeError("Projection matrix is not initialized")

        stacked = torch.stack(
            [self._current_vectors[layer] for layer in self.monitored_layers],
            dim=0,
        )
        projected = stacked @ self._projection
        self._current_vectors.clear()
        self.steps_processed += 1

        if self.mode == "monitor":
            if self.steps_processed == self.max_steps:
                self._mark_ready()
            return

        if self.predictor is None:
            raise RuntimeError("Predictor is not initialized")
        source = torch.stack(
            [projected[src] for src, _ in self._pair_positions],
            dim=0,
        )
        target = torch.stack(
            [projected[tgt] for _, tgt in self._pair_positions],
            dim=0,
        )
        prediction = self.predictor(
            source,
            self._src_layers,
            self._tgt_layers,
        )
        prediction = prediction.float()
        target = target.float()
        if self.feature_profile == "cos_sim_mean":
            metrics = F.cosine_similarity(
                prediction,
                target,
                dim=1,
                eps=1e-12,
            )
        else:
            metrics = self._pair_metrics(prediction, target)
        self._step_metrics.append(metrics)
        if self.steps_processed == self.max_steps:
            self._complete_prediction()

    def _complete_prediction(self) -> None:
        if self.feature_vector is not None:
            return
        features = self._aggregate_features(self._step_metrics)
        self.feature_vector = features.detach().cpu().numpy().astype(
            np.float32,
            copy=False,
        )
        finite = np.isfinite(self.feature_vector)
        self.feature_vector = np.where(
            finite,
            self.feature_vector,
            np.nan,
        )
        if self.mode == "sieve":
            probability = self.detector.predict_proba(
                self.feature_vector.reshape(1, -1)
            )[0]
            self.detector_probability = float(probability[1])
            self.detector_prediction = int(np.argmax(probability))
        self._mark_ready()

    def _aggregate_features(
        self,
        metrics_by_step: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        values = torch.stack(tuple(metrics_by_step), dim=0)
        if self.feature_profile == "cos_sim_mean":
            finite = torch.isfinite(values)
            counts = finite.sum(dim=0)
            means = (
                torch.where(finite, values, 0.0).sum(dim=0)
                / counts.clamp_min(1)
            )
            return means.masked_fill(counts == 0, torch.nan)

        finite = torch.isfinite(values)
        counts = finite.sum(dim=0)
        means = torch.where(finite, values, 0.0).sum(dim=0) / counts.clamp_min(1)
        maxima = torch.where(finite, values, -torch.inf).max(dim=0).values
        minima = torch.where(finite, values, torch.inf).min(dim=0).values
        invalid = counts == 0
        means = means.masked_fill(invalid, torch.nan)
        maxima = maxima.masked_fill(invalid, torch.nan)
        minima = minima.masked_fill(invalid, torch.nan)
        statistics = torch.stack((means, maxima, minima), dim=-1)

        base = statistics[:, :3, :].reshape(-1)
        distance = statistics[:, 3, :].reshape(-1)
        return torch.cat((base, distance), dim=0)

    @staticmethod
    def _pair_metrics(
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        difference = prediction - target
        mean_difference = prediction.mean(dim=1) - target.mean(dim=1)
        std_difference = (
            prediction.std(dim=1, unbiased=False)
            - target.std(dim=1, unbiased=False)
        )
        cosine = F.cosine_similarity(
            prediction,
            target,
            dim=1,
            eps=1e-12,
        )
        l2_distance = torch.linalg.vector_norm(difference, ord=2, dim=1)
        return torch.stack(
            (cosine, mean_difference, std_difference, l2_distance),
            dim=1,
        )

    def expected_feature_columns(self) -> tuple[str, ...]:
        if self.feature_profile == "cos_sim_mean":
            return tuple(
                f"cos_sim_mean_p{src}_{tgt}"
                for src, tgt in self.layer_pairs
            )
        spec = FeatureSpec(
            selected_layer_pairs=self.layer_pairs,
            distance_pairs=self.layer_pairs,
            last_k_steps=self.max_steps,
            finite_only=True,
            step_window="prefix",
        )
        return spec.feature_columns

    def _mark_ready(self) -> None:
        if self._sample_start is not None:
            self.detection_ready_seconds = (
                time.perf_counter() - self._sample_start
            )
            if self.prefill_complete_seconds is not None:
                self.detection_after_prefill_seconds = (
                    self.detection_ready_seconds
                    - self.prefill_complete_seconds
                )

    def _get_layers(self) -> Any:
        language_model = getattr(self.model, "language_model", None)
        if language_model is None:
            inner = getattr(self.model, "model", None)
            language_model = (
                getattr(inner, "language_model", None)
                if inner is not None
                else None
            )
        if language_model is not None:
            layers = getattr(language_model, "layers", None)
            if layers is not None:
                return layers
        inner = getattr(self.model, "model", None)
        layers = getattr(inner, "layers", None)
        if layers is None:
            raise TypeError("Cannot locate Transformer layers")
        return layers

    @staticmethod
    def _to_vector(output: Any) -> torch.Tensor | None:
        if output is None:
            return None
        if isinstance(output, (tuple, list)):
            if not output:
                return None
            output = output[0]
        vector = output.detach().to(dtype=torch.float32)
        if vector.dim() == 3:
            vector = vector[0, -1, :]
        elif vector.dim() == 2:
            vector = vector[-1, :]
        elif vector.dim() != 1:
            return None
        return vector.reshape(-1)
