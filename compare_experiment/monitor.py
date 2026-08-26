"""Read-only activation collection for detector comparison experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class ActivationTrace:
    """Last-token decoder activations indexed by decode step and layer."""

    vectors: Mapping[int, Mapping[int, np.ndarray]]
    max_steps: int

    @property
    def steps(self) -> tuple[int, ...]:
        return tuple(sorted(self.vectors))

    def vector(self, step: int, layer: int) -> np.ndarray:
        return np.asarray(self.vectors[step][layer], dtype=np.float32)

    def has_non_finite(self) -> bool:
        return any(
            not np.isfinite(vector).all()
            for layers in self.vectors.values()
            for vector in layers.values()
        )


class OnlineActivationMonitor:
    """Collect selected decoder outputs without modifying model execution."""

    def __init__(
        self,
        model: Any,
        *,
        monitored_layers: Sequence[int],
        max_steps: int,
    ) -> None:
        layers = tuple(sorted({int(layer) for layer in monitored_layers}))
        if not layers:
            raise ValueError("monitored_layers must not be empty")
        if layers[0] < 0:
            raise ValueError("monitored_layers must be non-negative")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")

        self.model = model
        self.monitored_layers = layers
        self.max_steps = int(max_steps)
        self._handles: list[Any] = []
        self._step_handle: Any | None = None
        self._current_vectors: dict[int, torch.Tensor] = {}
        self._trace: dict[int, dict[int, np.ndarray]] = {}
        self.current_step = 0
        self.steps_processed = 0

    def register(self) -> None:
        if self._handles or self._step_handle is not None:
            raise RuntimeError("OnlineActivationMonitor is already registered")
        layers = self._get_layers()
        if self.monitored_layers[-1] >= len(layers):
            raise ValueError(
                f"Monitored layer exceeds model depth {len(layers)}: "
                f"{self.monitored_layers}"
            )
        for layer_idx in self.monitored_layers:
            handle = layers[layer_idx].self_attn.o_proj.register_forward_hook(
                self._make_layer_hook(layer_idx)
            )
            self._handles.append(handle)
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

    def start_sample(self) -> None:
        self.current_step = 0
        self.steps_processed = 0
        self._current_vectors.clear()
        self._trace.clear()

    def finish_sample(self) -> ActivationTrace:
        if self.steps_processed < self.max_steps and self._current_vectors:
            self._finalize_step()
        return ActivationTrace(
            vectors={
                step: dict(layer_vectors)
                for step, layer_vectors in self._trace.items()
            },
            max_steps=self.max_steps,
        )

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

    def _step_hook(self, _module: Any, _inputs: Any, output: Any) -> Any:
        if (
            0 < self.current_step <= self.max_steps
            and self._current_vectors
        ):
            self._finalize_step()
        self.current_step += 1
        return output

    def _finalize_step(self) -> None:
        missing = [
            layer
            for layer in self.monitored_layers
            if layer not in self._current_vectors
        ]
        if missing:
            raise RuntimeError(
                f"Missing monitored layers at decode step "
                f"{self.current_step}: {missing}"
            )
        step = self.steps_processed
        self._trace[step] = {
            layer: self._current_vectors[layer]
            .detach()
            .float()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
            for layer in self.monitored_layers
        }
        self._current_vectors.clear()
        self.steps_processed += 1

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
        if not isinstance(output, torch.Tensor):
            return None
        vector = output.detach()
        if vector.dim() == 3:
            vector = vector[0, -1, :]
        elif vector.dim() == 2:
            vector = vector[-1, :]
        elif vector.dim() != 1:
            return None
        return vector.reshape(-1)
