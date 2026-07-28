"""Interface implemented by each multimodal model family."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ModelAdapter(Protocol):
    @property
    def model(self) -> Any:
        """Return the underlying model for instrumentation."""

    def load(self, device: str) -> None:
        """Load model resources on the requested device."""

    def generate(
        self,
        question: str,
        image: Any,
        *,
        max_new_tokens: int,
    ) -> str:
        """Generate one deterministic answer."""

    def close(self) -> None:
        """Release hooks and model resources."""

