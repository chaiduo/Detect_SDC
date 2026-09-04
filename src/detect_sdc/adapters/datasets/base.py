"""Interface implemented by each benchmark dataset."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol, runtime_checkable


@dataclass(frozen=True)
class DatasetSample:
    orig_id: str
    semantic_group_id: str
    question: str
    ground_truth: str | None
    image: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.orig_id:
            raise ValueError("orig_id must not be empty")
        if not self.semantic_group_id:
            raise ValueError("semantic_group_id must not be empty")
        if not self.question:
            raise ValueError("question must not be empty")


@runtime_checkable
class DatasetAdapter(Protocol):
    @property
    def name(self) -> str:
        """Return the stable dataset identifier."""

    def iter_samples(self, max_samples: int | None = None) -> Iterable[DatasetSample]:
        """Yield normalized samples in deterministic order."""
