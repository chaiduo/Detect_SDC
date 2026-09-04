"""LingoQA dataset adapter."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from .base import DatasetSample


@dataclass(frozen=True)
class LingoQAAdapter:
    annotations: Path
    images: Path
    images_per_question: int = 5

    @property
    def name(self) -> str:
        return "lingoqa"

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "LingoQAAdapter":
        paths = _mapping(config.get("paths"), "paths")
        selection = config.get("selection", {})
        if not isinstance(selection, Mapping):
            raise ValueError("selection must be a mapping")
        return cls(
            annotations=Path(str(paths["annotations"])).expanduser(),
            images=Path(str(paths["images"])).expanduser(),
            images_per_question=int(selection.get("images_per_question", 5)),
        )

    def iter_samples(
        self,
        max_samples: int | None = None,
    ) -> Iterator[DatasetSample]:
        import pandas as pd

        if self.images_per_question <= 0:
            raise ValueError("images_per_question must be positive")
        frame = pd.read_parquet(self.annotations)
        required = {"question_id", "segment_id", "images", "question", "answer"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"LingoQA is missing columns: {sorted(missing)}")

        yielded = 0
        annotation_occurrences: dict[str, int] = defaultdict(int)
        for row in frame.itertuples(index=False):
            images = list(row.images)
            annotation_hash = _annotation_hash(row)
            occurrence = annotation_occurrences[annotation_hash]
            annotation_occurrences[annotation_hash] += 1
            for image_index, relative_path in enumerate(
                images[: self.images_per_question]
            ):
                orig_id = (
                    f"{row.question_id}:{annotation_hash}:"
                    f"{occurrence}:{image_index}"
                )
                if max_samples is not None and yielded >= max_samples:
                    return
                yield DatasetSample(
                    orig_id=orig_id,
                    semantic_group_id=str(row.question_id),
                    question=str(row.question).strip(),
                    ground_truth=str(row.answer),
                    image=self.images / str(relative_path),
                    metadata={
                        "question_id": str(row.question_id),
                        "segment_id": str(row.segment_id),
                        "annotation_hash": annotation_hash,
                        "annotation_occurrence": occurrence,
                        "image_index": image_index,
                        "image_path": str(relative_path),
                    },
                )
                yielded += 1


def _annotation_hash(row: Any) -> str:
    payload = json.dumps(
        {
            "question_id": str(row.question_id),
            "segment_id": str(row.segment_id),
            "question": str(row.question),
            "answer": str(row.answer),
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value
