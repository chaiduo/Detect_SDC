"""EarthVQA dataset adapter."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from .base import DatasetSample


@dataclass(frozen=True)
class EarthVQAAdapter:
    annotations: Path
    images: Path
    question_type: str = "Comprehensive Analysis"
    max_questions_per_image: int = 2

    @property
    def name(self) -> str:
        return "earthvqa"

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "EarthVQAAdapter":
        paths = _mapping(config.get("paths"), "paths")
        selection = config.get("selection", {})
        if not isinstance(selection, Mapping):
            raise ValueError("selection must be a mapping")
        return cls(
            annotations=Path(str(paths["annotations"])).expanduser(),
            images=Path(str(paths["images"])).expanduser(),
            question_type=str(
                selection.get("question_type", "Comprehensive Analysis")
            ),
            max_questions_per_image=int(
                selection.get("max_questions_per_image", 2)
            ),
        )

    def iter_samples(
        self,
        max_samples: int | None = None,
    ) -> Iterator[DatasetSample]:
        if self.max_questions_per_image <= 0:
            raise ValueError("max_questions_per_image must be positive")
        with self.annotations.open("r", encoding="utf-8") as stream:
            raw = json.load(stream)
        if not isinstance(raw, dict):
            raise ValueError("EarthVQA annotations must be an image-to-QA mapping")

        yielded = 0
        for image_filename, qa_list in raw.items():
            if not isinstance(qa_list, list):
                raise ValueError(f"EarthVQA QA list is invalid: {image_filename}")
            image_path = self.images / str(image_filename)
            selected = 0
            for qa_index, qa in enumerate(qa_list):
                if not isinstance(qa, Mapping):
                    continue
                if str(qa.get("Type", "")) != self.question_type:
                    continue
                if selected >= self.max_questions_per_image:
                    break
                if max_samples is not None and yielded >= max_samples:
                    return

                question = str(qa.get("Question", "")).strip()
                if not question:
                    continue
                yield DatasetSample(
                    orig_id=f"{image_filename}:{qa_index}",
                    question=question,
                    ground_truth=str(qa.get("Answer", "")),
                    image=image_path,
                    metadata={
                        "image_filename": str(image_filename),
                        "qa_index": qa_index,
                        "question_type": self.question_type,
                    },
                )
                selected += 1
                yielded += 1


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value
