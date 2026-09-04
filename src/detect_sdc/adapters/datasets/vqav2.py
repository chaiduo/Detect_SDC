"""VQAv2 parquet dataset adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from .base import DatasetSample


@dataclass(frozen=True)
class VQAv2Adapter:
    parquet: Path

    @property
    def name(self) -> str:
        return "vqav2"

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "VQAv2Adapter":
        paths = _mapping(config.get("paths"), "paths")
        return cls(parquet=Path(str(paths["parquet"])).expanduser())

    def iter_samples(
        self,
        max_samples: int | None = None,
    ) -> Iterator[DatasetSample]:
        import pandas as pd

        yielded = 0
        for parquet_file in self._parquet_files():
            frame = pd.read_parquet(parquet_file)
            required = {
                "question_id",
                "question",
                "multiple_choice_answer",
                "image",
                "image_id",
            }
            missing = required - set(frame.columns)
            if missing:
                raise ValueError(
                    f"VQAv2 shard {parquet_file} is missing columns: {sorted(missing)}"
                )

            for row_index, row in frame.iterrows():
                if max_samples is not None and yielded >= max_samples:
                    return
                question_id = str(row["question_id"])
                image_id = str(row["image_id"])
                yield DatasetSample(
                    orig_id=question_id,
                    semantic_group_id=image_id,
                    question=str(row["question"]).strip(),
                    ground_truth=str(row["multiple_choice_answer"]),
                    image=row["image"],
                    metadata={
                        "question_id": question_id,
                        "image_id": image_id,
                        "question_type": _optional_string(
                            row.get("question_type")
                        ),
                        "source_file": str(parquet_file),
                        "row_idx": int(row_index),
                    },
                )
                yielded += 1

    def _parquet_files(self) -> list[Path]:
        if self.parquet.is_file():
            return [self.parquet]
        if self.parquet.is_dir():
            files = sorted(self.parquet.glob("*.parquet"))
            if files:
                return files
            raise FileNotFoundError(
                f"No parquet files found in directory: {self.parquet}"
            )
        raise FileNotFoundError(f"VQAv2 parquet path does not exist: {self.parquet}")


def _optional_string(value: Any) -> str | None:
    return None if value is None else str(value)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value
