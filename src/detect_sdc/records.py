"""Stable data contracts for records passed between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class LabelStatus(str, Enum):
    VALID = "valid"
    IDENTICAL_ANSWER = "identical_answer"
    PARSE_ERROR = "parse_error"


def significant_sdc_target(
    pred_answer: str,
    clean_answer: str,
    significance: int | None,
) -> bool:
    return pred_answer != clean_answer and significance == 2


@dataclass(frozen=True)
class SampleRef:
    dataset: str
    orig_id: str
    sample_uid: str

    def __post_init__(self) -> None:
        if not self.dataset:
            raise ValueError("dataset must not be empty")
        if not self.orig_id:
            raise ValueError("orig_id must not be empty")
        if not self.sample_uid:
            raise ValueError("sample_uid must not be empty")


@dataclass(frozen=True)
class GoldenRecord:
    ref: SampleRef
    question: str
    clean_answer: str
    ground_truth: str | None = None
    image_ref: str | None = None


@dataclass(frozen=True)
class LabeledRecord:
    ref: SampleRef
    question: str
    clean_answer: str
    pred_answer: str
    quality_score: int | None
    significance: int | None
    status: LabelStatus = LabelStatus.VALID
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in (
            ("quality_score", self.quality_score),
            ("significance", self.significance),
        ):
            if value is not None and value not in (0, 1, 2):
                raise ValueError(f"{name} must be one of 0, 1, 2, or None")

        if self.status == LabelStatus.PARSE_ERROR:
            if self.quality_score is not None or self.significance is not None:
                raise ValueError("parse-error records must not carry numeric labels")
            return

        if self.quality_score is None or self.significance is None:
            raise ValueError("valid records require quality_score and significance")
        if self.quality_score + self.significance != 2:
            raise ValueError("quality_score and significance must be inverse scales")
        if self.pred_answer == self.clean_answer:
            if self.quality_score != 2 or self.significance != 0:
                raise ValueError("identical answers must map to quality=2, significance=0")

    @property
    def target_significant_sdc(self) -> bool:
        return significant_sdc_target(
            pred_answer=self.pred_answer,
            clean_answer=self.clean_answer,
            significance=self.significance,
        )

    @classmethod
    def from_legacy(cls, record: Mapping[str, Any], dataset: str) -> "LabeledRecord":
        quality_score = _optional_score(record.get("quality_score"))
        significance = _optional_score(record.get("significance"))
        if quality_score is None or significance is None:
            status = LabelStatus.PARSE_ERROR
            quality_score = None
            significance = None
        elif str(record.get("pred_answer", "")) == str(record.get("clean_answer", "")):
            status = LabelStatus.IDENTICAL_ANSWER
            quality_score = 2
            significance = 0
        else:
            status = LabelStatus.VALID

        orig_id = str(record.get("orig_id", record.get("id", "")))
        sample_uid = str(record.get("sample_uid", orig_id))
        return cls(
            ref=SampleRef(dataset=dataset, orig_id=orig_id, sample_uid=sample_uid),
            question=str(record.get("question", "")),
            clean_answer=str(record.get("clean_answer", "")),
            pred_answer=str(record.get("pred_answer", "")),
            quality_score=quality_score,
            significance=significance,
            status=status,
            metadata=record,
        )


@dataclass(frozen=True)
class FeatureRow:
    ref: SampleRef
    features: Mapping[str, float | None]
    target_significant_sdc: bool
    significance: int

    def __post_init__(self) -> None:
        if self.significance not in (0, 1, 2):
            raise ValueError("significance must be one of 0, 1, or 2")
        if not self.features:
            raise ValueError("features must not be empty")


def _optional_score(value: Any) -> int | None:
    try:
        score = int(value)
    except (TypeError, ValueError):
        return None
    return score if score in (0, 1, 2) else None

