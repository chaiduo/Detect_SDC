"""Pipeline orchestration primitives."""

from enum import Enum


class PipelineStage(str, Enum):
    PROFILE = "profile"
    COLLECT_MAPPING = "collect_mapping"
    TRAIN_MAPPING = "train_mapping"
    INJECT = "inject"
    LABEL = "label"
    FEATURIZE = "featurize"
    TRAIN_DETECTOR = "train_detector"
    REPORT = "report"


DEFAULT_STAGE_ORDER = tuple(PipelineStage)

__all__ = ["DEFAULT_STAGE_ORDER", "PipelineStage"]
