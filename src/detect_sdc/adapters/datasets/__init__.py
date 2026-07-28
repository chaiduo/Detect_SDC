"""Dataset-specific adapters."""

from .base import DatasetAdapter, DatasetSample
from .earthvqa import EarthVQAAdapter
from .lingoqa import LingoQAAdapter
from .vqav2 import VQAv2Adapter

__all__ = [
    "DatasetAdapter",
    "DatasetSample",
    "EarthVQAAdapter",
    "LingoQAAdapter",
    "VQAv2Adapter",
]
