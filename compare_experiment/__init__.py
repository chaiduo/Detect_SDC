"""Mechanism-matched SDC detector comparisons for Detect_SDC."""

from .artifacts import load_profiles, save_profiles
from .cohorts import (
    ComparisonCohorts,
    deterministic_subset,
    load_comparison_cohorts,
)
from .evaluation import DetectionMetrics, evaluate_detection, threshold_at_fpr
from .monitor import ActivationTrace, OnlineActivationMonitor
from .profiles import (
    DrDNAConfig,
    DrDNAProfile,
    DrDNAProfiler,
    RangeProfile,
    RangeProfiler,
)

__all__ = [
    "ActivationTrace",
    "ComparisonCohorts",
    "DetectionMetrics",
    "DrDNAConfig",
    "DrDNAProfile",
    "DrDNAProfiler",
    "OnlineActivationMonitor",
    "RangeProfile",
    "RangeProfiler",
    "deterministic_subset",
    "evaluate_detection",
    "load_comparison_cohorts",
    "load_profiles",
    "save_profiles",
    "threshold_at_fpr",
]
