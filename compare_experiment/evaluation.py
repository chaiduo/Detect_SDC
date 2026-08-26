"""Threshold calibration and detection metrics for comparison methods."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class ThresholdCalibration:
    threshold: float
    target_fpr: float
    achieved_fpr: float
    budget_feasible: bool
    negative_samples: int
    forced_non_finite_positives: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DetectionMetrics:
    samples: int
    detected: int
    sdc_samples: int
    significant_sdc_samples: int
    non_sdc_samples: int
    detected_sdc: int
    detected_significant_sdc: int
    false_positive_non_sdc: int
    sdc_recall: float
    significant_sdc_recall: float
    non_sdc_fpr: float
    significant_sdc_precision: float
    significant_sdc_f1: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def apply_threshold(
    scores: Sequence[float],
    threshold: float,
) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if np.isnan(values).any():
        raise ValueError("Detection scores contain NaN")
    return np.isposinf(values) | (values > float(threshold))


def threshold_at_fpr(
    non_sdc_scores: Sequence[float],
    *,
    target_fpr: float,
) -> ThresholdCalibration:
    """Select the lowest threshold whose calibration FPR is within budget."""
    if not 0.0 <= target_fpr < 1.0:
        raise ValueError("target_fpr must be in [0, 1)")
    scores = np.asarray(non_sdc_scores, dtype=np.float64)
    if scores.ndim != 1 or not scores.size:
        raise ValueError("non_sdc_scores must be a non-empty vector")
    if np.isnan(scores).any():
        raise ValueError("non_sdc_scores contain NaN")

    allowed = int(np.floor(target_fpr * len(scores)))
    forced = int(np.isposinf(scores).sum())
    if forced > allowed:
        threshold = float("inf")
        predictions = apply_threshold(scores, threshold)
        return ThresholdCalibration(
            threshold=threshold,
            target_fpr=float(target_fpr),
            achieved_fpr=float(predictions.mean()),
            budget_feasible=False,
            negative_samples=len(scores),
            forced_non_finite_positives=forced,
        )

    finite = scores[np.isfinite(scores)]
    remaining = allowed - forced
    if not finite.size:
        threshold = float("inf")
    elif remaining <= 0:
        threshold = float(finite.max())
    elif remaining >= len(finite):
        threshold = float("-inf")
    else:
        descending = np.sort(finite)[::-1]
        threshold = float(descending[remaining])

    predictions = apply_threshold(scores, threshold)
    return ThresholdCalibration(
        threshold=threshold,
        target_fpr=float(target_fpr),
        achieved_fpr=float(predictions.mean()),
        budget_feasible=True,
        negative_samples=len(scores),
        forced_non_finite_positives=forced,
    )


def evaluate_detection(
    *,
    is_sdc: Sequence[bool | int],
    is_significant_sdc: Sequence[bool | int],
    detected: Sequence[bool | int],
) -> DetectionMetrics:
    sdc = np.asarray(is_sdc, dtype=bool)
    significant = np.asarray(is_significant_sdc, dtype=bool)
    predictions = np.asarray(detected, dtype=bool)
    if not (sdc.ndim == significant.ndim == predictions.ndim == 1):
        raise ValueError("Detection inputs must be one-dimensional")
    if not (len(sdc) == len(significant) == len(predictions)):
        raise ValueError("Detection inputs must have identical lengths")
    if np.any(significant & ~sdc):
        raise ValueError("Significant-SDC samples must also be SDC samples")

    non_sdc = ~sdc
    detected_sdc = int(np.sum(predictions & sdc))
    detected_significant = int(np.sum(predictions & significant))
    false_positive_non_sdc = int(np.sum(predictions & non_sdc))
    sdc_count = int(sdc.sum())
    significant_count = int(significant.sum())
    non_sdc_count = int(non_sdc.sum())
    detected_count = int(predictions.sum())

    sdc_recall = _safe_ratio(detected_sdc, sdc_count)
    significant_recall = _safe_ratio(
        detected_significant,
        significant_count,
    )
    significant_precision = _safe_ratio(
        detected_significant,
        detected_count,
    )
    significant_f1 = (
        0.0
        if significant_precision + significant_recall == 0.0
        else 2.0
        * significant_precision
        * significant_recall
        / (significant_precision + significant_recall)
    )
    return DetectionMetrics(
        samples=len(sdc),
        detected=detected_count,
        sdc_samples=sdc_count,
        significant_sdc_samples=significant_count,
        non_sdc_samples=non_sdc_count,
        detected_sdc=detected_sdc,
        detected_significant_sdc=detected_significant,
        false_positive_non_sdc=false_positive_non_sdc,
        sdc_recall=sdc_recall,
        significant_sdc_recall=significant_recall,
        non_sdc_fpr=_safe_ratio(false_positive_non_sdc, non_sdc_count),
        significant_sdc_precision=significant_precision,
        significant_sdc_f1=significant_f1,
    )


def _safe_ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else float(numerator / denominator)
