"""Leakage-safe sample cohorts for detector comparisons."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ComparisonCohorts:
    fit_orig_ids: frozenset[str]
    calibration_orig_ids: frozenset[str]
    test_orig_ids: frozenset[str]

    def __post_init__(self) -> None:
        if not self.fit_orig_ids:
            raise ValueError("Fit cohort must not be empty")
        if not self.calibration_orig_ids:
            raise ValueError("Calibration cohort must not be empty")
        if not self.test_orig_ids:
            raise ValueError("Test cohort must not be empty")
        if self.fit_orig_ids & self.calibration_orig_ids:
            raise ValueError("Fit and calibration cohorts overlap")
        if self.fit_orig_ids & self.test_orig_ids:
            raise ValueError("Fit and test cohorts overlap")
        if self.calibration_orig_ids & self.test_orig_ids:
            raise ValueError("Calibration and test cohorts overlap")

    def split_for(self, orig_id: str) -> str | None:
        value = str(orig_id)
        if value in self.fit_orig_ids:
            return "fit"
        if value in self.calibration_orig_ids:
            return "calibration"
        if value in self.test_orig_ids:
            return "test"
        return None

    def to_dict(self) -> dict[str, object]:
        return {
            "fit_orig_ids": sorted(self.fit_orig_ids),
            "calibration_orig_ids": sorted(self.calibration_orig_ids),
            "test_orig_ids": sorted(self.test_orig_ids),
            "counts": {
                "fit": len(self.fit_orig_ids),
                "calibration": len(self.calibration_orig_ids),
                "test": len(self.test_orig_ids),
            },
        }


def deterministic_subset(
    orig_ids: Iterable[str],
    *,
    limit: int | None,
    random_seed: int,
) -> tuple[str, ...]:
    unique = {str(value) for value in orig_ids}
    ordered = sorted(unique, key=lambda value: _stable_rank(value, random_seed))
    if limit is None:
        return tuple(ordered)
    if limit <= 0:
        raise ValueError("limit must be positive or None")
    return tuple(ordered[:limit])


def _stable_rank(value: str, seed: int) -> str:
    payload = f"{seed}:{value}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
