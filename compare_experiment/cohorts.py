"""Leakage-safe sample cohorts for detector comparisons."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


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


def load_comparison_cohorts(
    train_csv: str | Path,
    valid_csv: str | Path,
    *,
    calibration_ratio: float = 0.2,
    random_seed: int = 42,
) -> ComparisonCohorts:
    if not 0.0 < calibration_ratio < 1.0:
        raise ValueError("calibration_ratio must be between zero and one")
    train_ids = _read_orig_ids(train_csv)
    test_ids = _read_orig_ids(valid_csv)
    overlap = train_ids & test_ids
    if overlap:
        raise ValueError(
            f"Main train/test orig_id overlap: {len(overlap)}"
        )

    ranked = sorted(
        train_ids,
        key=lambda value: _stable_rank(value, random_seed),
    )
    calibration_count = max(1, round(len(ranked) * calibration_ratio))
    if calibration_count >= len(ranked):
        raise ValueError("Calibration split leaves no fit samples")
    calibration = frozenset(ranked[:calibration_count])
    fit = frozenset(ranked[calibration_count:])
    return ComparisonCohorts(
        fit_orig_ids=fit,
        calibration_orig_ids=calibration,
        test_orig_ids=frozenset(test_ids),
    )


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


def _read_orig_ids(path: str | Path) -> set[str]:
    frame = pd.read_csv(path, usecols=["orig_id"])
    values = {
        str(value)
        for value in frame["orig_id"].dropna().astype(str)
        if str(value).strip()
    }
    if not values:
        raise ValueError(f"No orig_id values found in {path}")
    return values


def _stable_rank(value: str, seed: int) -> str:
    payload = f"{seed}:{value}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
