"""Deterministic dataset-level split manifests for all experiment stages."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping


SPLIT_NAMES = ("fit", "calibration", "test")


@dataclass(frozen=True)
class SplitAssignment:
    sequence_id: int
    orig_id: str
    semantic_group_id: str
    split: str

    def __post_init__(self) -> None:
        if self.sequence_id < 0:
            raise ValueError("sequence_id must be non-negative")
        if not self.orig_id.strip():
            raise ValueError("orig_id must not be empty")
        if not self.semantic_group_id.strip():
            raise ValueError("semantic_group_id must not be empty")
        if self.split not in SPLIT_NAMES:
            raise ValueError(f"Unsupported split: {self.split}")


@dataclass(frozen=True)
class DatasetSplitManifest:
    dataset: str
    seed: int
    fit_ratio: float
    calibration_ratio: float
    test_ratio: float
    assignments: tuple[SplitAssignment, ...]
    assignment_sha256: str
    _by_orig_id: Mapping[str, SplitAssignment] = field(
        init=False, repr=False, compare=False
    )
    _by_sequence_id: Mapping[int, SplitAssignment] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        _validate_ratios(
            self.fit_ratio,
            self.calibration_ratio,
            self.test_ratio,
        )
        if not self.dataset.strip():
            raise ValueError("dataset must not be empty")
        if not self.assignments:
            raise ValueError("split manifest must contain assignments")

        orig_ids = [assignment.orig_id for assignment in self.assignments]
        sequence_ids = [
            assignment.sequence_id for assignment in self.assignments
        ]
        if len(orig_ids) != len(set(orig_ids)):
            raise ValueError("split manifest contains duplicate orig_id values")
        if len(sequence_ids) != len(set(sequence_ids)):
            raise ValueError(
                "split manifest contains duplicate sequence_id values"
            )
        if sorted(sequence_ids) != list(range(len(sequence_ids))):
            raise ValueError(
                "split manifest sequence_id values must be contiguous from zero"
            )

        group_splits: dict[str, str] = {}
        for assignment in self.assignments:
            previous = group_splits.setdefault(
                assignment.semantic_group_id,
                assignment.split,
            )
            if previous != assignment.split:
                raise ValueError(
                    "semantic group appears in multiple splits: "
                    f"{assignment.semantic_group_id}"
                )
        if set(group_splits.values()) != set(SPLIT_NAMES):
            raise ValueError("fit, calibration, and test must all be non-empty")

        object.__setattr__(
            self,
            "_by_orig_id",
            {assignment.orig_id: assignment for assignment in self.assignments},
        )
        object.__setattr__(
            self,
            "_by_sequence_id",
            {
                assignment.sequence_id: assignment
                for assignment in self.assignments
            },
        )

        expected = _assignment_digest(self.assignments)
        if self.assignment_sha256 != expected:
            raise ValueError(
                "split manifest assignment_sha256 does not match assignments"
            )

    @property
    def by_orig_id(self) -> Mapping[str, SplitAssignment]:
        return self._by_orig_id

    @property
    def by_sequence_id(self) -> Mapping[int, SplitAssignment]:
        return self._by_sequence_id

    def assignment_for_orig_id(self, orig_id: str) -> SplitAssignment:
        try:
            return self._by_orig_id[str(orig_id)]
        except KeyError as error:
            raise KeyError(
                f"orig_id is missing from {self.dataset} split manifest: "
                f"{orig_id}"
            ) from error

    def assignment_for_sequence_id(
        self,
        sequence_id: int,
    ) -> SplitAssignment:
        try:
            return self._by_sequence_id[int(sequence_id)]
        except KeyError as error:
            raise KeyError(
                f"sequence_id is missing from {self.dataset} split manifest: "
                f"{sequence_id}"
            ) from error

    def to_dict(self) -> dict[str, Any]:
        sample_counts = {
            split: sum(
                assignment.split == split
                for assignment in self.assignments
            )
            for split in SPLIT_NAMES
        }
        group_counts = {
            split: len(
                {
                    assignment.semantic_group_id
                    for assignment in self.assignments
                    if assignment.split == split
                }
            )
            for split in SPLIT_NAMES
        }
        return {
            "schema_version": 1,
            "dataset": self.dataset,
            "seed": self.seed,
            "ratios": {
                "fit": self.fit_ratio,
                "calibration": self.calibration_ratio,
                "test": self.test_ratio,
            },
            "sample_count": len(self.assignments),
            "group_count": len(
                {
                    assignment.semantic_group_id
                    for assignment in self.assignments
                }
            ),
            "sample_counts": sample_counts,
            "group_counts": group_counts,
            "assignment_sha256": self.assignment_sha256,
            "assignments": [
                asdict(assignment) for assignment in self.assignments
            ],
        }

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "DatasetSplitManifest":
        if int(value.get("schema_version", -1)) != 1:
            raise ValueError("Unsupported dataset split manifest schema")
        ratios = _required_mapping(value.get("ratios"), "ratios")
        raw_assignments = value.get("assignments")
        if not isinstance(raw_assignments, list) or not raw_assignments:
            raise ValueError("split manifest assignments must be non-empty")
        assignments = tuple(
            SplitAssignment(
                sequence_id=int(item["sequence_id"]),
                orig_id=str(item["orig_id"]),
                semantic_group_id=str(item["semantic_group_id"]),
                split=str(item["split"]),
            )
            for item in raw_assignments
        )
        return cls(
            dataset=str(value["dataset"]),
            seed=int(value["seed"]),
            fit_ratio=float(ratios["fit"]),
            calibration_ratio=float(ratios["calibration"]),
            test_ratio=float(ratios["test"]),
            assignments=assignments,
            assignment_sha256=str(value["assignment_sha256"]),
        )


def create_split_manifest(
    dataset: str,
    samples: Iterable[Any],
    *,
    seed: int,
    fit_ratio: float,
    calibration_ratio: float,
    test_ratio: float,
) -> DatasetSplitManifest:
    """Assign semantic groups to three deterministic, disjoint splits."""

    _validate_ratios(fit_ratio, calibration_ratio, test_ratio)
    identities: list[tuple[int, str, str]] = []
    groups: dict[str, list[int]] = {}
    seen_orig_ids: set[str] = set()
    for sequence_id, sample in enumerate(samples):
        orig_id = str(sample.orig_id)
        semantic_group_id = str(sample.semantic_group_id)
        if orig_id in seen_orig_ids:
            raise ValueError(f"Duplicate dataset orig_id: {orig_id}")
        seen_orig_ids.add(orig_id)
        identities.append((sequence_id, orig_id, semantic_group_id))
        groups.setdefault(semantic_group_id, []).append(sequence_id)

    if len(groups) < len(SPLIT_NAMES):
        raise ValueError(
            "At least three semantic groups are required for splitting"
        )

    ordered_groups = sorted(
        groups,
        key=lambda group_id: _stable_group_rank(dataset, group_id, seed),
    )
    group_sizes = [len(groups[group_id]) for group_id in ordered_groups]
    total_samples = len(identities)
    fit_target = round(total_samples * fit_ratio)
    calibration_target = round(total_samples * calibration_ratio)

    fit_end = _closest_boundary(
        group_sizes,
        start=0,
        target=fit_target,
        minimum_end=1,
        maximum_end=len(ordered_groups) - 2,
    )
    calibration_end = _closest_boundary(
        group_sizes,
        start=fit_end,
        target=calibration_target,
        minimum_end=fit_end + 1,
        maximum_end=len(ordered_groups) - 1,
    )
    split_by_group = {
        group_id: (
            "fit"
            if index < fit_end
            else "calibration"
            if index < calibration_end
            else "test"
        )
        for index, group_id in enumerate(ordered_groups)
    }
    assignments = tuple(
        SplitAssignment(
            sequence_id=sequence_id,
            orig_id=orig_id,
            semantic_group_id=semantic_group_id,
            split=split_by_group[semantic_group_id],
        )
        for sequence_id, orig_id, semantic_group_id in identities
    )
    return DatasetSplitManifest(
        dataset=dataset,
        seed=seed,
        fit_ratio=fit_ratio,
        calibration_ratio=calibration_ratio,
        test_ratio=test_ratio,
        assignments=assignments,
        assignment_sha256=_assignment_digest(assignments),
    )


def load_split_manifest(path: str | Path) -> DatasetSplitManifest:
    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, Mapping):
        raise ValueError(f"Split manifest must be a JSON object: {source}")
    return DatasetSplitManifest.from_mapping(value)


def write_split_manifest(
    manifest: DatasetSplitManifest,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Split manifest exists; pass overwrite=True: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(
                manifest.to_dict(),
                stream,
                ensure_ascii=False,
                indent=2,
            )
            stream.write("\n")
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _closest_boundary(
    group_sizes: list[int],
    *,
    start: int,
    target: int,
    minimum_end: int,
    maximum_end: int,
) -> int:
    candidates = []
    running = 0
    for index in range(start, maximum_end):
        running += group_sizes[index]
        end = index + 1
        if end >= minimum_end:
            candidates.append((abs(running - target), end))
    if not candidates:
        raise ValueError("Cannot create non-empty dataset splits")
    return min(candidates)[1]


def _stable_group_rank(dataset: str, group_id: str, seed: int) -> str:
    payload = f"{seed}:{dataset}:{group_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _assignment_digest(
    assignments: Iterable[SplitAssignment],
) -> str:
    payload = "\n".join(
        (
            f"{assignment.sequence_id}\t{assignment.orig_id}\t"
            f"{assignment.semantic_group_id}\t{assignment.split}"
        )
        for assignment in assignments
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_ratios(
    fit_ratio: float,
    calibration_ratio: float,
    test_ratio: float,
) -> None:
    ratios = (fit_ratio, calibration_ratio, test_ratio)
    if any(ratio <= 0.0 or ratio >= 1.0 for ratio in ratios):
        raise ValueError("split ratios must be between zero and one")
    if abs(sum(ratios) - 1.0) > 1e-12:
        raise ValueError("fit, calibration, and test ratios must sum to one")


def _required_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value
