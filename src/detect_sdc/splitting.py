"""Leakage-safe dataframe splitting by stable sample groups."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class GroupSplitSummary:
    group_column: str
    holdout_ratio: float
    random_state: int
    train_rows: int
    holdout_rows: int
    train_groups: int
    holdout_groups: int
    group_overlap: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GroupSplit:
    train: Any
    holdout: Any
    summary: GroupSplitSummary


def validate_identity_columns(
    frame: Any,
    *,
    group_column: str = "orig_id",
    sample_uid_column: str = "sample_uid",
) -> None:
    _validate_group_column(frame, group_column)
    if sample_uid_column not in frame.columns:
        raise ValueError(f"Dataframe missing sample UID column: {sample_uid_column}")

    sample_uids = frame[sample_uid_column]
    missing = sample_uids.isna() | sample_uids.astype(str).str.strip().eq("")
    if missing.any():
        raise ValueError(
            f"{sample_uid_column} contains {int(missing.sum())} missing or empty values"
        )

    duplicated = sample_uids.duplicated(keep=False)
    if duplicated.any():
        examples = sample_uids.loc[duplicated].astype(str).drop_duplicates().head(5).tolist()
        raise ValueError(
            f"{sample_uid_column} contains {int(duplicated.sum())} duplicate rows; "
            f"examples: {examples}"
        )


def split_by_group(
    frame: Any,
    *,
    group_column: str = "orig_id",
    holdout_ratio: float = 0.15,
    random_state: int = 42,
) -> GroupSplit:
    """Split a dataframe while keeping every group entirely on one side."""
    from sklearn.model_selection import GroupShuffleSplit

    _validate_group_column(frame, group_column)
    if not 0.0 < holdout_ratio < 1.0:
        raise ValueError("holdout_ratio must be between 0 and 1")

    group_count = int(frame[group_column].nunique(dropna=False))
    if group_count < 2:
        raise ValueError("Grouped splitting requires at least two distinct groups")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=holdout_ratio,
        random_state=random_state,
    )
    train_idx, holdout_idx = next(splitter.split(frame, groups=frame[group_column]))
    train = frame.iloc[train_idx].copy()
    holdout = frame.iloc[holdout_idx].copy()

    train_groups = set(train[group_column].unique())
    holdout_groups = set(holdout[group_column].unique())
    overlap = train_groups.intersection(holdout_groups)
    if overlap:
        examples = sorted(map(str, overlap))[:5]
        raise AssertionError(
            f"{group_column} overlap between train and holdout: "
            f"{len(overlap)}; examples: {examples}"
        )

    if len(train) + len(holdout) != len(frame):
        raise AssertionError("Grouped split lost dataframe rows")

    return GroupSplit(
        train=train,
        holdout=holdout,
        summary=GroupSplitSummary(
            group_column=group_column,
            holdout_ratio=holdout_ratio,
            random_state=random_state,
            train_rows=len(train),
            holdout_rows=len(holdout),
            train_groups=len(train_groups),
            holdout_groups=len(holdout_groups),
            group_overlap=0,
        ),
    )


def _validate_group_column(frame: Any, group_column: str) -> None:
    if group_column not in frame.columns:
        raise ValueError(f"Dataframe missing group column: {group_column}")
    if frame.empty:
        raise ValueError("Cannot split an empty dataframe")

    groups = frame[group_column]
    missing = groups.isna() | groups.astype(str).str.strip().eq("")
    if missing.any():
        raise ValueError(
            f"{group_column} contains {int(missing.sum())} missing or empty values"
        )
