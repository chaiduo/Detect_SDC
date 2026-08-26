"""Structured comparison artifact I/O."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .profiles import DrDNAProfile, RangeProfile


def save_profiles(
    path: str | Path,
    *,
    range_profile: RangeProfile,
    drdna_profile: DrDNAProfile,
    metadata: Mapping[str, Any],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp")
    value = {
        "schema_version": 1,
        "metadata": dict(metadata),
        "range_profile": range_profile.to_dict(),
        "drdna_profile": drdna_profile.to_dict(),
    }
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def load_profiles(
    path: str | Path,
) -> tuple[RangeProfile, DrDNAProfile, dict[str, Any]]:
    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    if int(value.get("schema_version", -1)) != 1:
        raise ValueError(f"Unsupported profile schema: {source}")
    return (
        RangeProfile.from_dict(value["range_profile"]),
        DrDNAProfile.from_dict(value["drdna_profile"]),
        dict(value.get("metadata", {})),
    )
