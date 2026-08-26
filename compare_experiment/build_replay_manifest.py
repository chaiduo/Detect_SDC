#!/usr/bin/env python3
"""Select a deterministic fault replay cohort from labeled JSONL records."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from detect_sdc.features.jobs import load_feature_job
from detect_sdc.pipeline.jobs import load_pipeline_job

from .cohorts import load_comparison_cohorts
from .config import load_comparison_config


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--repository-root", type=Path, default=root)
    parser.add_argument(
        "--comparison-config",
        type=Path,
        default=root
        / "compare_experiment/configs/detection_comparison.yaml",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.repository_root.resolve()
    config = load_comparison_config(
        args.comparison_config, repository_root=root
    )
    pipeline_job = load_pipeline_job(
        config.source_config, args.job, repository_root=root
    )
    feature_job = load_feature_job(
        config.source_config, args.job, repository_root=root
    )
    cohorts = load_comparison_cohorts(
        feature_job.train_output,
        feature_job.valid_output,
        calibration_ratio=config.calibration_ratio,
        random_seed=config.split_seed,
    )
    output = args.output or (
        root
        / "compare_experiment/results"
        / args.job
        / "replay_manifest.jsonl"
    )
    output = output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Manifest exists; pass --overwrite: {output}"
        )

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    significant: dict[str, dict[str, Any]] = {}
    scanned = injected = 0
    with pipeline_job.paths.labeled_output.open(encoding="utf-8") as stream:
        for line in stream:
            scanned += 1
            sample = json.loads(line)
            if not isinstance(sample.get("fault"), dict):
                continue
            injected += 1
            split = cohorts.split_for(str(sample.get("orig_id", "")))
            if split not in {"calibration", "test"}:
                continue
            record = compact_record(sample, split)
            if record is None:
                buckets["invalid_significance"].append({})
                continue
            run_index = int(record["run_index"])
            is_sdc = bool(record["is_sdc"])
            if split == "calibration" and run_index == 0 and not is_sdc:
                buckets["calibration_non_sdc"].append(record)
            elif split == "test" and run_index == 0 and not is_sdc:
                buckets["test_non_sdc"].append(record)
            elif split == "test" and is_sdc:
                buckets[f"test_sdc_run_{run_index}"].append(record)
                if record["is_significant_sdc"]:
                    significant[str(record["sample_uid"])] = record

    selected: dict[str, dict[str, Any]] = {}
    for name, offset in (
        ("calibration_non_sdc", 0),
        ("test_non_sdc", 1),
    ):
        for record in take_ranked(
            buckets[name],
            config.maximum_non_sdc,
            config.bootstrap_seed + offset,
        ):
            selected[str(record["sample_uid"])] = record
    per_run_limit = (
        None
        if config.maximum_sdc is None
        else max(1, (config.maximum_sdc + 9) // 10)
    )
    for run_index in range(10):
        for record in take_ranked(
            buckets[f"test_sdc_run_{run_index}"],
            per_run_limit,
            config.bootstrap_seed + 100 + run_index,
        ):
            selected[str(record["sample_uid"])] = record
    selected.update(significant)

    ordered = sorted(
        selected.values(),
        key=lambda row: (
            row["split"],
            row["run_index"],
            row["orig_id"],
            row["sample_uid"],
        ),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in ordered:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(output)

    summary = summarize_manifest(
        args.job,
        scanned,
        injected,
        len(buckets["invalid_significance"]),
        ordered,
    )
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def compact_record(
    sample: dict[str, Any],
    split: str,
) -> dict[str, Any] | None:
    fault = sample["fault"]
    sample_uid = str(sample["sample_uid"])
    is_sdc = int(sample.get("is_sdc", 0)) == 1
    raw_significance = sample.get("significance")
    if raw_significance is None:
        if is_sdc:
            return None
        significance = 0
    else:
        significance = int(raw_significance)
    if significance not in {0, 1, 2}:
        return None
    return {
        "orig_id": str(sample["orig_id"]),
        "sample_uid": sample_uid,
        "split": split,
        "run_index": run_index_from_uid(sample_uid),
        "is_sdc": int(is_sdc),
        "significance": significance,
        "is_significant_sdc": int(is_sdc and significance == 2),
        "fault": {
            "mode": str(fault["mode"]),
            "module": str(fault["module"]),
            "forward": int(fault["forward"]),
            "dtype": str(fault["dtype"]),
            "idx": int(fault["idx"]),
            "bit_positions": [int(bit) for bit in fault["bit_positions"]],
            "before": fault.get("before"),
            "after": fault.get("after"),
        },
        "clean_answer": str(sample.get("clean_answer", "")),
        "recorded_fault_answer": str(sample.get("pred_answer", "")),
    }


def run_index_from_uid(sample_uid: str) -> int:
    marker = ":fault:"
    if marker not in sample_uid:
        raise ValueError(f"Cannot parse fault run from {sample_uid}")
    return int(sample_uid.rsplit(marker, maxsplit=1)[1])


def take_ranked(
    records: Iterable[dict[str, Any]],
    limit: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    ranked = sorted(
        records,
        key=lambda row: hashlib.sha256(
            f"{seed}:{row['sample_uid']}".encode()
        ).hexdigest(),
    )
    return ranked if limit is None else ranked[:limit]


def summarize_manifest(
    job: str,
    scanned: int,
    injected: int,
    invalid_significance: int,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "job": job,
        "scanned_rows": scanned,
        "injected_rows": injected,
        "invalid_significance_rows": invalid_significance,
        "selected_rows": len(rows),
        "calibration_non_sdc": sum(
            row["split"] == "calibration" for row in rows
        ),
        "test_non_sdc": sum(
            row["split"] == "test" and not row["is_sdc"]
            for row in rows
        ),
        "test_sdc": sum(
            row["split"] == "test" and row["is_sdc"] for row in rows
        ),
        "test_significant_sdc": sum(
            row["split"] == "test" and row["is_significant_sdc"]
            for row in rows
        ),
    }


if __name__ == "__main__":
    raise SystemExit(main())
