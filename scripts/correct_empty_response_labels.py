#!/usr/bin/env python3
"""Apply an auditable correction for empty fault responses."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


RULE_ID = "nonempty_clean_to_empty_fault_v1"
REASON = (
    "A non-empty fault-free answer became empty. Under the project rubric "
    "this loses the core answer and is a major semantic deviation."
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("labels", nargs="+", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "analysis/manual_label_corrections_20260831.jsonl",
    )
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def needs_empty_response_correction(record: Mapping[str, Any]) -> bool:
    return (
        bool(record.get("injected"))
        and bool(str(record.get("clean_answer", "")).strip())
        and not bool(str(record.get("pred_answer", "")).strip())
        and record.get("significance") != 2
    )


def correct_record(record: Mapping[str, Any]) -> dict[str, Any]:
    corrected = dict(record)
    corrected["manual_label_correction"] = {
        "rule_id": RULE_ID,
        "reason": REASON,
        "original_quality_score": record.get("quality_score"),
        "original_significance": record.get("significance"),
        "original_label_status": record.get("label_status"),
    }
    corrected["quality_score"] = 0
    corrected["significance"] = 2
    corrected["label_status"] = "valid"
    return corrected


def correct_file(
    path: Path,
    *,
    apply: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_hash = _sha256(path)
    temporary = path.with_name(f"{path.name}.correction.tmp")
    corrections = []
    split_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    output_stream = (
        temporary.open("w", encoding="utf-8") if apply else None
    )
    try:
        with path.open(encoding="utf-8") as input_stream:
            for line_number, line in enumerate(input_stream, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if needs_empty_response_correction(record):
                    original = {
                        "quality_score": record.get("quality_score"),
                        "significance": record.get("significance"),
                        "label_status": record.get("label_status"),
                    }
                    record = correct_record(record)
                    corrections.append(
                        {
                            "labels_path": str(path),
                            "line_number": line_number,
                            "sample_uid": record.get("sample_uid"),
                            "split": record.get("split"),
                            "question": record.get("question"),
                            "clean_answer": record.get("clean_answer"),
                            "pred_answer": record.get("pred_answer"),
                            "original": original,
                            "corrected": {
                                "quality_score": 0,
                                "significance": 2,
                                "label_status": "valid",
                            },
                            "rule_id": RULE_ID,
                            "reason": REASON,
                        }
                    )
                    split_counts[str(record.get("split"))] += 1
                    status_counts[str(original["label_status"])] += 1
                if output_stream is not None:
                    output_stream.write(
                        json.dumps(record, ensure_ascii=False) + "\n"
                    )
        if output_stream is not None:
            output_stream.close()
            temporary.replace(path)
    except BaseException:
        if output_stream is not None and not output_stream.closed:
            output_stream.close()
        temporary.unlink(missing_ok=True)
        raise
    return corrections, {
        "labels_path": str(path),
        "source_sha256": source_hash,
        "result_sha256": _sha256(path) if apply else None,
        "applied": apply,
        "corrections": len(corrections),
        "split_counts": dict(sorted(split_counts.items())),
        "original_status_counts": dict(sorted(status_counts.items())),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _append_manifest(
    path: Path,
    corrections: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for correction in corrections:
            stream.write(json.dumps(correction, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    paths = [Path(value).resolve() for value in args.labels]
    all_corrections = []
    summaries = []
    for path in paths:
        corrections, summary = correct_file(path, apply=args.apply)
        all_corrections.extend(corrections)
        summaries.append(summary)
    if args.apply:
        manifest = args.manifest.resolve()
        _append_manifest(manifest, all_corrections)
    else:
        manifest = None
    result = {
        "rule_id": RULE_ID,
        "files": summaries,
        "total_corrections": len(all_corrections),
        "manifest": str(manifest) if manifest is not None else None,
    }
    if manifest is not None:
        summary_path = manifest.with_suffix(".summary.json")
        summary_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        result["summary"] = str(summary_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
