#!/usr/bin/env python3
"""Audit deterministic replay on clean records whose answers changed."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from detect_sdc.adapters import load_dataset_adapter, load_model_adapter
from detect_sdc.adapters.models.determinism import (
    configure_deterministic_execution,
    prepare_deterministic_environment,
    seed_torch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--injection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=2)
    return parser.parse_args()


def seed_everything(torch: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    seed_torch(torch, seed)


def load_candidates(path: Path) -> dict[str, dict[str, Any]]:
    candidates = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("injected") is True:
                break
            if int(record.get("is_sdc", 0)) != 1:
                continue
            candidates[str(record["orig_id"])] = {
                "canonical_answer": str(record.get("clean_answer", "")),
                "clean_replay_answer": str(record.get("pred_answer", "")),
            }
    return candidates


def main() -> int:
    args = parse_args()
    if args.repeats < 2:
        raise ValueError("--repeats must be at least 2")
    prepare_deterministic_environment(True)

    import torch

    seed_everything(torch, args.seed)
    configure_deterministic_execution(
        torch,
        enabled=True,
        seed=args.seed,
    )

    candidates = load_candidates(args.injection.resolve())
    dataset = load_dataset_adapter(args.dataset_config.resolve())
    samples = {
        sample.orig_id: sample
        for sample in dataset.iter_samples(max_samples=args.max_samples)
        if sample.orig_id in candidates
    }
    missing = sorted(set(candidates) - set(samples))
    if missing:
        raise ValueError(f"Candidate samples missing from dataset: {missing[:5]}")

    adapter = load_model_adapter(args.model_config.resolve())
    records = []
    adapter.load(args.device)
    try:
        for index, orig_id in enumerate(sorted(candidates), start=1):
            sample = samples[orig_id]
            answers = []
            for _ in range(args.repeats):
                seed_everything(torch, args.seed)
                answers.append(
                    adapter.generate(
                        sample.question,
                        sample.image,
                        max_new_tokens=args.max_new_tokens,
                    )
                )
            source = candidates[orig_id]
            records.append(
                {
                    "orig_id": orig_id,
                    "question": sample.question,
                    "canonical_answer": source["canonical_answer"],
                    "clean_replay_answer": source["clean_replay_answer"],
                    "deterministic_answers": answers,
                    "repeats_identical": len(set(answers)) == 1,
                    "matches_canonical": all(
                        answer == source["canonical_answer"] for answer in answers
                    ),
                    "matches_clean_replay": all(
                        answer == source["clean_replay_answer"] for answer in answers
                    ),
                }
            )
            if index % 25 == 0 or index == len(candidates):
                print(f"[determinism-audit] {index}/{len(candidates)}", flush=True)
    finally:
        adapter.close()

    summary = {
        "candidate_count": len(records),
        "repeat_count": args.repeats,
        "repeats_identical": sum(
            int(record["repeats_identical"]) for record in records
        ),
        "matches_canonical": sum(
            int(record["matches_canonical"]) for record in records
        ),
        "matches_clean_replay": sum(
            int(record["matches_clean_replay"]) for record in records
        ),
    }
    payload = {"summary": summary, "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(f"{args.output.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
