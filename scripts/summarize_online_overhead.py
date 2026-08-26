#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


MODES = ("vanilla", "step_hook", "monitor", "predictor", "sieve")
MODE_NAMES = {
    "vanilla": "Vanilla",
    "step_hook": "Step-hook",
    "monitor": "Monitor + Projection",
    "predictor": "Predictor + Aggregation",
    "sieve": "Full SIEVE",
}
MODELS = {
    "qwen25_vl": "Qwen2.5-VL-7B",
    "internvl3": "InternVL3-8B",
    "llava15": "LLaVA-1.5-7B",
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--forward-root",
        type=Path,
        default=root / "analysis/online_overhead_20260814",
    )
    parser.add_argument(
        "--reverse-root",
        type=Path,
        default=root / "analysis/online_overhead_reverse_20260814",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "analysis/online_overhead_20260814/combined",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_260_814)
    args = parser.parse_args()
    if args.bootstrap_replicates <= 0:
        parser.error("--bootstrap-replicates must be positive")
    return args


def read_rows(path: Path, order: str) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["order"] = order
        for key in (
            "repeat",
            "sample_index",
            "tokens",
            "steps_processed",
            "detector_prediction",
        ):
            row[key] = _number(row.get(key), integer=True)
        for key in (
            "latency_ms",
            "peak_allocated_mb",
            "peak_reserved_mb",
            "detection_ready_ms",
            "detection_after_prefill_ms",
            "detector_probability",
        ):
            row[key] = _number(row.get(key))
    return rows


def _number(value: Any, *, integer: bool = False) -> int | float | None:
    if value is None or value == "":
        return None
    return int(float(value)) if integer else float(value)


def percentile(values: list[float], quantile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def latency_statistics(
    rows: list[dict[str, Any]],
    key: str,
) -> dict[str, float] | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return {
        "mean_ms": float(np.mean(values)),
        "p50_ms": percentile(values, 50),
        "p95_ms": percentile(values, 95),
    }


def bootstrap_overhead_ci(
    rows: list[dict[str, Any]],
    mode: str,
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    keyed = defaultdict(dict)
    for row in rows:
        if row["mode"] not in {"vanilla", mode}:
            continue
        key = (row["order"], int(row["repeat"]), int(row["sample_index"]))
        keyed[key][row["mode"]] = float(row["latency_ms"])
    pairs = [
        (values["vanilla"], values[mode])
        for values in keyed.values()
        if "vanilla" in values and mode in values
    ]
    if not pairs:
        raise ValueError(f"No paired observations for {mode}")
    values = np.asarray(pairs, dtype=np.float64)
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        selected = rng.integers(0, len(values), size=len(values))
        sample = values[selected]
        estimates[index] = 100.0 * (
            sample[:, 1].mean() / sample[:, 0].mean() - 1.0
        )
    low, high = np.percentile(estimates, (2.5, 97.5))
    return float(low), float(high)


def summarize_model(
    model_key: str,
    rows: list[dict[str, Any]],
    forward_rows: list[dict[str, Any]],
    *,
    bootstrap_replicates: int,
    seed: int,
) -> list[dict[str, Any]]:
    by_mode = {
        mode: [row for row in rows if row["mode"] == mode]
        for mode in MODES
    }
    baseline = by_mode["vanilla"]
    baseline_latency = np.mean(
        [float(row["latency_ms"]) for row in baseline]
    )
    baseline_throughput = sum(int(row["tokens"]) for row in baseline) / (
        sum(float(row["latency_ms"]) for row in baseline) / 1000.0
    )
    forward_baseline_peak = max(
        float(row["peak_allocated_mb"])
        for row in forward_rows
        if row["mode"] == "vanilla"
    )

    summaries = []
    for mode_index, mode in enumerate(MODES):
        selected = by_mode[mode]
        latency = latency_statistics(selected, "latency_ms")
        if latency is None:
            raise ValueError(f"No latency values for {model_key}/{mode}")
        throughput = sum(int(row["tokens"]) for row in selected) / (
            sum(float(row["latency_ms"]) for row in selected) / 1000.0
        )
        forward_peak = max(
            float(row["peak_allocated_mb"])
            for row in forward_rows
            if row["mode"] == mode
        )
        overhead = 100.0 * (latency["mean_ms"] / baseline_latency - 1.0)
        ci_low = ci_high = None
        if mode != "vanilla":
            ci_low, ci_high = bootstrap_overhead_ci(
                rows,
                mode,
                replicates=bootstrap_replicates,
                seed=seed + mode_index,
            )
        matches = [
            row["answer_matches_vanilla"].lower() == "true"
            for row in selected
            if row.get("answer_matches_vanilla") not in (None, "")
        ]
        summaries.append(
            {
                "model_key": model_key,
                "model": MODELS[model_key],
                "mode": mode,
                "mode_name": MODE_NAMES[mode],
                "observations": len(selected),
                "latency_mean_ms": latency["mean_ms"],
                "latency_p50_ms": latency["p50_ms"],
                "latency_p95_ms": latency["p95_ms"],
                "latency_overhead_percent": overhead,
                "overhead_ci95_low_percent": ci_low,
                "overhead_ci95_high_percent": ci_high,
                "tokens_per_second": throughput,
                "throughput_change_percent": 100.0
                * (throughput / baseline_throughput - 1.0),
                "extra_peak_allocated_mb": (
                    forward_peak - forward_baseline_peak
                ),
                "answer_match_rate": (
                    None if not matches else float(np.mean(matches))
                ),
                "detection_ready": latency_statistics(
                    selected, "detection_ready_ms"
                ),
                "detection_after_prefill": latency_statistics(
                    selected, "detection_after_prefill_ms"
                ),
                "steps_processed": {
                    str(step): sum(
                        row.get("steps_processed") == step for row in selected
                    )
                    for step in sorted(
                        {
                            int(row["steps_processed"])
                            for row in selected
                            if row.get("steps_processed") is not None
                        }
                    )
                },
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def flatten_component_rows(
    summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    fields = (
        "model",
        "mode_name",
        "observations",
        "latency_mean_ms",
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_overhead_percent",
        "overhead_ci95_low_percent",
        "overhead_ci95_high_percent",
        "tokens_per_second",
        "throughput_change_percent",
        "extra_peak_allocated_mb",
        "answer_match_rate",
    )
    return [{field: item[field] for field in fields} for item in summaries]


def deployment_rows(
    summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for item in summaries:
        if item["mode"] != "sieve":
            continue
        ready = item["detection_ready"]
        after_prefill = item["detection_after_prefill"]
        rows.append(
            {
                "model": item["model"],
                "vanilla_latency_mean_ms": next(
                    candidate["latency_mean_ms"]
                    for candidate in summaries
                    if candidate["model_key"] == item["model_key"]
                    and candidate["mode"] == "vanilla"
                ),
                "sieve_latency_mean_ms": item["latency_mean_ms"],
                "end_to_end_overhead_percent": item[
                    "latency_overhead_percent"
                ],
                "overhead_ci95_low_percent": item[
                    "overhead_ci95_low_percent"
                ],
                "overhead_ci95_high_percent": item[
                    "overhead_ci95_high_percent"
                ],
                "throughput_change_percent": item[
                    "throughput_change_percent"
                ],
                "detection_ready_mean_ms": ready["mean_ms"],
                "detection_ready_p95_ms": ready["p95_ms"],
                "detection_after_prefill_mean_ms": after_prefill["mean_ms"],
                "detection_after_prefill_p95_ms": after_prefill["p95_ms"],
                "extra_peak_allocated_mb": item[
                    "extra_peak_allocated_mb"
                ],
                "steps_processed": json.dumps(
                    item["steps_processed"], sort_keys=True
                ),
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    all_summaries = []
    for model_index, model_key in enumerate(MODELS):
        forward = read_rows(
            args.forward_root / model_key / "samples.csv", "forward"
        )
        reverse = read_rows(
            args.reverse_root / model_key / "samples.csv", "reverse"
        )
        all_summaries.extend(
            summarize_model(
                model_key,
                forward + reverse,
                forward,
                bootstrap_replicates=args.bootstrap_replicates,
                seed=args.seed + 100 * model_index,
            )
        )

    deployed = deployment_rows(all_summaries)
    aggregate = {
        "model_count": len(deployed),
        "mean_end_to_end_overhead_percent": float(
            np.mean(
                [row["end_to_end_overhead_percent"] for row in deployed]
            )
        ),
        "mean_throughput_change_percent": float(
            np.mean([row["throughput_change_percent"] for row in deployed])
        ),
        "mean_step_hook_overhead_percent": float(
            np.mean(
                [
                    row["latency_overhead_percent"]
                    for row in all_summaries
                    if row["mode"] == "step_hook"
                ]
            )
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "component_breakdown.csv",
        flatten_component_rows(all_summaries),
    )
    write_csv(args.output_dir / "deployment_summary.csv", deployed)
    (args.output_dir / "combined_summary.json").write_text(
        json.dumps(
            {
                "protocol": {
                    "dataset": "LingoQA",
                    "gpu": "NVIDIA H20",
                    "batch_size": 1,
                    "max_new_tokens": 50,
                    "samples_per_model": 50,
                    "repeats_per_order": 2,
                    "orders": ["forward", "reverse"],
                    "observations_per_model_mode": 200,
                    "warmup_samples": 5,
                    "online_steps": 2,
                    "monitored_layers": [6, 7, 22, 23, 24, 25, 26, 27],
                    "bootstrap_replicates": args.bootstrap_replicates,
                    "bootstrap_seed": args.seed,
                    "memory_note": (
                        "Peak-memory deltas use the forward order because "
                        "the reverse order loads the predictor before Vanilla."
                    ),
                },
                "aggregate": aggregate,
                "deployment": deployed,
                "components": all_summaries,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(aggregate, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
