#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import xgboost as xgb

from detect_sdc.adapters import load_dataset_adapter, load_model_adapter
from detect_sdc.adapters.models.images import load_pil_image
from detect_sdc.online_monitor import OnlineSieveMonitor
from detect_sdc.pipeline.injection import load_mapping_model
from detect_sdc.pipeline.jobs import load_pipeline_job


MODES = ("vanilla", "step_hook", "monitor", "predictor", "sieve")
PAIRS = ((6, 7), (22, 23), (23, 24), (24, 25), (25, 26), (26, 27))
MODEL_INFO = {
    "qwen25_vl": ("Qwen2.5-VL-7B", "Qwen2.5-VL-7B"),
    "internvl3": ("InternVL3-8B", "InternVL3-8B"),
    "llava15": ("LLaVA-1.5-7B", "llava-v1.5-7B"),
}


class DetectorAdapter:
    def __init__(self, booster: xgb.Booster) -> None:
        self.booster = booster

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        positive = self.booster.inplace_predict(
            values, validate_features=False
        )
        return np.stack((1.0 - positive, positive), axis=1)


def load_deployed_detector(
    summary_path: Path,
) -> tuple[DetectorAdapter, tuple[str, ...], float, int]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    booster = xgb.Booster()
    booster.load_model(summary["model_path"])
    return (
        DetectorAdapter(booster),
        tuple(summary["feature_columns"]),
        float(summary["threshold_calibration"]["threshold"]),
        int(summary["training"]["parameters"]["max_depth"]),
    )


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--job",
        choices=(
            "qwen25_vl_lingoqa",
            "internvl3_lingoqa",
            "llava15_lingoqa",
        ),
        required=True,
    )
    parser.add_argument("--repository-root", type=Path, default=root)
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configs/experiments/current.yaml",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--warmup-samples", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--online-steps", type=int, default=2)
    parser.add_argument(
        "--feature-profile",
        choices=("full",),
        default="full",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=MODES,
        default=list(MODES),
    )
    parser.add_argument(
        "--mode-order",
        choices=("forward", "reverse"),
        default="forward",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    for name in ("samples", "warmup_samples", "repeats", "online_steps"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if "vanilla" not in args.modes:
        parser.error("--modes must include vanilla")
    if len(set(args.modes)) != len(args.modes):
        parser.error("--modes must be unique")
    return args


def preload_samples(dataset: Any, count: int) -> list[Any]:
    samples = []
    for sample in dataset.iter_samples(max_samples=count):
        samples.append(
            type(sample)(
                orig_id=sample.orig_id,
                semantic_group_id=sample.semantic_group_id,
                question=sample.question,
                ground_truth=sample.ground_truth,
                image=load_pil_image(sample.image),
                metadata=sample.metadata,
            )
        )
    if len(samples) != count:
        raise ValueError(f"Expected {count} samples, got {len(samples)}")
    return samples


def token_count(adapter: Any, answer: str) -> int:
    tokenizer = getattr(adapter, "_tokenizer", None)
    if tokenizer is None:
        tokenizer = getattr(
            getattr(adapter, "_processor", None), "tokenizer", None
        )
    if tokenizer is None:
        raise TypeError("Cannot locate tokenizer")
    return max(1, len(tokenizer.encode(answer, add_special_tokens=False)))


def run_generation(
    adapter: Any,
    sample: Any,
    max_new_tokens: int,
    monitor: OnlineSieveMonitor | None,
    *,
    measured: bool,
) -> dict[str, Any]:
    torch.cuda.synchronize()
    if measured:
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    if monitor is not None:
        monitor.start_sample(start)
    answer = adapter.generate(
        sample.question,
        sample.image,
        max_new_tokens=max_new_tokens,
    )
    if monitor is not None:
        monitor.finish_sample()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    result: dict[str, Any] = {
        "answer": answer,
        "latency_ms": elapsed * 1000.0,
    }
    if measured:
        result.update(
            {
                "tokens": token_count(adapter, answer),
                "peak_allocated_mb": (
                    torch.cuda.max_memory_allocated() / (1024.0**2)
                ),
                "peak_reserved_mb": (
                    torch.cuda.max_memory_reserved() / (1024.0**2)
                ),
            }
        )
    if monitor is not None:
        result.update(
            {
                "steps_processed": monitor.steps_processed,
                "detection_ready_ms": (
                    None
                    if monitor.detection_ready_seconds is None
                    else monitor.detection_ready_seconds * 1000.0
                ),
                "detection_after_prefill_ms": (
                    None
                    if monitor.detection_after_prefill_seconds is None
                    else monitor.detection_after_prefill_seconds * 1000.0
                ),
                "detector_probability": monitor.detector_probability,
                "detector_prediction": monitor.detector_prediction,
            }
        )
    return result


def benchmark_mode(
    mode: str,
    adapter: Any,
    warmup: list[Any],
    samples: list[Any],
    repeats: int,
    max_new_tokens: int,
    monitor: OnlineSieveMonitor | None,
    vanilla_answers: dict[tuple[int, int], str],
) -> list[dict[str, Any]]:
    if monitor is not None:
        monitor.register()
    try:
        for sample in warmup:
            run_generation(
                adapter, sample, max_new_tokens, monitor, measured=False
            )
        rows = []
        for repeat in range(repeats):
            for sample_index, sample in enumerate(samples):
                result = run_generation(
                    adapter,
                    sample,
                    max_new_tokens,
                    monitor,
                    measured=True,
                )
                identity = (repeat, sample_index)
                if mode == "vanilla":
                    vanilla_answers[identity] = result["answer"]
                baseline = vanilla_answers.get(identity)
                matches_vanilla = (
                    None
                    if baseline is None
                    else result["answer"] == baseline
                )
                rows.append(
                    {
                        "mode": mode,
                        "repeat": repeat,
                        "sample_index": sample_index,
                        "orig_id": sample.orig_id,
                        "answer_sha256": hashlib.sha256(
                            result["answer"].encode("utf-8")
                        ).hexdigest(),
                        "answer_matches_vanilla": matches_vanilla,
                        **{
                            key: value
                            for key, value in result.items()
                            if key != "answer"
                        },
                    }
                )
        return rows
    finally:
        if monitor is not None:
            monitor.unregister()


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = {}
    for mode in MODES:
        selected = [row for row in rows if row["mode"] == mode]
        if not selected:
            continue
        latency = np.asarray(
            [float(row["latency_ms"]) for row in selected]
        )
        elapsed = float(latency.sum() / 1000.0)
        matches = [
            bool(row["answer_matches_vanilla"])
            for row in selected
            if row.get("answer_matches_vanilla") is not None
        ]
        summary = {
            "samples": len(selected),
            "answer_match_rate": (
                None if not matches else float(statistics.mean(matches))
            ),
            "latency_mean_ms": float(latency.mean()),
            "latency_p50_ms": float(np.percentile(latency, 50)),
            "latency_p95_ms": float(np.percentile(latency, 95)),
            "tokens_per_second": float(
                sum(int(row["tokens"]) for row in selected) / elapsed
            ),
            "peak_allocated_mean_mb": float(
                statistics.mean(
                    float(row["peak_allocated_mb"]) for row in selected
                )
            ),
            "peak_allocated_max_mb": float(
                max(float(row["peak_allocated_mb"]) for row in selected)
            ),
            "peak_reserved_max_mb": float(
                max(float(row["peak_reserved_mb"]) for row in selected)
            ),
        }
        ready = [
            float(row["detection_ready_ms"])
            for row in selected
            if row.get("detection_ready_ms") is not None
        ]
        after_prefill = [
            float(row["detection_after_prefill_ms"])
            for row in selected
            if row.get("detection_after_prefill_ms") is not None
        ]
        if ready:
            summary.update(_latency_stats("detection_ready", ready))
        if after_prefill:
            summary.update(
                _latency_stats("detection_after_prefill", after_prefill)
            )
        summaries[mode] = summary

    if "vanilla" not in summaries:
        return summaries
    baseline = summaries["vanilla"]
    for mode in MODES[1:]:
        if mode not in summaries:
            continue
        item = summaries[mode]
        item["latency_overhead_percent"] = 100.0 * (
            item["latency_mean_ms"] / baseline["latency_mean_ms"] - 1.0
        )
        item["throughput_change_percent"] = 100.0 * (
            item["tokens_per_second"] / baseline["tokens_per_second"] - 1.0
        )
        item["extra_peak_allocated_mb"] = (
            item["peak_allocated_max_mb"]
            - baseline["peak_allocated_max_mb"]
        )
    return summaries


def _latency_stats(prefix: str, values: list[float]) -> dict[str, float]:
    array = np.asarray(values)
    return {
        f"{prefix}_mean_ms": float(array.mean()),
        f"{prefix}_p50_ms": float(np.percentile(array, 50)),
        f"{prefix}_p95_ms": float(np.percentile(array, 95)),
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(
        dict.fromkeys(key for row in rows for key in row)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def refresh_answer_matches(rows: list[dict[str, Any]]) -> None:
    baseline = {
        (int(row["repeat"]), int(row["sample_index"])): row["answer_sha256"]
        for row in rows
        if row["mode"] == "vanilla"
    }
    for row in rows:
        identity = (int(row["repeat"]), int(row["sample_index"]))
        expected = baseline.get(identity)
        row["answer_matches_vanilla"] = (
            None
            if expected is None
            else row["answer_sha256"] == expected
        )


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    root = args.repository_root.resolve()
    config_path = args.config.resolve()
    job = load_pipeline_job(
        config_path,
        args.job,
        repository_root=root,
    )
    display_name, _ = MODEL_INFO[job.model_name]
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else (
            root
            / "analysis/iclr_v2/online_overhead"
            / args.mode_order
            / job.model_name
        )
    )
    (
        detector,
        detector_columns,
        detector_threshold,
        detector_depth,
    ) = load_deployed_detector(
        job.paths.labeled_output.parent.parent / "output/metrics_summary.json"
    )

    dataset = load_dataset_adapter(job.dataset_config_path)
    samples = preload_samples(
        dataset, args.warmup_samples + args.samples
    )
    warmup = samples[: args.warmup_samples]
    measured = samples[args.warmup_samples :]
    adapter = load_model_adapter(job.model_config_path)
    adapter.load(args.device)

    print(f"[benchmark] {display_name} prewarming measured samples", flush=True)
    for sample in measured:
        run_generation(adapter, sample, job.max_new_tokens, None, measured=False)

    predictor = None
    rows: list[dict[str, Any]] = []
    vanilla_answers: dict[tuple[int, int], str] = {}
    selected_modes = tuple(args.modes)
    mode_order = (
        selected_modes
        if args.mode_order == "forward"
        else tuple(reversed(selected_modes))
    )
    try:
        for mode in mode_order:
            if mode in {"predictor", "sieve"} and predictor is None:
                predictor = load_mapping_model(
                    job.injection_config,
                    job.paths.mapping_model,
                    device=args.device,
                ).to(args.device).eval()

            monitor = None
            if mode != "vanilla":
                monitor = OnlineSieveMonitor(
                    adapter.model,
                    mode=mode,
                    layer_pairs=PAIRS,
                    projection_dim=job.projection_dim,
                    projection_seed=job.profiler_seed,
                    max_steps=args.online_steps,
                    predictor=(
                        predictor
                        if mode in {"predictor", "sieve"}
                        else None
                    ),
                    detector=detector if mode == "sieve" else None,
                    detector_feature_columns=(
                        detector_columns if mode == "sieve" else None
                    ),
                    detector_threshold=detector_threshold,
                    feature_profile=args.feature_profile,
                )
            print(f"[benchmark] {display_name} mode={mode}", flush=True)
            rows.extend(
                benchmark_mode(
                    mode,
                    adapter,
                    warmup,
                    measured,
                    args.repeats,
                    job.max_new_tokens,
                    monitor,
                    vanilla_answers,
                )
            )
            refresh_answer_matches(rows)
            write_rows(output_root / "samples.csv", rows)
            _write_json(
                output_root / "summary.json",
                {
                    "model": display_name,
                    "model_key": job.model_name,
                    "dataset": "LingoQA",
                    "gpu_physical_id": os.environ.get("GPU_PHYSICAL_ID"),
                    "cuda_visible_devices": os.environ.get(
                        "CUDA_VISIBLE_DEVICES"
                    ),
                    "samples_per_repeat": args.samples,
                    "warmup_samples": args.warmup_samples,
                    "repeats": args.repeats,
                    "max_new_tokens": job.max_new_tokens,
                    "online_steps": args.online_steps,
                    "feature_profile": args.feature_profile,
                    "detector_depth": detector_depth,
                    "detector_feature_count": len(detector_columns),
                    "mode_order": args.mode_order,
                    "configured_modes": selected_modes,
                    "layer_pairs": PAIRS,
                    "monitored_layers": sorted(
                        {layer for pair in PAIRS for layer in pair}
                    ),
                    "modes_completed": [
                        candidate
                        for candidate in MODES
                        if any(row["mode"] == candidate for row in rows)
                    ],
                    "metrics": summarize(rows),
                },
            )
    finally:
        adapter.close()
        predictor = None
        gc.collect()
        torch.cuda.empty_cache()

    print(json.dumps(summarize(rows), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
