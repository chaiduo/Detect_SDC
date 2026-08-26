#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from benchmark_online_overhead import (
    MODEL_INFO,
    PAIRS,
    detector_config,
    prepare_detector,
    preload_samples,
    run_generation,
    write_rows,
)
from detect_sdc.adapters import load_dataset_adapter, load_model_adapter
from detect_sdc.online_monitor import OnlineSieveMonitor
from detect_sdc.pipeline.injection import load_mapping_model
from detect_sdc.pipeline.jobs import load_pipeline_job


PROFILES = ("baseline", "full_72d", "cos_sim_mean_6d")
ORDERS = {
    "forward": PROFILES,
    "reverse": tuple(reversed(PROFILES)),
}


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
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_260_815)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def load_expected_metrics(path: Path, *, compact: bool) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if compact:
        return value["valid_metrics"]
    return value["valid_full_metrics"]["target_significant_sdc"]


def run_profile(
    *,
    profile: str,
    order: str,
    adapter: Any,
    monitor: OnlineSieveMonitor | None,
    warmup: list[Any],
    samples: list[Any],
    repeats: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    if monitor is not None:
        monitor.register()
    try:
        for sample in warmup:
            run_generation(
                adapter,
                sample,
                max_new_tokens,
                monitor,
                measured=False,
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
                answer = result.pop("answer")
                rows.append(
                    {
                        "profile": profile,
                        "order": order,
                        "repeat": repeat,
                        "sample_index": sample_index,
                        "orig_id": sample.orig_id,
                        "answer_sha256": hashlib.sha256(
                            answer.encode("utf-8")
                        ).hexdigest(),
                        **result,
                    }
                )
        return rows
    finally:
        if monitor is not None:
            monitor.unregister()


def paired_ci(
    rows: list[dict[str, Any]],
    numerator: str,
    denominator: str,
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    keyed: dict[tuple[str, int, int], dict[str, float]] = {}
    for row in rows:
        key = (
            str(row["order"]),
            int(row["repeat"]),
            int(row["sample_index"]),
        )
        keyed.setdefault(key, {})[str(row["profile"])] = float(
            row["latency_ms"]
        )
    pairs = np.asarray(
        [
            (values[denominator], values[numerator])
            for values in keyed.values()
            if denominator in values and numerator in values
        ],
        dtype=np.float64,
    )
    if len(pairs) == 0:
        raise ValueError(f"No pairs for {denominator}->{numerator}")
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates)
    for index in range(replicates):
        selected = pairs[
            rng.integers(0, len(pairs), size=len(pairs))
        ]
        estimates[index] = 100.0 * (
            selected[:, 1].mean() / selected[:, 0].mean() - 1.0
        )
    low, high = np.percentile(estimates, (2.5, 97.5))
    return float(low), float(high)


def summarize(
    rows: list[dict[str, Any]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    by_profile = {
        profile: [row for row in rows if row["profile"] == profile]
        for profile in PROFILES
    }
    baseline = by_profile["baseline"]
    baseline_mean = float(
        np.mean([float(row["latency_ms"]) for row in baseline])
    )
    baseline_tps = sum(int(row["tokens"]) for row in baseline) / (
        sum(float(row["latency_ms"]) for row in baseline) / 1000.0
    )
    result: dict[str, Any] = {}
    for index, profile in enumerate(PROFILES):
        selected = by_profile[profile]
        latency = np.asarray(
            [float(row["latency_ms"]) for row in selected]
        )
        throughput = sum(int(row["tokens"]) for row in selected) / (
            latency.sum() / 1000.0
        )
        item = {
            "observations": len(selected),
            "latency_mean_ms": float(latency.mean()),
            "latency_p50_ms": float(np.percentile(latency, 50)),
            "latency_p95_ms": float(np.percentile(latency, 95)),
            "latency_overhead_vs_baseline_percent": 100.0
            * (float(latency.mean()) / baseline_mean - 1.0),
            "tokens_per_second": float(throughput),
            "throughput_change_vs_baseline_percent": 100.0
            * (throughput / baseline_tps - 1.0),
        }
        if profile != "baseline":
            low, high = paired_ci(
                rows,
                profile,
                "baseline",
                replicates=replicates,
                seed=seed + index,
            )
            item["overhead_ci95_low_percent"] = low
            item["overhead_ci95_high_percent"] = high
        result[profile] = item
    low, high = paired_ci(
        rows,
        "cos_sim_mean_6d",
        "full_72d",
        replicates=replicates,
        seed=seed + 10,
    )
    full_mean = result["full_72d"]["latency_mean_ms"]
    compact_mean = result["cos_sim_mean_6d"]["latency_mean_ms"]
    result["direct_6d_vs_72d"] = {
        "latency_change_percent": 100.0
        * (compact_mean / full_mean - 1.0),
        "ci95_low_percent": low,
        "ci95_high_percent": high,
    }
    return result


def main() -> int:
    args = parse_args()
    root = args.repository_root.resolve()
    config_path = args.config.resolve()
    output_root = args.output_root.resolve()
    job = load_pipeline_job(
        config_path,
        args.job,
        repository_root=root,
    )
    display_name, model_directory = MODEL_INFO[job.model_name]
    online_root = (
        root
        / model_directory
        / "online_step_ablation_20260814/LingoQA"
        / f"k_{args.online_steps}"
    )
    compact_root = (
        root
        / model_directory
        / "online_cosine_mean_depth_ablation_20260815"
        / "LingoQA/depth_6/detector_metadata.json"
    )
    config = detector_config(config_path, job.model_name)
    full_detector, full_columns = prepare_detector(
        online_root / "train_data/train.csv",
        online_root / "train_data/valid_fixed_k50.csv",
        output_root / "detectors/full_72d",
        config,
        "full",
        load_expected_metrics(
            online_root / "output/metrics_summary.json",
            compact=False,
        ),
    )
    compact_detector, compact_columns = prepare_detector(
        online_root / "train_data/train.csv",
        online_root / "train_data/valid_fixed_k50.csv",
        output_root / "detectors/cos_sim_mean_6d",
        config,
        "cos_sim_mean",
        load_expected_metrics(compact_root, compact=True),
    )

    dataset = load_dataset_adapter(job.dataset_config_path)
    samples = preload_samples(
        dataset,
        args.warmup_samples + args.samples,
    )
    warmup = samples[: args.warmup_samples]
    measured = samples[args.warmup_samples :]
    adapter = load_model_adapter(job.model_config_path)
    adapter.load(args.device)
    predictor = load_mapping_model(
        job.injection_config,
        job.paths.mapping_model,
        device=args.device,
    ).to(args.device).eval()

    rows: list[dict[str, Any]] = []
    try:
        for sample in measured:
            run_generation(
                adapter,
                sample,
                job.max_new_tokens,
                None,
                measured=False,
            )
        for order_name, profiles in ORDERS.items():
            for profile in profiles:
                monitor = None
                if profile != "baseline":
                    compact = profile == "cos_sim_mean_6d"
                    monitor = OnlineSieveMonitor(
                        adapter.model,
                        mode="sieve",
                        layer_pairs=PAIRS,
                        projection_dim=job.projection_dim,
                        projection_seed=job.profiler_seed,
                        max_steps=args.online_steps,
                        predictor=predictor,
                        detector=(
                            compact_detector if compact else full_detector
                        ),
                        detector_feature_columns=(
                            compact_columns if compact else full_columns
                        ),
                        feature_profile=(
                            "cos_sim_mean" if compact else "full"
                        ),
                    )
                print(
                    f"[profile-comparison] {display_name} "
                    f"order={order_name} profile={profile}",
                    flush=True,
                )
                rows.extend(
                    run_profile(
                        profile=profile,
                        order=order_name,
                        adapter=adapter,
                        monitor=monitor,
                        warmup=warmup,
                        samples=measured,
                        repeats=args.repeats,
                        max_new_tokens=job.max_new_tokens,
                    )
                )
                write_rows(output_root / "samples.csv", rows)
    finally:
        adapter.close()
        predictor = None
        gc.collect()
        torch.cuda.empty_cache()

    summary = {
        "model": display_name,
        "model_key": job.model_name,
        "gpu_physical_id": os.environ.get("GPU_PHYSICAL_ID"),
        "feature_counts": {
            "full_72d": len(full_columns),
            "cos_sim_mean_6d": len(compact_columns),
        },
        "detector_max_depth": 6,
        "orders": ORDERS,
        "samples": args.samples,
        "repeats_per_order": args.repeats,
        "bootstrap_replicates": args.bootstrap_replicates,
        "metrics": summarize(
            rows,
            replicates=args.bootstrap_replicates,
            seed=args.seed,
        ),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
