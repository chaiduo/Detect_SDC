#!/usr/bin/env python3
"""Regenerate paper figures from compact, Git-friendly source data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from analyze_projection_preservation import _plot_results
from plot_fault_quadrant_comparison import (
    QuadrantStats as ComparisonQuadrantStats,
)
from plot_fault_quadrant_comparison import (
    build_jobs as build_quadrant_jobs,
)
from plot_fault_quadrant_comparison import (
    plot as plot_quadrant_comparison,
)
from plot_lingoqa_cosine_across_models import (
    build_jobs as build_model_jobs,
)
from plot_lingoqa_cosine_across_models import (
    plot as plot_model_cosine,
)
from plot_online_step_ablation import plot as plot_online_steps
from plot_qwen_lingoqa_fault_quadrants import (
    QuadrantStats as LegacyQuadrantStats,
)
from plot_qwen_lingoqa_fault_quadrants import (
    plot as plot_legacy_quadrant,
)
from plot_sdc_cosine_by_layer_pair import (
    GROUP_ORDER,
    PairAccumulator,
)
from plot_sdc_cosine_by_layer_pair import (
    plot as plot_single_cosine,
)
from plot_sdc_metrics_by_layer_pair import (
    METRICS,
)
from plot_sdc_metrics_by_layer_pair import (
    plot as plot_single_metrics,
)
from plot_significant_sdc_share import (
    Counts,
)
from plot_significant_sdc_share import (
    build_jobs as build_share_jobs,
)
from plot_significant_sdc_share import (
    plot as plot_significant_share,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=REPOSITORY_ROOT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination directory. Defaults to <repository-root>/figures.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--skip-manifest-check",
        action="store_true",
    )
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(root: Path) -> None:
    manifest_path = root / "figures/figure_data_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    for item in manifest["files"]:
        path = root / item["path"]
        if not path.is_file():
            failures.append(f"missing: {item['path']}")
            continue
        if path.stat().st_size != int(item["bytes"]):
            failures.append(f"size: {item['path']}")
            continue
        if _sha256(path) != item["sha256"]:
            failures.append(f"sha256: {item['path']}")
    if failures:
        raise ValueError("Figure-data manifest validation failed:\n" + "\n".join(failures))


def replot_significant_share(
    root: Path,
    output_dir: Path,
    dpi: int,
) -> None:
    rows = _read_csv(root / "figures/significant_sdc_share.csv")
    counts = {
        f"{row['model']} / {row['dataset']}": Counts(
            all_sdc=int(row["all_sdc"]),
            significant_sdc=int(row["significant_sdc"]),
            removed_all_feature_nan=int(row["removed_all_feature_nan"]),
        )
        for row in rows
    }
    plot_significant_share(
        build_share_jobs(root),
        counts,
        output_dir / "significant_sdc_share.png",
        dpi,
    )


def _quadrant_counts(row: dict[str, str]) -> Counter[tuple[bool, bool]]:
    return Counter(
        {
            (False, False): int(row["small_non_significant"]),
            (False, True): int(row["small_significant"]),
            (True, False): int(row["large_non_significant"]),
            (True, True): int(row["large_significant"]),
        }
    )


def _select_quadrant_rows(
    rows: Iterable[dict[str, str]],
    *,
    threshold: float,
    policy: str,
) -> list[dict[str, str]]:
    selected = [
        row for row in rows if math.isclose(float(row["threshold"]), threshold) and row["non_finite_policy"] == policy
    ]
    return selected


def replot_quadrants(root: Path, output_dir: Path, dpi: int) -> None:
    rows = _read_csv(root / "figures/fault_quadrant_counts.csv")
    jobs = build_quadrant_jobs(root)
    finite_rows = _select_quadrant_rows(
        rows,
        threshold=1.0,
        policy="exclude",
    )
    finite_by_key = {
        (row["model"], row["dataset"]): ComparisonQuadrantStats(
            counts=_quadrant_counts(row),
            total=int(row["total"]),
            non_finite=int(row["non_finite"]),
            skipped=int(row["skipped"]),
        )
        for row in finite_rows
    }
    plot_quadrant_comparison(
        jobs,
        finite_by_key,
        output_dir / "fault_quadrant_comparison_non_nan.png",
        1.0,
        dpi,
    )

    legacy_outputs = {
        ("Qwen2.5-VL-7B", "EarthVQA", 1.0): "qwen25_vl_earthvqa_fault_quadrants_threshold1.png",
        ("Qwen2.5-VL-7B", "LingoQA", 1.0): "qwen_lingoqa_fault_quadrants.png",
        ("Qwen2.5-VL-7B", "VQAv2", 1.0): "qwen25_vl_vqav2_fault_quadrants_threshold1.png",
        ("LLaVA-1.5-7B", "EarthVQA", 1.0): "llava15_earthvqa_fault_quadrants_threshold1.png",
        ("LLaVA-1.5-7B", "LingoQA", 1.0): "llava15_lingoqa_fault_quadrants_threshold1.png",
        ("LLaVA-1.5-7B", "VQAv2", 1.0): "llava15_vqav2_fault_quadrants_threshold1.png",
        ("Qwen2.5-VL-7B", "LingoQA", 5.0): "qwen_lingoqa_fault_quadrants_threshold5.png",
        ("Qwen2.5-VL-7B", "LingoQA", 10.0): "qwen_lingoqa_fault_quadrants_threshold10.png",
    }
    legacy_rows = _select_quadrant_rows(
        rows,
        threshold=1.0,
        policy="count_as_large",
    )
    legacy_rows.extend(
        _select_quadrant_rows(
            rows,
            threshold=5.0,
            policy="count_as_large",
        )
    )
    legacy_rows.extend(
        _select_quadrant_rows(
            rows,
            threshold=10.0,
            policy="count_as_large",
        )
    )
    for row in legacy_rows:
        key = (row["model"], row["dataset"], float(row["threshold"]))
        output_name = legacy_outputs.get(key)
        if output_name is None:
            continue
        stats = LegacyQuadrantStats(
            counts=_quadrant_counts(row),
            total=int(row["total"]),
            skipped=int(row["skipped"]),
            non_finite=int(row["non_finite"]),
        )
        plot_legacy_quadrant(
            stats,
            output_dir / output_name,
            float(row["threshold"]),
            dpi,
        )


def _accumulator(
    mean: float,
    samples: int,
    steps: int,
) -> PairAccumulator:
    return PairAccumulator(
        total=mean * samples if samples else 0.0,
        samples=samples,
        steps=steps,
    )


def load_single_cosine(
    path: Path,
) -> dict[str, dict[tuple[int, int], PairAccumulator]]:
    output = {group: {} for group in GROUP_ORDER}
    for row in _read_csv(path):
        group = row["sdc_group"]
        pair = (int(row["src_layer"]), int(row["tgt_layer"]))
        output[group][pair] = _accumulator(
            float(row["mean_cos_sim"]),
            int(row["sample_count"]),
            int(row["finite_step_count"]),
        )
    return output


def load_metric_data(
    path: Path,
) -> dict[str, dict[str, dict[tuple[int, int], PairAccumulator]]]:
    output = {metric.key: {group: {} for group in GROUP_ORDER} for metric in METRICS}
    for row in _read_csv(path):
        metric = row["metric"]
        group = row["sdc_group"]
        pair = (int(row["src_layer"]), int(row["tgt_layer"]))
        output[metric][group][pair] = _accumulator(
            float(row["mean_value"]),
            int(row["sample_count"]),
            int(row["finite_step_count"]),
        )
    return output


def load_model_cosine(
    path: Path,
) -> tuple[
    dict[str, dict[str, dict[tuple[int, int], PairAccumulator]]],
    tuple[str, ...],
]:
    rows = _read_csv(path)
    groups = tuple(dict.fromkeys(row["sdc_group"] for row in rows))
    output: dict[
        str,
        dict[str, dict[tuple[int, int], PairAccumulator]],
    ] = defaultdict(lambda: {group: {} for group in groups})
    for row in rows:
        pair = (int(row["src_layer"]), int(row["tgt_layer"]))
        output[row["model"]][row["sdc_group"]][pair] = _accumulator(
            float(row["mean_cos_sim"]),
            int(row["sample_count"]),
            int(row["finite_step_count"]),
        )
    return dict(output), groups


def replot_layer_figures(
    root: Path,
    output_dir: Path,
    dpi: int,
) -> None:
    plot_single_cosine(
        load_single_cosine(root / "figures/qwen_lingoqa_sdc_cosine_by_layer_pair.csv"),
        output=output_dir / "qwen_lingoqa_sdc_cosine_by_layer_pair.png",
        title="Average cosine similarity by layer pair",
        dpi=dpi,
    )
    plot_single_metrics(
        load_metric_data(root / "figures/qwen_lingoqa_sdc_metrics_by_layer_pair.csv"),
        output=output_dir / "qwen_lingoqa_sdc_metrics_by_layer_pair.png",
        dpi=dpi,
    )

    names = (
        "earthvqa_cosine_by_layer_pair_three_models",
        "earthvqa_cosine_non_sdc_vs_non_significant_three_models",
        "lingoqa_cosine_by_layer_pair_three_models",
        "lingoqa_cosine_clean_sdc_significant_three_models",
        "lingoqa_cosine_non_sdc_vs_non_significant_three_models",
        "vqav2_cosine_by_layer_pair_three_models",
        "vqav2_cosine_non_sdc_vs_non_significant_three_models",
    )
    for name in names:
        csv_path = root / "figures" / f"{name}.csv"
        results, groups = load_model_cosine(csv_path)
        dataset = next(iter(_read_csv(csv_path)))["dataset"]
        plot_model_cosine(
            build_model_jobs(root, dataset),
            results,
            groups=groups,
            output=output_dir / f"{name}.png",
            dpi=dpi,
        )


def replot_projection(root: Path, output_dir: Path, dpi: int) -> None:
    summary_path = (
        root / "analysis/iclr_v2/qwen_lingoqa_projection_preservation.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    _plot_results(
        summary,
        output_dir / "qwen_lingoqa_projection_preservation.png",
        dpi=dpi,
    )


def replot_online_step(root: Path, output_dir: Path) -> None:
    aggregate = pd.read_csv(root / "figures/online_step_ablation_aggregate.csv")
    plot_online_steps(
        aggregate,
        selected_k=2,
        output_prefix=output_dir / "online_step_ablation",
    )


def main() -> int:
    args = parse_args()
    root = args.repository_root.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir is not None else root / "figures"
    if not args.skip_manifest_check:
        validate_manifest(root)
    replot_significant_share(root, output_dir, args.dpi)
    replot_quadrants(root, output_dir, args.dpi)
    replot_layer_figures(root, output_dir, args.dpi)
    replot_projection(root, output_dir, args.dpi)
    replot_online_step(root, output_dir)
    print("Regenerated all portable paper figures from compact data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
