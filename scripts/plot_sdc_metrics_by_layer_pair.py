#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_sdc_cosine_by_layer_pair import (
    GROUP_LABELS,
    GROUP_ORDER,
    GROUP_STYLES,
    PairAccumulator,
    classify_record,
    configure_style,
)


LayerPair = tuple[int, int]


@dataclass(frozen=True)
class MetricSpec:
    key: str
    title: str
    ylabel: str
    transform: Callable[[float], float]
    log_scale: bool = False


METRICS = (
    MetricSpec(
        key="cos_sim",
        title="(a) Cosine similarity",
        ylabel="Mean cosine similarity",
        transform=lambda value: value,
    ),
    MetricSpec(
        key="mean_diff",
        title="(b) Mean difference",
        ylabel="Mean absolute difference (log)",
        transform=abs,
        log_scale=True,
    ),
    MetricSpec(
        key="std_diff",
        title="(c) Standard deviation difference",
        ylabel="Std. absolute difference (log)",
        transform=abs,
        log_scale=True,
    ),
    MetricSpec(
        key="l2_distance",
        title="(d) L2 distance",
        ylabel="Mean L2 distance (log)",
        transform=lambda value: value,
        log_scale=True,
    ),
)


@dataclass
class CollectionStats:
    records_read: int = 0
    injected_records: int = 0
    records_used: int = 0
    clean_records_skipped: int = 0
    invalid_labels_skipped: int = 0
    missing_telemetry_skipped: int = 0
    non_finite_values_skipped: int = 0


MetricAccumulators = dict[
    str,
    dict[str, dict[LayerPair, PairAccumulator]],
]


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Plot predictor-target differences by adjacent layer pair for "
            "non-SDC, non-significant SDC, and significant SDC."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=repository_root / "Qwen2.5-VL-7B/LingoQA/json/labels.jsonl",
        help="Labeled injection JSONL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            repository_root
            / "figures/qwen_lingoqa_sdc_metrics_by_layer_pair.png"
        ),
        help="Output PNG path. A PDF with the same stem is also written.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to the output path with .csv suffix.",
    )
    parser.add_argument(
        "--last-k-steps",
        type=int,
        default=None,
        help="Use only the last K decode steps. By default, use all steps.",
    )
    parser.add_argument(
        "--include-non-adjacent",
        action="store_true",
        help="Include non-adjacent layer pairs if present.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if args.last_k_steps is not None and args.last_k_steps <= 0:
        parser.error("--last-k-steps must be positive")
    return args


def _valid_records(
    records: Iterable[Mapping[str, object]],
    *,
    last_k_steps: int | None,
    adjacent_only: bool,
) -> list[Mapping[str, object]]:
    valid: list[Mapping[str, object]] = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        try:
            step = int(record["step"])
            src = int(record["src_layer"])
            tgt = int(record["tgt_layer"])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if adjacent_only and tgt != src + 1:
            continue
        valid.append(record)

    if last_k_steps is not None and valid:
        steps = sorted({int(record["step"]) for record in valid})
        selected_steps = set(steps[-last_k_steps:])
        valid = [
            record
            for record in valid
            if int(record["step"]) in selected_steps
        ]
    return valid


def collect_sample_means(
    records: Iterable[Mapping[str, object]],
    *,
    last_k_steps: int | None,
    adjacent_only: bool,
) -> tuple[dict[str, dict[LayerPair, tuple[float, int]]], int]:
    valid_records = _valid_records(
        records,
        last_k_steps=last_k_steps,
        adjacent_only=adjacent_only,
    )
    sums: dict[str, dict[LayerPair, float]] = {
        metric.key: defaultdict(float)
        for metric in METRICS
    }
    counts: dict[str, dict[LayerPair, int]] = {
        metric.key: defaultdict(int)
        for metric in METRICS
    }
    non_finite = 0

    for record in valid_records:
        pair = (int(record["src_layer"]), int(record["tgt_layer"]))
        for metric in METRICS:
            try:
                value = float(record[metric.key])
            except (KeyError, TypeError, ValueError, OverflowError):
                non_finite += 1
                continue
            if not math.isfinite(value):
                non_finite += 1
                continue
            value = metric.transform(value)
            sums[metric.key][pair] += value
            counts[metric.key][pair] += 1

    means: dict[str, dict[LayerPair, tuple[float, int]]] = {}
    for metric in METRICS:
        means[metric.key] = {
            pair: (sums[metric.key][pair] / count, count)
            for pair, count in counts[metric.key].items()
            if count > 0
        }
    return means, non_finite


def collect(
    path: Path,
    *,
    last_k_steps: int | None,
    adjacent_only: bool,
) -> tuple[MetricAccumulators, CollectionStats]:
    accumulators: MetricAccumulators = {
        metric.key: {
            group: defaultdict(PairAccumulator)
            for group in GROUP_ORDER
        }
        for metric in METRICS
    }
    stats = CollectionStats()

    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            stats.records_read += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error

            group = classify_record(record)
            if group is None:
                stats.clean_records_skipped += 1
                continue
            stats.injected_records += 1
            if group == "invalid":
                stats.invalid_labels_skipped += 1
                continue

            telemetry = record.get("mean_std_cos")
            records = (
                telemetry.get("records")
                if isinstance(telemetry, Mapping)
                else None
            )
            if not isinstance(records, list) or not records:
                stats.missing_telemetry_skipped += 1
                continue

            sample_means, non_finite = collect_sample_means(
                records,
                last_k_steps=last_k_steps,
                adjacent_only=adjacent_only,
            )
            stats.non_finite_values_skipped += non_finite
            if not any(sample_means.values()):
                stats.missing_telemetry_skipped += 1
                continue

            for metric in METRICS:
                for pair, (sample_mean, step_count) in sample_means[
                    metric.key
                ].items():
                    accumulators[metric.key][group][pair].add(
                        sample_mean,
                        step_count,
                    )
            stats.records_used += 1

    return accumulators, stats


def all_pairs(accumulators: MetricAccumulators) -> list[LayerPair]:
    return sorted(
        {
            pair
            for metric_accumulators in accumulators.values()
            for group_accumulators in metric_accumulators.values()
            for pair in group_accumulators
        }
    )


def write_summary(
    path: Path,
    accumulators: MetricAccumulators,
) -> None:
    pairs = all_pairs(accumulators)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "metric",
                "src_layer",
                "tgt_layer",
                "sdc_group",
                "mean_value",
                "sample_count",
                "finite_step_count",
            ]
        )
        for metric in METRICS:
            for pair in pairs:
                for group in GROUP_ORDER:
                    accumulator = accumulators[metric.key][group].get(
                        pair,
                        PairAccumulator(),
                    )
                    writer.writerow(
                        [
                            metric.key,
                            pair[0],
                            pair[1],
                            group,
                            accumulator.mean,
                            accumulator.samples,
                            accumulator.steps,
                        ]
                    )


def plot(
    accumulators: MetricAccumulators,
    *,
    output: Path,
    dpi: int,
) -> None:
    pairs = all_pairs(accumulators)
    if not pairs:
        raise ValueError("No finite layer-pair metrics were found")

    configure_style()
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7.5,
            "lines.linewidth": 0.8,
        }
    )

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(5.5, 4.25),
        sharex=True,
    )
    positions = list(range(len(pairs)))
    for axis, metric in zip(axes.flat, METRICS):
        for group in GROUP_ORDER:
            values = [
                accumulators[metric.key][group]
                .get(pair, PairAccumulator())
                .mean
                for pair in pairs
            ]
            style = GROUP_STYLES[group]
            axis.plot(
                positions,
                values,
                label=GROUP_LABELS[group],
                color=style["color"],
                marker=style["marker"],
                markersize=2.2,
                markeredgewidth=0.35,
            )
        axis.set_title(metric.title, pad=2)
        axis.set_ylabel(metric.ylabel, labelpad=2)
        if metric.log_scale:
            axis.set_yscale("log")
        axis.grid(
            axis="both",
            linestyle="--",
            linewidth=0.35,
            alpha=0.3,
        )
        axis.margins(x=0.01)

    tick_positions = list(range(0, len(pairs), 2))
    if tick_positions[-1] != len(pairs) - 1:
        tick_positions.append(len(pairs) - 1)
    tick_labels = [
        f"({pairs[index][0]},{pairs[index][1]})"
        for index in tick_positions
    ]
    for axis in axes[-1]:
        axis.set_xticks(tick_positions)
        axis.set_xticklabels(tick_labels, rotation=90)
        axis.set_xlabel("Layer pair", labelpad=2)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=True,
        handlelength=2.2,
        columnspacing=1.2,
    )
    figure.subplots_adjust(
        left=0.11,
        right=0.985,
        bottom=0.13,
        top=0.91,
        wspace=0.27,
        hspace=0.25,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    figure.savefig(
        output.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(figure)


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output = args.output.resolve()
    summary_output = (
        args.summary_output.resolve()
        if args.summary_output is not None
        else output.with_suffix(".csv")
    )
    if not input_path.is_file():
        raise FileNotFoundError(f"Missing labeled injection file: {input_path}")

    accumulators, stats = collect(
        input_path,
        last_k_steps=args.last_k_steps,
        adjacent_only=not args.include_non_adjacent,
    )
    write_summary(summary_output, accumulators)
    plot(accumulators, output=output, dpi=args.dpi)

    print(f"Input: {input_path}")
    print(f"Collection stats: {stats}")
    print(f"Summary: {summary_output}")
    print(f"PNG: {output}")
    print(f"PDF: {output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
