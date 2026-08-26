#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


LayerPair = tuple[int, int]

GROUP_ORDER = (
    "non_sdc",
    "non_significant_sdc",
    "significant_sdc",
)
GROUP_LABELS = {
    "non_sdc": "Non-SDC",
    "non_significant_sdc": "Non-significant SDC",
    "significant_sdc": "Significant SDC",
}
GROUP_STYLES = {
    "non_sdc": {"color": "#2878B5", "marker": "o"},
    "non_significant_sdc": {"color": "#F28E2B", "marker": "s"},
    "significant_sdc": {"color": "#C44E52", "marker": "^"},
}


@dataclass
class PairAccumulator:
    total: float = 0.0
    samples: int = 0
    steps: int = 0

    def add(self, sample_mean: float, step_count: int) -> None:
        self.total += sample_mean
        self.samples += 1
        self.steps += step_count

    @property
    def mean(self) -> float:
        return self.total / self.samples if self.samples else math.nan


@dataclass
class CollectionStats:
    records_read: int = 0
    injected_records: int = 0
    records_used: int = 0
    clean_records_skipped: int = 0
    invalid_labels_skipped: int = 0
    missing_telemetry_skipped: int = 0
    non_finite_values_skipped: int = 0


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Plot mean predictor-target cosine similarity by adjacent layer "
            "pair for non-SDC, non-significant SDC, and significant SDC."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=repository_root / "Qwen2.5-VL-7B/LingoQA/json/labels.jsonl",
        help=(
            "Labeled injection JSONL. It must contain is_sdc, significance, "
            "injected, and mean_std_cos.records."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            repository_root
            / "figures/qwen_lingoqa_sdc_cosine_by_layer_pair.png"
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
    parser.add_argument(
        "--title",
        default="Average cosine similarity by layer pair",
        help="Figure title. Pass an empty string to omit it.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if args.last_k_steps is not None and args.last_k_steps <= 0:
        parser.error("--last-k-steps must be positive")
    return args


def classify_record(record: Mapping[str, object]) -> str | None:
    if not bool(record.get("injected")):
        return None

    is_sdc = record.get("is_sdc")
    significance = record.get("significance")
    if is_sdc in (0, False):
        return "non_sdc"
    if is_sdc not in (1, True):
        return "invalid"
    if significance == 2:
        return "significant_sdc"
    if significance in (0, 1):
        return "non_significant_sdc"
    return "invalid"


def collect_sample_pair_means(
    records: Iterable[Mapping[str, object]],
    *,
    last_k_steps: int | None,
    adjacent_only: bool,
) -> tuple[dict[LayerPair, tuple[float, int]], int]:
    valid_records: list[Mapping[str, object]] = []
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
        valid_records.append(record)

    if last_k_steps is not None and valid_records:
        steps = sorted({int(record["step"]) for record in valid_records})
        selected_steps = set(steps[-last_k_steps:])
        valid_records = [
            record
            for record in valid_records
            if int(record["step"]) in selected_steps
        ]

    sums: dict[LayerPair, float] = defaultdict(float)
    counts: dict[LayerPair, int] = defaultdict(int)
    non_finite = 0
    for record in valid_records:
        pair = (int(record["src_layer"]), int(record["tgt_layer"]))
        try:
            value = float(record["cos_sim"])
        except (KeyError, TypeError, ValueError, OverflowError):
            non_finite += 1
            continue
        if not math.isfinite(value):
            non_finite += 1
            continue
        sums[pair] += value
        counts[pair] += 1

    means = {
        pair: (sums[pair] / count, count)
        for pair, count in counts.items()
        if count > 0
    }
    return means, non_finite


def collect(
    path: Path,
    *,
    last_k_steps: int | None,
    adjacent_only: bool,
) -> tuple[dict[str, dict[LayerPair, PairAccumulator]], CollectionStats]:
    accumulators = {
        group: defaultdict(PairAccumulator)
        for group in GROUP_ORDER
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

            pair_means, non_finite = collect_sample_pair_means(
                records,
                last_k_steps=last_k_steps,
                adjacent_only=adjacent_only,
            )
            stats.non_finite_values_skipped += non_finite
            if not pair_means:
                stats.missing_telemetry_skipped += 1
                continue

            for pair, (sample_mean, step_count) in pair_means.items():
                accumulators[group][pair].add(sample_mean, step_count)
            stats.records_used += 1

    return accumulators, stats


def write_summary(
    path: Path,
    accumulators: Mapping[str, Mapping[LayerPair, PairAccumulator]],
) -> None:
    pairs = sorted(
        {
            pair
            for group_accumulators in accumulators.values()
            for pair in group_accumulators
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "src_layer",
                "tgt_layer",
                "sdc_group",
                "mean_cos_sim",
                "sample_count",
                "finite_step_count",
            ]
        )
        for pair in pairs:
            for group in GROUP_ORDER:
                accumulator = accumulators[group].get(pair, PairAccumulator())
                writer.writerow(
                    [
                        pair[0],
                        pair[1],
                        group,
                        accumulator.mean,
                        accumulator.samples,
                        accumulator.steps,
                    ]
                )


def configure_style() -> None:
    for font_path in font_manager.findSystemFonts():
        if Path(font_path).name == "NimbusRoman-Regular.otf":
            font_manager.fontManager.addfont(font_path)
            break
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Nimbus Roman",
                "Times New Roman",
                "Times",
                "DejaVu Serif",
            ],
            "font.size": 9,
            "font.weight": "normal",
            "axes.linewidth": 0.6,
            "axes.labelweight": "normal",
            "lines.linewidth": 1.0,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "legend.frameon": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot(
    accumulators: Mapping[str, Mapping[LayerPair, PairAccumulator]],
    *,
    output: Path,
    title: str,
    dpi: int,
) -> None:
    pairs = sorted(
        {
            pair
            for group_accumulators in accumulators.values()
            for pair in group_accumulators
        }
    )
    if not pairs:
        raise ValueError("No finite layer-pair cosine similarities were found")

    configure_style()
    figure, axis = plt.subplots(figsize=(7.1, 2.75))
    positions = list(range(len(pairs)))
    for group in GROUP_ORDER:
        values = [
            accumulators[group].get(pair, PairAccumulator()).mean
            for pair in pairs
        ]
        style = GROUP_STYLES[group]
        axis.plot(
            positions,
            values,
            label=GROUP_LABELS[group],
            color=style["color"],
            marker=style["marker"],
            markersize=3.0,
            markeredgewidth=0.5,
        )

    axis.set_xticks(positions)
    axis.set_xticklabels(
        [f"({src},{tgt})" for src, tgt in pairs],
        rotation=90,
    )
    axis.set_xlabel("Layer pair")
    axis.set_ylabel("Mean cosine similarity")
    if title:
        axis.set_title(title, fontsize=9.5)
    axis.grid(axis="both", linestyle="--", linewidth=0.4, alpha=0.35)
    axis.legend(loc="best", fontsize=8)
    axis.margins(x=0.01)
    figure.tight_layout(pad=0.4)

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
    plot(
        accumulators,
        output=output,
        title=args.title,
        dpi=args.dpi,
    )

    group_counts = {
        group: max(
            (accumulator.samples for accumulator in pairs.values()),
            default=0,
        )
        for group, pairs in accumulators.items()
    }
    print(f"Input: {input_path}")
    print(f"Groups: {group_counts}")
    print(f"Collection stats: {stats}")
    print(f"Summary: {summary_output}")
    print(f"PNG: {output}")
    print(f"PDF: {output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
