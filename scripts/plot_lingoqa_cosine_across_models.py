#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_sdc_cosine_by_layer_pair import (
    GROUP_LABELS,
    GROUP_ORDER,
    GROUP_STYLES,
    CollectionStats,
    PairAccumulator,
    collect,
    configure_style,
)


LayerPair = tuple[int, int]

PAPER_GROUP_ORDER = ("clean", "sdc", "significant_sdc")
PLOT_LABELS = {
    **GROUP_LABELS,
    "clean": "Clean",
    "sdc": "SDC",
}
PLOT_STYLES = {
    **GROUP_STYLES,
    "clean": GROUP_STYLES["non_sdc"],
    "sdc": GROUP_STYLES["non_significant_sdc"],
}


@dataclass(frozen=True)
class ModelJob:
    name: str
    panel_title: str
    labels_path: Path


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Compare layer-pair cosine similarity for three VLMs on one "
            "dataset."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=("LingoQA", "EarthVQA", "VQAv2"),
        default="LingoQA",
        help="Dataset to plot.",
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=repository_root,
        help="SIEVE repository root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output PNG path. By default, derive it from --dataset. "
            "A PDF with the same stem is also written."
        ),
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
        "--exclude-significant",
        action="store_true",
        help=(
            "Plot only non-SDC and non-significant SDC, and shade their gap."
        ),
    )
    parser.add_argument(
        "--clean-sdc-significant",
        action="store_true",
        help=(
            "Plot Clean (the original non-SDC group), all SDC, and "
            "Significant SDC. The last group is a subset of SDC."
        ),
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if args.last_k_steps is not None and args.last_k_steps <= 0:
        parser.error("--last-k-steps must be positive")
    if args.exclude_significant and args.clean_sdc_significant:
        parser.error(
            "--exclude-significant and --clean-sdc-significant are "
            "mutually exclusive"
        )
    return args


def build_jobs(root: Path, dataset: str) -> list[ModelJob]:
    return [
        ModelJob(
            name="Qwen2.5-VL-7B",
            panel_title="(a) Qwen2.5-VL-7B",
            labels_path=(
                root / f"Qwen2.5-VL-7B/{dataset}/json/labels.jsonl"
            ),
        ),
        ModelJob(
            name="InternVL3-8B",
            panel_title="(b) InternVL3-8B",
            labels_path=root / f"InternVL3-8B/{dataset}/json/labels.jsonl",
        ),
        ModelJob(
            name="LLaVA-1.5-7B",
            panel_title="(c) LLaVA-1.5-7B",
            labels_path=(
                root / f"llava-v1.5-7B/{dataset}/json/labels.jsonl"
            ),
        ),
    ]


def collect_all(
    jobs: list[ModelJob],
    *,
    last_k_steps: int | None,
) -> tuple[
    dict[str, dict[str, dict[LayerPair, PairAccumulator]]],
    dict[str, CollectionStats],
]:
    results: dict[str, dict[str, dict[LayerPair, PairAccumulator]]] = {}
    collection_stats: dict[str, CollectionStats] = {}
    for job in jobs:
        if not job.labels_path.is_file():
            raise FileNotFoundError(f"Missing labels file: {job.labels_path}")
        accumulators, stats = collect(
            job.labels_path,
            last_k_steps=last_k_steps,
            adjacent_only=True,
        )
        results[job.name] = accumulators
        collection_stats[job.name] = stats
    return results, collection_stats


def merge_accumulators(
    first: PairAccumulator,
    second: PairAccumulator,
) -> PairAccumulator:
    return PairAccumulator(
        total=first.total + second.total,
        samples=first.samples + second.samples,
        steps=first.steps + second.steps,
    )


def build_clean_sdc_results(
    results: Mapping[
        str,
        Mapping[str, Mapping[LayerPair, PairAccumulator]],
    ],
) -> dict[str, dict[str, dict[LayerPair, PairAccumulator]]]:
    regrouped: dict[
        str,
        dict[str, dict[LayerPair, PairAccumulator]],
    ] = {}
    for model, model_accumulators in results.items():
        non_significant = model_accumulators["non_significant_sdc"]
        significant = model_accumulators["significant_sdc"]
        sdc_pairs = set(non_significant) | set(significant)
        regrouped[model] = {
            "clean": dict(model_accumulators["non_sdc"]),
            "sdc": {
                pair: merge_accumulators(
                    non_significant.get(pair, PairAccumulator()),
                    significant.get(pair, PairAccumulator()),
                )
                for pair in sdc_pairs
            },
            "significant_sdc": dict(significant),
        }
    return regrouped


def all_pairs(
    results: Mapping[
        str,
        Mapping[str, Mapping[LayerPair, PairAccumulator]],
    ],
) -> list[LayerPair]:
    return sorted(
        {
            pair
            for model_accumulators in results.values()
            for group_accumulators in model_accumulators.values()
            for pair in group_accumulators
        }
    )


def model_pairs(
    model_accumulators: Mapping[
        str,
        Mapping[LayerPair, PairAccumulator],
    ],
) -> list[LayerPair]:
    return sorted(
        {
            pair
            for group_accumulators in model_accumulators.values()
            for pair in group_accumulators
        }
    )


def write_summary(
    path: Path,
    jobs: list[ModelJob],
    dataset: str,
    groups: tuple[str, ...],
    results: Mapping[
        str,
        Mapping[str, Mapping[LayerPair, PairAccumulator]],
    ],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "model",
                "dataset",
                "src_layer",
                "tgt_layer",
                "sdc_group",
                "mean_cos_sim",
                "sample_count",
                "finite_step_count",
            ]
        )
        for job in jobs:
            pairs = model_pairs(results[job.name])
            for pair in pairs:
                for group in groups:
                    accumulator = results[job.name][group].get(
                        pair,
                        PairAccumulator(),
                    )
                    writer.writerow(
                        [
                            job.name,
                            dataset,
                            pair[0],
                            pair[1],
                            group,
                            accumulator.mean,
                            accumulator.samples,
                            accumulator.steps,
                        ]
                    )


def plot(
    jobs: list[ModelJob],
    results: Mapping[
        str,
        Mapping[str, Mapping[LayerPair, PairAccumulator]],
    ],
    *,
    groups: tuple[str, ...],
    output: Path,
    dpi: int,
) -> None:
    if not all_pairs(results):
        raise ValueError("No finite layer-pair cosine similarities were found")

    configure_style()
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7.5,
            "lines.linewidth": 0.8,
        }
    )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(5.5, 2.25),
        sharey=True,
    )
    for axis, job in zip(axes, jobs):
        pairs = model_pairs(results[job.name])
        positions = list(range(len(pairs)))
        group_values: dict[str, list[float]] = {}
        for group in groups:
            values = [
                results[job.name][group]
                .get(pair, PairAccumulator())
                .mean
                for pair in pairs
            ]
            group_values[group] = values
            style = PLOT_STYLES[group]
            axis.plot(
                positions,
                values,
                label=PLOT_LABELS[group],
                color=style["color"],
                marker=style["marker"],
                markersize=1.9,
                markeredgewidth=0.3,
            )
        if groups == ("non_sdc", "non_significant_sdc"):
            axis.fill_between(
                positions,
                group_values["non_sdc"],
                group_values["non_significant_sdc"],
                color=PLOT_STYLES["non_sdc"]["color"],
                alpha=0.10,
                linewidth=0,
            )
        axis.set_title(job.panel_title, pad=2)
        axis.grid(
            axis="both",
            linestyle="--",
            linewidth=0.3,
            alpha=0.3,
        )
        axis.margins(x=0.01)
        axis.set_xlabel("Layer pair", labelpad=2)
        tick_positions = list(range(0, len(pairs), 4))
        if tick_positions[-1] != len(pairs) - 1:
            tick_positions.append(len(pairs) - 1)
        tick_labels = [
            f"({pairs[index][0]},{pairs[index][1]})"
            for index in tick_positions
        ]
        axis.set_xticks(tick_positions)
        axis.set_xticklabels(tick_labels, rotation=90)

    axes[0].set_ylabel("Mean cosine similarity", labelpad=2)

    finite_values = [
        accumulator.mean
        for model_accumulators in results.values()
        for group, group_accumulators in model_accumulators.items()
        if group in groups
        for accumulator in group_accumulators.values()
        if math.isfinite(accumulator.mean)
    ]
    lower = min(finite_values)
    upper = max(finite_values)
    padding = max(0.02, 0.04 * (upper - lower))
    axes[0].set_ylim(max(-1.0, lower - padding), min(1.0, upper + padding))

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(groups),
        frameon=True,
        handlelength=2.0,
        columnspacing=1.0,
    )
    figure.subplots_adjust(
        left=0.105,
        right=0.99,
        bottom=0.25,
        top=0.82,
        wspace=0.10,
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
    root = args.repository_root.resolve()
    if args.clean_sdc_significant:
        groups = PAPER_GROUP_ORDER
        output_stem = (
            f"{args.dataset.lower()}_cosine_clean_sdc_significant_"
            "three_models.png"
        )
    elif args.exclude_significant:
        groups = ("non_sdc", "non_significant_sdc")
        output_stem = (
            f"{args.dataset.lower()}_cosine_non_sdc_vs_non_significant_"
            "three_models.png"
        )
    else:
        groups = GROUP_ORDER
        output_stem = (
            f"{args.dataset.lower()}_cosine_by_layer_pair_"
            "three_models.png"
        )
    output = (
        args.output.resolve()
        if args.output is not None
        else root / "figures" / output_stem
    )
    summary_output = (
        args.summary_output.resolve()
        if args.summary_output is not None
        else output.with_suffix(".csv")
    )
    jobs = build_jobs(root, args.dataset)
    results, collection_stats = collect_all(
        jobs,
        last_k_steps=args.last_k_steps,
    )
    if args.clean_sdc_significant:
        results = build_clean_sdc_results(results)
    write_summary(summary_output, jobs, args.dataset, groups, results)
    plot(jobs, results, groups=groups, output=output, dpi=args.dpi)

    for job in jobs:
        group_counts = {
            group: max(
                (
                    accumulator.samples
                    for accumulator in results[job.name][group].values()
                ),
                default=0,
            )
            for group in groups
        }
        print(f"{job.name}: groups={group_counts}")
        print(f"{job.name}: stats={collection_stats[job.name]}")
    print(f"Summary: {summary_output}")
    print(f"PNG: {output}")
    print(f"PDF: {output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
