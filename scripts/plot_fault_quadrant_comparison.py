#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle


@dataclass(frozen=True)
class Job:
    model: str
    dataset: str
    labels_path: Path

    @property
    def key(self) -> tuple[str, str]:
        return (self.model, self.dataset)


@dataclass(frozen=True)
class QuadrantStats:
    counts: Counter[tuple[bool, bool]]
    total: int
    non_finite: int
    skipped: int

    def percentage(self, large_deviation: bool, significant_sdc: bool) -> float:
        if self.total <= 0:
            return 0.0
        return 100.0 * self.counts[(large_deviation, significant_sdc)] / self.total


def parse_args() -> argparse.Namespace:
    script_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Plot non-NaN fault-value deviation and significant-SDC quadrant "
            "shares across models and datasets."
        )
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=script_root,
        help="Detect_SDC repository root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_root / "figures/fault_quadrant_comparison_non_nan.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Absolute fault value deviation threshold.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def build_jobs(root: Path) -> list[Job]:
    return [
        Job(
            "Qwen2.5-VL-7B",
            "EarthVQA",
            root / "Qwen2.5-VL-7B/EarthVQA/json/labels.jsonl",
        ),
        Job(
            "Qwen2.5-VL-7B",
            "LingoQA",
            root / "Qwen2.5-VL-7B/LingoQA/json/labels.jsonl",
        ),
        Job(
            "Qwen2.5-VL-7B",
            "VQAv2",
            root / "Qwen2.5-VL-7B/VQAv2/json/labels.jsonl",
        ),
        Job(
            "InternVL3-8B",
            "EarthVQA",
            root / "InternVL3-8B/EarthVQA/json/labels.jsonl",
        ),
        Job(
            "InternVL3-8B",
            "LingoQA",
            root / "InternVL3-8B/LingoQA/json/labels.jsonl",
        ),
        Job(
            "InternVL3-8B",
            "VQAv2",
            root / "InternVL3-8B/VQAv2/json/labels.jsonl",
        ),
        Job(
            "LLaVA-1.5-7B",
            "EarthVQA",
            root / "llava-v1.5-7B/EarthVQA/json/labels.jsonl",
        ),
        Job(
            "LLaVA-1.5-7B",
            "LingoQA",
            root / "llava-v1.5-7B/LingoQA/json/labels.jsonl",
        ),
        Job(
            "LLaVA-1.5-7B",
            "VQAv2",
            root / "llava-v1.5-7B/VQAv2/json/labels.jsonl",
        ),
    ]


def collect_stats(path: Path, threshold: float) -> QuadrantStats:
    counts: Counter[tuple[bool, bool]] = Counter()
    total = 0
    non_finite = 0
    skipped = 0
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error

            fault = record.get("fault")
            if not isinstance(fault, dict):
                continue
            before = _parse_float(fault.get("before"))
            after = _parse_float(fault.get("after"))
            if before is None or after is None:
                skipped += 1
                continue
            if not math.isfinite(before) or not math.isfinite(after):
                non_finite += 1
                continue

            large_deviation = abs(after - before) > threshold
            significant_sdc = (
                record.get("is_sdc") == 1 and record.get("significance") == 2
            )
            counts[(large_deviation, significant_sdc)] += 1
            total += 1

    return QuadrantStats(
        counts=counts,
        total=total,
        non_finite=non_finite,
        skipped=skipped,
    )


def _parse_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def collect_all(jobs: list[Job], threshold: float) -> dict[tuple[str, str], QuadrantStats]:
    stats: dict[tuple[str, str], QuadrantStats] = {}
    for job in jobs:
        if not job.labels_path.is_file():
            raise FileNotFoundError(f"Missing labels file: {job.labels_path}")
        stats[job.key] = collect_stats(job.labels_path, threshold)
    return stats


def plot(
    jobs: list[Job],
    stats: dict[tuple[str, str], QuadrantStats],
    output: Path,
    threshold: float,
    dpi: int,
) -> None:
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
            "font.size": 10.5,
            "font.weight": "normal",
            "axes.edgecolor": "black",
            "axes.linewidth": 0.5,
            "axes.labelweight": "normal",
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    models = list(dict.fromkeys(job.model for job in jobs))
    datasets = list(dict.fromkeys(job.dataset for job in jobs))
    categories = [
        ((False, False), "Small deviation / non-significant SDC", "#d7e6f5"),
        ((False, True), "Small deviation / significant SDC", "#f4c7cc"),
        ((True, False), "Large deviation / non-significant SDC", "#7da6d6"),
        ((True, True), "Large deviation / significant SDC", "#c44e52"),
    ]

    model_labels = {
        "Qwen2.5-VL-7B": "Qwen2.5-VL-7B",
        "InternVL3-8B": "InternVL3-8B",
        "LLaVA-1.5-7B": "LLaVA-1.5-7B",
    }
    figure_width = 6.1
    plot_left = 0.10
    plot_right = 0.92
    plot_bottom = 0.09
    plot_top = 0.92
    subplot_spacing = 0.02
    # A square plotting region makes equal wspace/hspace physically equal.
    figure_height = (
        figure_width
        * (plot_right - plot_left)
        / (plot_top - plot_bottom)
    )
    figure, axes = plt.subplots(
        len(datasets),
        len(models),
        figsize=(figure_width, figure_height),
        sharex=True,
        sharey=True,
    )
    color_by_key = {key: color for key, _, color in categories}

    for row_index, dataset in enumerate(datasets):
        for column_index, model in enumerate(models):
            axis = axes[row_index][column_index]
            item = stats[(model, dataset)]
            for large_deviation in (False, True):
                for significant_sdc in (False, True):
                    key = (large_deviation, significant_sdc)
                    x = 1 if large_deviation else 0
                    y = 1 if significant_sdc else 0
                    value = item.percentage(large_deviation, significant_sdc)
                    axis.add_patch(
                        Rectangle(
                            (x, y),
                            1,
                            1,
                            facecolor=color_by_key[key],
                            edgecolor="black",
                            linewidth=0.4,
                        )
                    )
                    text_color = "white" if key == (True, True) else "black"
                    axis.text(
                        x + 0.5,
                        y + 0.5,
                        f"{value:.1f}%",
                        ha="center",
                        va="center",
                        fontsize=11.5,
                        color=text_color,
                    )

            axis.set_xlim(0, 2)
            axis.set_ylim(0, 2)
            axis.set_aspect("equal")
            axis.set_anchor("C")
            axis.set_xticks([0.5, 1.5])
            axis.set_yticks([0.5, 1.5])
            axis.tick_params(
                axis="both",
                which="major",
                direction="out",
                length=0,
                width=0.4,
                labelsize=11.0,
                pad=1.0,
            )
            if row_index == len(datasets) - 1:
                axis.set_xticklabels(
                    [
                        f"Deviation\n≤ {threshold:g}",
                        f"Deviation\n> {threshold:g}",
                    ]
                )
            else:
                axis.set_xticklabels([])
            if column_index == 0:
                axis.set_yticklabels(["Non-sig.", "Sig."])
                axis.set_ylabel(dataset, labelpad=5.0, fontsize=12.0)
            else:
                axis.set_yticklabels([])
            if row_index == 0:
                axis.set_title(model_labels.get(model, model), pad=4.0, fontsize=11.0)
            axis.axvline(1, color="black", linewidth=0.4)
            axis.axhline(1, color="black", linewidth=0.4)
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_color("black")
                spine.set_linewidth(0.4)

    handles = [
        Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="black", linewidth=0.4)
        for _, _, color in categories
    ]
    labels = [label for _, label, _ in categories]
    legend = figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        framealpha=1.0,
        fontsize=10.0,
        handlelength=0.85,
        columnspacing=0.5,
        borderpad=0.2,
        labelspacing=0.2,
        handletextpad=0.2,
        borderaxespad=0,
    )
    legend.get_frame().set_linewidth(0.45)
    figure.subplots_adjust(
        left=plot_left,
        right=plot_right,
        bottom=plot_bottom,
        top=plot_top,
        wspace=subplot_spacing,
        hspace=subplot_spacing,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    if output.suffix.lower() != ".pdf":
        figure.savefig(
            output.with_suffix(".pdf"),
            bbox_inches="tight",
            pad_inches=0.02,
        )
    plt.close(figure)


def print_stats(
    jobs: list[Job],
    stats: dict[tuple[str, str], QuadrantStats],
) -> None:
    header = (
        f"{'Model':<16} {'Dataset':<8} {'Total':>8} {'NaN/Inf':>8} "
        f"{'Small non-sig':>13} {'Small sig':>10} "
        f"{'Large non-sig':>13} {'Large sig':>10}"
    )
    print(header)
    for job in jobs:
        item = stats[job.key]
        print(
            f"{job.model:<16} {job.dataset:<8} {item.total:>8,} "
            f"{item.non_finite:>8,} "
            f"{item.percentage(False, False):>12.2f}% "
            f"{item.percentage(False, True):>9.2f}% "
            f"{item.percentage(True, False):>12.2f}% "
            f"{item.percentage(True, True):>9.2f}%"
        )


def main() -> int:
    args = parse_args()
    root = args.repository_root.resolve()
    jobs = build_jobs(root)
    stats = collect_all(jobs, args.threshold)
    print_stats(jobs, stats)
    output = args.output.resolve()
    plot(jobs, stats, output, args.threshold, args.dpi)
    print(f"\nSaved figure: {output}")
    if output.suffix.lower() != ".pdf":
        print(f"Saved vector figure: {output.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
