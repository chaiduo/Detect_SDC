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
class QuadrantStats:
    counts: Counter[tuple[bool, bool]]
    total: int
    skipped: int
    non_finite: int


def parse_args() -> argparse.Namespace:
    script_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Plot Qwen2.5-VL-7B/LingoQA quadrants for activation-value "
            "deviation and significant SDC."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=script_root / "Qwen2.5-VL-7B/LingoQA/json/labels.jsonl",
        help="Qwen LingoQA labeled JSONL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_root / "figures/qwen_lingoqa_fault_quadrants.png",
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


def collect_stats(path: Path, threshold: float) -> QuadrantStats:
    counts: Counter[tuple[bool, bool]] = Counter()
    total = 0
    skipped = 0
    non_finite = 0
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
            before = _parse_float(fault.get("before"), allow_non_finite=True)
            after = _parse_float(fault.get("after"), allow_non_finite=True)
            if before is None or after is None:
                skipped += 1
                continue
            if not math.isfinite(before) or not math.isfinite(after):
                large_deviation = True
                non_finite += 1
            else:
                large_deviation = abs(after - before) > threshold
            significant_sdc = record.get("is_sdc") == 1 and record.get("significance") == 2
            counts[(large_deviation, significant_sdc)] += 1
            total += 1
    return QuadrantStats(
        counts=counts,
        total=total,
        skipped=skipped,
        non_finite=non_finite,
    )


def _parse_float(value: object, *, allow_non_finite: bool = False) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if allow_non_finite or math.isfinite(number):
        return number
    return None


def plot(stats: QuadrantStats, output: Path, threshold: float, dpi: int) -> None:
    if stats.total <= 0:
        raise ValueError("No valid fault records with numeric before/after were found")

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
            "font.size": 7,
            "font.weight": "normal",
            "axes.edgecolor": "black",
            "axes.linewidth": 0.7,
            "axes.labelweight": "normal",
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    colors = {
        (False, False): "#e8eef7",
        (True, False): "#cfe1f2",
        (False, True): "#f4d1d3",
        (True, True): "#c44e52",
    }
    labels = {
        (False, False): "Small deviation\nNon-significant",
        (True, False): "Large deviation\nNon-significant",
        (False, True): "Small deviation\nSignificant",
        (True, True): "Large deviation\nSignificant",
    }

    figure, axis = plt.subplots(figsize=(3.35, 2.45))
    for large_deviation in (False, True):
        for significant_sdc in (False, True):
            x = 1 if large_deviation else 0
            y = 1 if significant_sdc else 0
            count = stats.counts[(large_deviation, significant_sdc)]
            percentage = 100.0 * count / stats.total
            axis.add_patch(
                Rectangle(
                    (x, y),
                    1,
                    1,
                    facecolor=colors[(large_deviation, significant_sdc)],
                    edgecolor="black",
                    linewidth=0.7,
                )
            )
            text_color = "white" if (large_deviation, significant_sdc) == (True, True) else "black"
            axis.text(
                x + 0.5,
                y + 0.57,
                f"{percentage:.1f}%",
                ha="center",
                va="center",
                fontsize=9,
                color=text_color,
            )
            axis.text(
                x + 0.5,
                y + 0.35,
                labels[(large_deviation, significant_sdc)],
                ha="center",
                va="center",
                fontsize=5.8,
                color=text_color,
                linespacing=1.05,
            )

    axis.set_xlim(0, 2)
    axis.set_ylim(0, 2)
    axis.set_aspect("equal")
    axis.set_xticks([0.5, 1.5])
    axis.set_xticklabels(
        [
            f"|after-before| <= {threshold:g}",
            f"|after-before| > {threshold:g}\n/ non-finite",
        ]
    )
    axis.set_yticks([0.5, 1.5])
    axis.set_yticklabels(["Not significant SDC", "Significant SDC"])
    axis.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=0,
        width=0.7,
        labelsize=5.8,
    )
    axis.axvline(1, color="black", linewidth=0.7)
    axis.axhline(1, color="black", linewidth=0.7)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color("black")

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(pad=0.25)
    figure.savefig(output, dpi=dpi)
    if output.suffix.lower() != ".pdf":
        figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def print_stats(stats: QuadrantStats, threshold: float) -> None:
    print(f"Valid fault records: {stats.total:,}")
    if stats.non_finite:
        print(f"Non-finite before/after counted as large deviation: {stats.non_finite:,}")
    if stats.skipped:
        print(f"Skipped non-numeric fault records: {stats.skipped:,}")
    print(f"{'|after-before| > threshold':<28} {'Significant SDC':<16} {'Count':>10} {'Share':>9}")
    for large_deviation in (False, True):
        for significant_sdc in (False, True):
            count = stats.counts[(large_deviation, significant_sdc)]
            share = 100.0 * count / stats.total if stats.total else 0.0
            print(
                f"{str(large_deviation):<28} {str(significant_sdc):<16} "
                f"{count:>10,} {share:>8.2f}%"
            )
    print(f"Threshold: {threshold:g}")


def main() -> int:
    args = parse_args()
    stats = collect_stats(args.input.resolve(), args.threshold)
    print_stats(stats, args.threshold)
    output = args.output.resolve()
    plot(stats, output, args.threshold, args.dpi)
    print(f"\nSaved figure: {output}")
    if output.suffix.lower() != ".pdf":
        print(f"Saved vector figure: {output.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
