#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle


MODELS = ("Qwen2.5-VL-7B", "InternVL3-8B", "LLaVA-1.5-7B")
DATASETS = ("EarthVQA", "LingoQA", "VQAv2")
JOB_KEYS = {
    ("Qwen2.5-VL-7B", "EarthVQA"): "qwen25_vl_earthvqa",
    ("Qwen2.5-VL-7B", "LingoQA"): "qwen25_vl_lingoqa",
    ("Qwen2.5-VL-7B", "VQAv2"): "qwen25_vl_vqav2",
    ("InternVL3-8B", "EarthVQA"): "internvl3_earthvqa",
    ("InternVL3-8B", "LingoQA"): "internvl3_lingoqa",
    ("InternVL3-8B", "VQAv2"): "internvl3_vqav2",
    ("LLaVA-1.5-7B", "EarthVQA"): "llava15_earthvqa",
    ("LLaVA-1.5-7B", "LingoQA"): "llava15_lingoqa",
    ("LLaVA-1.5-7B", "VQAv2"): "llava15_vqav2",
}


@dataclass(frozen=True)
class Counts:
    injected: int
    significant: int
    finite: int
    small_non_significant: int
    small_significant: int
    large_non_significant: int
    large_significant: int
    non_finite: int

    @property
    def significant_share(self) -> float:
        return 100.0 * self.significant / self.injected

    def quadrant_share(self, large: bool, significant: bool) -> float:
        field = {
            (False, False): self.small_non_significant,
            (False, True): self.small_significant,
            (True, False): self.large_non_significant,
            (True, True): self.large_significant,
        }[(large, significant)]
        return 100.0 * field / self.finite


def read_counts(path: Path) -> Counts:
    values = {
        "injected": 0,
        "significant": 0,
        "finite": 0,
        "small_non_significant": 0,
        "small_significant": 0,
        "large_non_significant": 0,
        "large_significant": 0,
        "non_finite": 0,
    }
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            if not bool(record.get("injected")):
                continue
            significant = bool(
                int(
                    record.get(
                        "significant_sdc_target",
                        int(
                            record.get("is_sdc", 0) == 1
                            and record.get("significance") == 2
                        ),
                    )
                )
            )
            values["injected"] += 1
            values["significant"] += int(significant)
            fault = record.get("fault")
            if not isinstance(fault, dict):
                continue
            try:
                before = float(fault.get("before"))
                after = float(fault.get("after"))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(before) or not math.isfinite(after):
                values["non_finite"] += 1
                continue
            values["finite"] += 1
            large = abs(after - before) > 1.0
            key = (
                ("large" if large else "small")
                + "_"
                + ("significant" if significant else "non_significant")
            )
            values[key] += 1
    return Counts(**values)


def configure_plotting() -> None:
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
            "axes.edgecolor": "black",
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def plot_significant_share(
    counts: dict[tuple[str, str], Counts],
    output: Path,
) -> None:
    colors = {"EarthVQA": "#4C72B0", "LingoQA": "#55A868", "VQAv2": "#C44E52"}
    positions = list(range(len(MODELS)))
    width = 0.22
    figure, axis = plt.subplots(figsize=(6.6, 3.2))
    maximum = 0.0
    for dataset_index, dataset in enumerate(DATASETS):
        offsets = [
            value + (dataset_index - 1) * width for value in positions
        ]
        shares = [
            counts[(model, dataset)].significant_share for model in MODELS
        ]
        maximum = max(maximum, max(shares))
        bars = axis.bar(
            offsets,
            shares,
            width=width,
            color=colors[dataset],
            edgecolor="black",
            linewidth=0.6,
            label=dataset,
        )
        for bar, value in zip(bars, shares):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.12,
                f"{value:.2f}%",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )
    axis.set_ylabel("Significant SDC / injected runs (%)")
    axis.set_xticks(positions)
    axis.set_xticklabels(MODELS)
    axis.set_ylim(0, math.ceil((maximum + 0.8) / 2) * 2)
    axis.legend(frameon=False, ncol=3, loc="upper center")
    axis.grid(axis="y", color="#dddddd", linewidth=0.5)
    figure.tight_layout()
    figure.savefig(output, dpi=300)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def plot_quadrants(
    counts: dict[tuple[str, str], Counts],
    output: Path,
) -> None:
    categories = {
        (False, False): "#d7e6f5",
        (False, True): "#f4c7cc",
        (True, False): "#7da6d6",
        (True, True): "#c44e52",
    }
    figure, axes = plt.subplots(
        len(DATASETS),
        len(MODELS),
        figsize=(7.2, 6.6),
        sharex=True,
        sharey=True,
    )
    for row, dataset in enumerate(DATASETS):
        for column, model in enumerate(MODELS):
            axis = axes[row][column]
            item = counts[(model, dataset)]
            for large in (False, True):
                for significant in (False, True):
                    x = int(large)
                    y = int(significant)
                    value = item.quadrant_share(large, significant)
                    axis.add_patch(
                        Rectangle(
                            (x, y),
                            1,
                            1,
                            facecolor=categories[(large, significant)],
                            edgecolor="black",
                            linewidth=0.5,
                        )
                    )
                    axis.text(
                        x + 0.5,
                        y + 0.5,
                        f"{value:.1f}%",
                        ha="center",
                        va="center",
                        fontsize=10,
                        color="white" if (large and significant) else "black",
                    )
            axis.set_xlim(0, 2)
            axis.set_ylim(0, 2)
            axis.set_aspect("equal")
            axis.set_xticks([0.5, 1.5])
            axis.set_yticks([0.5, 1.5])
            axis.set_xticklabels(
                ["Deviation\n≤ 1", "Deviation\n> 1"]
                if row == len(DATASETS) - 1
                else []
            )
            axis.set_yticklabels(
                ["Non-sig.", "Sig."] if column == 0 else []
            )
            if row == 0:
                axis.set_title(model)
            if column == 0:
                axis.set_ylabel(dataset)
            axis.tick_params(length=0)
    legend_items = [
        ((False, False), "Small deviation / non-significant"),
        ((False, True), "Small deviation / significant"),
        ((True, False), "Large deviation / non-significant"),
        ((True, True), "Large deviation / significant"),
    ]
    handles = [
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor=categories[key],
            edgecolor="black",
            linewidth=0.5,
        )
        for key, _ in legend_items
    ]
    figure.legend(
        handles,
        [label for _, label in legend_items],
        loc="upper center",
        ncol=2,
        frameon=False,
    )
    figure.subplots_adjust(
        left=0.10,
        right=0.98,
        bottom=0.08,
        top=0.88,
        wspace=0.06,
        hspace=0.08,
    )
    figure.savefig(output, dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def write_csvs(
    counts: dict[tuple[str, str], Counts],
    output_dir: Path,
) -> None:
    fields = [
        "model",
        "dataset",
        "injected",
        "significant",
        "significant_share_percent",
        "finite",
        "small_non_significant",
        "small_significant",
        "large_non_significant",
        "large_significant",
        "non_finite",
    ]
    with (output_dir / "significant_sdc_share_v2.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for model in MODELS:
            for dataset in DATASETS:
                item = counts[(model, dataset)]
                writer.writerow(
                    {
                        "model": model,
                        "dataset": dataset,
                        **item.__dict__,
                        "significant_share_percent": item.significant_share,
                    }
                )


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "figures"
    counts = {
        key: read_counts(
            root
            / "artifacts"
            / "iclr_v2"
            / job
            / "json"
            / "labels.jsonl"
        )
        for key, job in JOB_KEYS.items()
    }
    configure_plotting()
    plot_significant_share(
        counts,
        output_dir / "significant_sdc_share_v2.png",
    )
    plot_quadrants(
        counts,
        output_dir / "fault_quadrant_comparison_v2.png",
    )
    write_csvs(counts, output_dir)

    finite_total = sum(item.finite for item in counts.values())
    small_significant = sum(
        item.small_significant for item in counts.values()
    )
    large_non_significant = sum(
        item.large_non_significant for item in counts.values()
    )
    print(
        json.dumps(
            {
                "significant_share_range_percent": [
                    min(item.significant_share for item in counts.values()),
                    max(item.significant_share for item in counts.values()),
                ],
                "significant_share_macro_percent": sum(
                    item.significant_share for item in counts.values()
                )
                / len(counts),
                "significant_share_pooled_percent": 100.0
                * sum(item.significant for item in counts.values())
                / sum(item.injected for item in counts.values()),
                "finite_small_significant_percent": 100.0
                * small_significant
                / finite_total,
                "finite_large_non_significant_percent": 100.0
                * large_non_significant
                / finite_total,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
