#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


@dataclass(frozen=True)
class Job:
    model: str
    dataset: str
    train_path: Path
    valid_path: Path

    @property
    def name(self) -> str:
        return f"{self.model} / {self.dataset}"


@dataclass(frozen=True)
class Counts:
    all_sdc: int
    significant_sdc: int
    removed_all_feature_nan: int

    @property
    def percentage(self) -> float:
        return 100.0 * self.significant_sdc / self.all_sdc


def parse_args() -> argparse.Namespace:
    script_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Plot all observed SDC and significant SDC counts."
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=script_root,
        help="SIEVE repository root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_root / "figures/significant_sdc_share.png",
        help="Output PNG path.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def build_jobs(root: Path) -> list[Job]:
    return [
        Job(
            "Qwen2.5-VL-7B",
            "EarthVQA",
            root / "Qwen2.5-VL-7B/EarthVQA/train_data/Qwen2.5_EarthVQA_train_set.csv",
            root / "Qwen2.5-VL-7B/EarthVQA/train_data/Qwen2.5_EarthVQA_valid_set.csv",
        ),
        Job(
            "Qwen2.5-VL-7B",
            "LingoQA",
            root / "Qwen2.5-VL-7B/LingoQA/train_data/Qwen2.5_LingoQA_train_set.csv",
            root / "Qwen2.5-VL-7B/LingoQA/train_data/Qwen2.5_LingoQA_valid_set.csv",
        ),
        Job(
            "Qwen2.5-VL-7B",
            "VQAv2",
            root / "Qwen2.5-VL-7B/VQAv2/train_data/Qwen2.5_VQAv2_train_set.csv",
            root / "Qwen2.5-VL-7B/VQAv2/train_data/Qwen2.5_VQAv2_valid_set.csv",
        ),
        Job(
            "InternVL3-8B",
            "EarthVQA",
            root / "InternVL3-8B/EarthVQA/train_data/InternVL3_EarthVQA_train_set.csv",
            root / "InternVL3-8B/EarthVQA/train_data/InternVL3_EarthVQA_valid_set.csv",
        ),
        Job(
            "InternVL3-8B",
            "LingoQA",
            root / "InternVL3-8B/LingoQA/train_data/InternVL3_LingoQA_train_set.csv",
            root / "InternVL3-8B/LingoQA/train_data/InternVL3_LingoQA_valid_set.csv",
        ),
        Job(
            "InternVL3-8B",
            "VQAv2",
            root / "InternVL3-8B/VQAv2/train_data/InternVL3_VQAv2_train_set.csv",
            root / "InternVL3-8B/VQAv2/train_data/InternVL3_VQAv2_valid_set.csv",
        ),
        Job(
            "LLaVA-1.5-7B",
            "EarthVQA",
            root / "llava-v1.5-7B/EarthVQA/train_data/llava-v1.5-7B_train_set.csv",
            root / "llava-v1.5-7B/EarthVQA/train_data/llava-v1.5-7B_valid_set.csv",
        ),
        Job(
            "LLaVA-1.5-7B",
            "LingoQA",
            root / "llava-v1.5-7B/LingoQA/train_data/llava-v1.5-7B_train_set.csv",
            root / "llava-v1.5-7B/LingoQA/train_data/llava-v1.5-7B_valid_set.csv",
        ),
        Job(
            "LLaVA-1.5-7B",
            "VQAv2",
            root / "llava-v1.5-7B/VQAv2/train_data/llava-v1.5-7B_train_set.csv",
            root / "llava-v1.5-7B/VQAv2/train_data/llava-v1.5-7B_valid_set.csv",
        ),
    ]


METADATA_COLUMNS = {
    "orig_id",
    "sample_uid",
    "total_steps",
    "last_k_steps",
    "num_steps_used",
    "significance",
    "label",
    "significant_sdc_target",
}


def count_sdc(paths: tuple[Path, Path]) -> Counts:
    all_sdc = 0
    significant_sdc = 0
    removed_all_feature_nan = 0
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            feature_columns = [
                column
                for column in (reader.fieldnames or [])
                if column not in METADATA_COLUMNS
            ]
            if not feature_columns:
                raise ValueError(f"No feature columns found: {path}")
            for row in reader:
                if _all_feature_nan(row, feature_columns):
                    removed_all_feature_nan += 1
                    continue
                label = _parse_int(row.get("label"))
                target = _parse_int(row.get("significant_sdc_target"))
                if label not in (1, 2):
                    continue
                all_sdc += 1
                significant_sdc += int(target == 1)
    return Counts(all_sdc, significant_sdc, removed_all_feature_nan)


def _all_feature_nan(row: dict[str, str], feature_columns: list[str]) -> bool:
    return all(_is_nan_feature(row.get(column)) for column in feature_columns)


def _is_nan_feature(value: str | None) -> bool:
    if value is None or not value.strip():
        return True
    try:
        number = float(value)
    except ValueError:
        return True
    return (not math.isfinite(number)) or abs(number) > 3.4028235e38


def _parse_int(value: str | None) -> int | None:
    if value is None or not value.strip():
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def collect_counts(jobs: list[Job]) -> dict[str, Counts | None]:
    results: dict[str, Counts | None] = {}
    for job in jobs:
        if not job.train_path.is_file() or not job.valid_path.is_file():
            results[job.name] = None
            continue
        results[job.name] = count_sdc((job.train_path, job.valid_path))
    return results


def plot(jobs: list[Job], counts: dict[str, Counts | None], output: Path, dpi: int) -> None:
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
            "font.size": 8.5,
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
    models = list(dict.fromkeys(job.model for job in jobs))
    datasets = list(dict.fromkeys(job.dataset for job in jobs))
    colors = {
        "EarthVQA": "#4C72B0",
        "LingoQA": "#55A868",
        "VQAv2": "#C44E52",
    }
    positions = list(range(len(models)))
    width = 0.21

    figure, axis = plt.subplots(figsize=(3.35, 2.35))
    maximum = 0.0
    for dataset_index, dataset in enumerate(datasets):
        offsets = [
            position + (dataset_index - (len(datasets) - 1) / 2) * width
            for position in positions
        ]
        share_values = []
        items: list[Counts | None] = []
        for model in models:
            item = counts.get(f"{model} / {dataset}")
            items.append(item)
            share = item.percentage if item is not None else 0.0
            share_values.append(share)
            maximum = max(maximum, share)
        bars = axis.bar(
            offsets,
            share_values,
            width=width,
            color=colors.get(dataset, "#8172B2"),
            edgecolor="black",
            linewidth=0.7,
            label=dataset,
        )
        for bar, item in zip(bars, items):
            if item is None:
                bar.set_facecolor("#dee2e6")
                continue
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.2,
                f"{item.percentage:.0f}%",
                ha="center",
                va="bottom",
                fontsize=7.0,
            )

    axis.set_ylabel("Significant SDC (%)", labelpad=1)
    axis.set_xticks(positions)
    axis.set_xticklabels(models, rotation=0, ha="center")
    y_max = min(100, max(20, math.ceil(maximum * 1.18 / 20) * 20))
    axis.set_ylim(0, y_max)
    axis.set_yticks(range(0, int(y_max) + 1, 20))
    axis.yaxis.set_major_formatter(
        lambda value, _: f"{value:g}%"
    )
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.7)
    axis.tick_params(
        axis="both",
        which="major",
        direction="in",
        length=3,
        width=0.7,
        top=True,
        right=True,
        labelsize=7.5,
    )

    legend = axis.legend(
        loc="upper left",
        frameon=True,
        fancybox=False,
        edgecolor="black",
        framealpha=1.0,
        fontsize=7.5,
        handlelength=1.2,
        columnspacing=0.9,
        borderpad=0.25,
        labelspacing=0.15,
        handletextpad=0.35,
        borderaxespad=0,
    )
    legend.get_frame().set_linewidth(0.7)
    figure.tight_layout(pad=0.35)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi)
    if output.suffix.lower() != ".pdf":
        figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def print_counts(jobs: list[Job], counts: dict[str, Counts | None]) -> None:
    print(f"{'Job':<31} {'Share':>9}")
    for job in jobs:
        item = counts[job.name]
        if item is None:
            print(f"{job.name:<31} {'pending':>10}")
            continue
        print(f"{job.name:<31} {item.percentage:>8.1f}%")


def main() -> int:
    args = parse_args()
    root = args.repository_root.resolve()
    jobs = build_jobs(root)
    counts = collect_counts(jobs)
    if not any(item is not None for item in counts.values()):
        raise FileNotFoundError("No completed feature CSV files were found")
    print_counts(jobs, counts)
    output = args.output.resolve()
    plot(jobs, counts, output, args.dpi)
    print(f"\nSaved figure: {output}")
    if output.suffix.lower() != ".pdf":
        print(f"Saved vector figure: {output.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
