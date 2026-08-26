#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


K_VALUES = (1, 2, 4, 8, 12, 16, 24, 32, 50)
MODELS = (
    ("Qwen2.5-VL-7B", "Qwen2.5-VL-7B", "#4C78A8", "o"),
    ("InternVL3-8B", "InternVL3-8B", "#F58518", "s"),
    ("LLaVA-1.5-7B", "llava-v1.5-7B", "#54A24B", "^"),
)
DATASETS = ("EarthVQA", "LingoQA", "VQAv2")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Plot prefix-step online detection ablation results."
    )
    parser.add_argument("--repository-root", type=Path, default=root)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=root / "figures/online_step_ablation",
    )
    parser.add_argument(
        "--selected-k",
        type=int,
        choices=K_VALUES,
        default=2,
        help="Prefix window selected for online deployment.",
    )
    return parser.parse_args()


def load_results(root: Path) -> pd.DataFrame:
    frames = []
    for model, directory, _, _ in MODELS:
        for dataset in DATASETS:
            path = (
                root
                / directory
                / "online_step_ablation_20260814"
                / dataset
                / "summary.csv"
            )
            frame = pd.read_csv(path)
            if set(frame["k"].astype(int)) != set(K_VALUES):
                raise ValueError(f"Incomplete K sweep: {path}")
            frame["model"] = model
            frame["dataset"] = dataset
            frames.append(frame)
    detailed = pd.concat(frames, ignore_index=True)
    if len(detailed) != len(MODELS) * len(DATASETS) * len(K_VALUES):
        raise ValueError("Unexpected number of online ablation rows")
    return detailed


def aggregate(detailed: pd.DataFrame) -> pd.DataFrame:
    model_rows = (
        detailed.groupby(["model", "k"], as_index=False)
        .agg(
            macro_f1=("f1", "mean"),
            macro_precision=("precision", "mean"),
            macro_recall=("recall", "mean"),
            mean_steps_used=("mean_steps_used", "mean"),
        )
    )
    overall = (
        detailed.groupby("k", as_index=False)
        .agg(
            overall_f1=("f1", "mean"),
            overall_precision=("precision", "mean"),
            overall_recall=("recall", "mean"),
            overall_mean_steps_used=("mean_steps_used", "mean"),
        )
    )
    wide = model_rows.pivot(index="k", columns="model", values="macro_f1")
    wide.columns = [f"{column}_f1" for column in wide.columns]
    result = overall.merge(wide.reset_index(), on="k", validate="one_to_one")
    baseline = result.loc[result["k"] == 50, "overall_f1"].iloc[0]
    result["delta_from_k50_pp"] = 100.0 * (
        result["overall_f1"] - baseline
    )
    return result.sort_values("k").reset_index(drop=True)


def plot(
    aggregated: pd.DataFrame,
    selected_k: int,
    output_prefix: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 7.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    positions = list(range(len(K_VALUES)))
    figure, axis = plt.subplots(figsize=(3.35, 2.25))
    for model, _, color, marker in MODELS:
        values = (
            aggregated.set_index("k")
            .loc[list(K_VALUES), f"{model}_f1"]
            .to_numpy()
            * 100.0
        )
        axis.plot(
            positions,
            values,
            color=color,
            marker=marker,
            markersize=3.0,
            markeredgewidth=0.5,
            label=model,
        )

    selected_position = list(K_VALUES).index(selected_k)
    axis.axvline(
        selected_position,
        color="#6B7280",
        linestyle="--",
        linewidth=0.8,
        zorder=0,
    )
    axis.text(
        selected_position + 0.08,
        0.02,
        rf"$K^*={selected_k}$",
        transform=axis.get_xaxis_transform(),
        color="#4B5563",
        fontsize=7,
        va="bottom",
    )
    axis.set_xticks(positions, [str(k) for k in K_VALUES])
    axis.set_xlabel(r"Number of prefix decoding steps ($K$)", labelpad=2)
    axis.set_ylabel("Significant-SDC F1 (%)", labelpad=2)
    axis.grid(axis="y", color="#D1D5DB", linewidth=0.45, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(
        loc="center",
        bbox_to_anchor=(0.62, 0.72),
        ncol=1,
        frameon=True,
        framealpha=1.0,
        edgecolor="#D1D5DB",
        borderpad=0.35,
        labelspacing=0.3,
        handlelength=1.5,
    )
    all_values = aggregated[
        [f"{model}_f1" for model, _, _, _ in MODELS]
    ].to_numpy() * 100.0
    lower = max(0.0, float(all_values.min()) - 1.5)
    upper = min(100.0, float(all_values.max()) + 1.0)
    axis.set_ylim(lower, upper)
    figure.tight_layout(pad=0.25)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_prefix.with_suffix(".png"),
        dpi=400,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    figure.savefig(
        output_prefix.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(figure)


def main() -> int:
    args = parse_args()
    root = args.repository_root.resolve()
    output_prefix = args.output_prefix.resolve()
    detailed = load_results(root)
    aggregated = aggregate(detailed)
    selected_k = args.selected_k

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    detailed.to_csv(
        output_prefix.with_name(
            f"{output_prefix.name}_detailed.csv"
        ),
        index=False,
    )
    aggregated.to_csv(
        output_prefix.with_name(
            f"{output_prefix.name}_aggregate.csv"
        ),
        index=False,
    )
    plot(aggregated, selected_k, output_prefix)
    print(f"selected_k={selected_k}")
    print(aggregated.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
