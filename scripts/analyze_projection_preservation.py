#!/usr/bin/env python3
"""Audit whether orthogonal projection preserves inter-layer relationships."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: I001


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from detect_sdc.adapters import (
    load_dataset_adapter,
    load_model_adapter,
)
from detect_sdc.profiler import Profiler


class ProjectionRelationCollector:
    """Accumulate relationship-preservation statistics without storing features."""

    def __init__(self, *, seed: int, bootstrap_replicates: int) -> None:
        self.seed = seed
        self.bootstrap_replicates = bootstrap_replicates
        self.layer_ids: tuple[int, ...] | None = None
        self.input_dim: int | None = None
        self.output_dim: int | None = None
        self.raw_rsm_sum: np.ndarray | None = None
        self.projected_rsm_sum: np.ndarray | None = None
        self.step_count = 0
        self.sample_metrics: list[dict[str, Any]] = []
        self._all_raw_pairs: list[np.ndarray] = []
        self._all_projected_pairs: list[np.ndarray] = []
        self._sample_id: int | None = None
        self._orig_id: str | None = None
        self._sample_raw_pairs: list[np.ndarray] = []
        self._sample_projected_pairs: list[np.ndarray] = []
        self._sample_relative_distance_errors: list[np.ndarray] = []

    def start_sample(self, sample_id: int, orig_id: str) -> None:
        if self._sample_id is not None:
            raise RuntimeError("Previous sample has not been finalized")
        self._sample_id = sample_id
        self._orig_id = orig_id
        self._sample_raw_pairs = []
        self._sample_projected_pairs = []
        self._sample_relative_distance_errors = []

    def observe(
        self,
        raw: torch.Tensor,
        projected: torch.Tensor,
        layer_ids: tuple[int, ...],
    ) -> None:
        if self._sample_id is None:
            raise RuntimeError("Projection observation arrived outside a sample")
        if raw.dim() != 2 or projected.dim() != 2:
            raise ValueError("Expected raw and projected features to be matrices")
        if raw.shape[0] != projected.shape[0]:
            raise ValueError("Raw and projected features have different layer counts")
        if raw.shape[0] != len(layer_ids):
            raise ValueError("Layer IDs do not match observed feature rows")

        if self.layer_ids is None:
            self._initialize_shapes(raw, projected, layer_ids)
        elif self.layer_ids != layer_ids:
            raise ValueError("Observed layer ordering changed during the run")

        raw = raw.detach().float()
        projected = projected.detach().float()
        if not torch.isfinite(raw).all() or not torch.isfinite(projected).all():
            raise ValueError("Clean projection audit observed NaN or Inf features")

        raw_normalized = F.normalize(raw, p=2, dim=1)
        projected_normalized = F.normalize(projected, p=2, dim=1)
        raw_rsm = raw_normalized @ raw_normalized.T
        projected_rsm = projected_normalized @ projected_normalized.T

        layer_count = raw.shape[0]
        triangle = torch.triu_indices(
            layer_count,
            layer_count,
            offset=1,
            device=raw.device,
        )
        raw_pairs = raw_rsm[triangle[0], triangle[1]]
        projected_pairs = projected_rsm[triangle[0], triangle[1]]

        scale = math.sqrt(raw.shape[1] / projected.shape[1])
        raw_distances = torch.cdist(raw, raw)[triangle[0], triangle[1]]
        projected_distances = (
            torch.cdist(projected, projected)[triangle[0], triangle[1]]
            * scale
        )
        valid_distances = raw_distances > 1e-12
        relative_distance_errors = (
            (projected_distances[valid_distances] - raw_distances[valid_distances])
            .abs()
            / raw_distances[valid_distances]
        )

        raw_pairs_cpu = raw_pairs.cpu().numpy().astype(np.float32, copy=False)
        projected_pairs_cpu = (
            projected_pairs.cpu().numpy().astype(np.float32, copy=False)
        )
        relative_errors_cpu = (
            relative_distance_errors.cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
        self._sample_raw_pairs.append(raw_pairs_cpu)
        self._sample_projected_pairs.append(projected_pairs_cpu)
        self._sample_relative_distance_errors.append(relative_errors_cpu)
        self.raw_rsm_sum += raw_rsm.double().cpu().numpy()
        self.projected_rsm_sum += projected_rsm.double().cpu().numpy()
        self.step_count += 1

    def finish_sample(self) -> None:
        if self._sample_id is None:
            raise RuntimeError("No active sample to finalize")
        if not self._sample_raw_pairs:
            raise RuntimeError(
                f"Sample {self._sample_id} produced no decode observations"
            )

        raw_pairs = np.concatenate(self._sample_raw_pairs)
        projected_pairs = np.concatenate(self._sample_projected_pairs)
        relative_errors = np.concatenate(
            self._sample_relative_distance_errors
        )
        difference = projected_pairs - raw_pairs
        self.sample_metrics.append(
            {
                "sample_id": self._sample_id,
                "orig_id": self._orig_id,
                "decode_steps": len(self._sample_raw_pairs),
                "pair_count": int(raw_pairs.size),
                "pearson_r": _correlation(raw_pairs, projected_pairs),
                "spearman_rho": _spearman(raw_pairs, projected_pairs),
                "cosine_mae": float(np.mean(np.abs(difference))),
                "cosine_rmse": float(np.sqrt(np.mean(difference**2))),
                "relative_distance_error_mean": float(
                    np.mean(relative_errors)
                ),
                "relative_distance_error_median": float(
                    np.median(relative_errors)
                ),
            }
        )
        self._all_raw_pairs.append(raw_pairs)
        self._all_projected_pairs.append(projected_pairs)
        self._sample_id = None
        self._orig_id = None
        self._sample_raw_pairs = []
        self._sample_projected_pairs = []
        self._sample_relative_distance_errors = []

    def build_summary(
        self,
        *,
        orthogonality_error: float,
        orthogonality_max_abs: float,
        configuration: dict[str, Any],
        elapsed_seconds: float,
    ) -> dict[str, Any]:
        if not self.sample_metrics or self.raw_rsm_sum is None:
            raise RuntimeError("No projection observations were collected")

        raw_pairs = np.concatenate(self._all_raw_pairs)
        projected_pairs = np.concatenate(self._all_projected_pairs)
        difference = projected_pairs - raw_pairs
        mean_raw_rsm = self.raw_rsm_sum / self.step_count
        mean_projected_rsm = self.projected_rsm_sum / self.step_count
        triangle = np.triu_indices_from(mean_raw_rsm, k=1)
        mean_raw_pairs = mean_raw_rsm[triangle]
        mean_projected_pairs = mean_projected_rsm[triangle]
        mean_difference = mean_projected_pairs - mean_raw_pairs
        sample_rho = np.asarray(
            [item["spearman_rho"] for item in self.sample_metrics],
            dtype=np.float64,
        )
        rho_ci = _bootstrap_median_ci(
            sample_rho,
            replicates=self.bootstrap_replicates,
            seed=self.seed,
        )
        median_rho = float(np.median(sample_rho))
        summary = {
            "configuration": configuration,
            "samples": len(self.sample_metrics),
            "decode_steps": self.step_count,
            "layer_count": len(self.layer_ids or ()),
            "input_dim": self.input_dim,
            "projection_dim": self.output_dim,
            "pair_count": int(raw_pairs.size),
            "elapsed_seconds": elapsed_seconds,
            "orthogonality": {
                "relative_frobenius_error": orthogonality_error,
                "max_abs_error": orthogonality_max_abs,
            },
            "global_metrics": {
                "pearson_r": _correlation(raw_pairs, projected_pairs),
                "spearman_rho": _spearman(raw_pairs, projected_pairs),
                "cosine_mae": float(np.mean(np.abs(difference))),
                "cosine_rmse": float(np.sqrt(np.mean(difference**2))),
            },
            "mean_rsm_metrics": {
                "pearson_r": _correlation(
                    mean_raw_pairs,
                    mean_projected_pairs,
                ),
                "spearman_rho": _spearman(
                    mean_raw_pairs,
                    mean_projected_pairs,
                ),
                "cosine_mae": float(np.mean(np.abs(mean_difference))),
                "cosine_rmse": float(
                    np.sqrt(np.mean(mean_difference**2))
                ),
            },
            "sample_level_rsa": {
                "median_spearman_rho": median_rho,
                "mean_spearman_rho": float(np.mean(sample_rho)),
                "minimum_spearman_rho": float(np.min(sample_rho)),
                "bootstrap_median_95_ci": rho_ci,
            },
            "mean_raw_rsm": mean_raw_rsm.tolist(),
            "mean_projected_rsm": mean_projected_rsm.tolist(),
            "sample_metrics": self.sample_metrics,
        }
        return summary

    def _initialize_shapes(
        self,
        raw: torch.Tensor,
        projected: torch.Tensor,
        layer_ids: tuple[int, ...],
    ) -> None:
        self.layer_ids = layer_ids
        self.input_dim = int(raw.shape[1])
        self.output_dim = int(projected.shape[1])
        shape = (len(layer_ids), len(layer_ids))
        self.raw_rsm_sum = np.zeros(shape, dtype=np.float64)
        self.projected_rsm_sum = np.zeros(shape, dtype=np.float64)


def _rankdata(values: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(
        values,
        return_inverse=True,
        return_counts=True,
    )
    ends = np.cumsum(counts)
    starts = ends - counts
    average_ranks = (starts + ends - 1) / 2.0
    return average_ranks[inverse]


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size != right.size or left.size < 2:
        raise ValueError("Correlation inputs must have equal nontrivial sizes")
    left_centered = left.astype(np.float64) - float(np.mean(left))
    right_centered = right.astype(np.float64) - float(np.mean(right))
    denominator = math.sqrt(
        float(np.dot(left_centered, left_centered))
        * float(np.dot(right_centered, right_centered))
    )
    if denominator == 0.0:
        return 0.0
    return float(np.dot(left_centered, right_centered) / denominator)


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    return _correlation(_rankdata(left), _rankdata(right))


def _bootstrap_median_ci(
    values: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> list[float]:
    if replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        len(values),
        size=(replicates, len(values)),
    )
    medians = np.median(values[indices], axis=1)
    lower, upper = np.percentile(medians, [2.5, 97.5])
    return [float(lower), float(upper)]


def _orthogonality_metrics(matrix: torch.Tensor) -> tuple[float, float]:
    matrix = matrix.detach().float()
    identity = torch.eye(
        matrix.shape[1],
        dtype=matrix.dtype,
        device=matrix.device,
    )
    error = matrix.T @ matrix - identity
    relative = torch.linalg.norm(error) / matrix.shape[1]
    return float(relative.item()), float(error.abs().max().item())


def _plot_results(
    summary: dict[str, Any],
    output: Path,
    *,
    dpi: int,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Nimbus Roman",
                "Times New Roman",
                "Times",
                "DejaVu Serif",
            ],
            "font.size": 8.0,
            "font.weight": "normal",
            "axes.linewidth": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    raw_rsm = np.asarray(summary["mean_raw_rsm"], dtype=np.float64)
    projected_rsm = np.asarray(
        summary["mean_projected_rsm"],
        dtype=np.float64,
    )
    off_diagonal = np.concatenate(
        [
            raw_rsm[np.triu_indices_from(raw_rsm, k=1)],
            projected_rsm[np.triu_indices_from(projected_rsm, k=1)],
        ]
    )
    color_min = float(np.quantile(off_diagonal, 0.01))

    figure = plt.figure(figsize=(5.5, 1.85))
    grid = figure.add_gridspec(
        1,
        4,
        width_ratios=(1.0, 1.0, 0.045, 1.15),
        left=0.07,
        right=0.98,
        bottom=0.23,
        top=0.86,
        wspace=0.34,
    )
    raw_axis = figure.add_subplot(grid[0, 0])
    projected_axis = figure.add_subplot(grid[0, 1])
    colorbar_axis = figure.add_subplot(grid[0, 2])
    scatter_axis = figure.add_subplot(grid[0, 3])

    image = raw_axis.imshow(
        raw_rsm,
        cmap="viridis",
        vmin=color_min,
        vmax=1.0,
        origin="lower",
        interpolation="nearest",
    )
    projected_axis.imshow(
        projected_rsm,
        cmap="viridis",
        vmin=color_min,
        vmax=1.0,
        origin="lower",
        interpolation="nearest",
    )
    for axis, title in (
        (raw_axis, "(a) Before projection"),
        (projected_axis, "(b) After projection"),
    ):
        axis.set_title(title, fontsize=8, pad=3)
        axis.set_xlabel("Layer index", labelpad=1)
        axis.set_xticks([0, 10, 20, 27])
        axis.set_yticks([0, 10, 20, 27])
        axis.tick_params(width=0.5, length=2, labelsize=7)
    raw_axis.set_ylabel("Layer index", labelpad=1)
    projected_axis.tick_params(labelleft=False)
    figure.colorbar(
        image,
        cax=colorbar_axis,
    )
    colorbar_axis.yaxis.set_ticks_position("left")
    colorbar_axis.tick_params(width=0.5, length=2, labelsize=7)

    triangle = np.triu_indices_from(raw_rsm, k=1)
    scatter_raw = raw_rsm[triangle]
    scatter_projected = projected_rsm[triangle]

    lower = float(min(scatter_raw.min(), scatter_projected.min()))
    upper = float(max(scatter_raw.max(), scatter_projected.max()))
    margin = max((upper - lower) * 0.03, 1e-3)
    limits = (lower - margin, upper + margin)
    scatter_axis.scatter(
        scatter_raw,
        scatter_projected,
        s=9,
        alpha=0.42,
        color="#228be6",
        edgecolors="none",
        rasterized=True,
    )
    scatter_axis.plot(limits, limits, color="#c44e52", linewidth=0.8)
    scatter_axis.set_xlim(limits)
    scatter_axis.set_ylim(limits)
    scatter_axis.set_aspect("equal", adjustable="box")
    scatter_axis.set_title(
        "(c) Preservation",
        fontsize=8,
        pad=3,
    )
    scatter_axis.set_xlabel("Before", labelpad=2)
    scatter_axis.set_ylabel("After", labelpad=2)
    scatter_axis.tick_params(width=0.5, length=2, labelsize=7)
    metrics = summary["mean_rsm_metrics"]
    scatter_axis.text(
        0.03,
        0.97,
        (
            f"Pearson r = {metrics['pearson_r']:.3f}\n"
            f"Spearman $\\rho$ = {metrics['spearman_rho']:.3f}"
        ),
        transform=scatter_axis.transAxes,
        ha="left",
        va="top",
        fontsize=6,
        bbox={
            "facecolor": "white",
            "edgecolor": "black",
            "linewidth": 0.4,
            "pad": 1.5,
        },
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    if output.suffix.lower() != ".pdf":
        figure.savefig(
            output.with_suffix(".pdf"),
            bbox_inches="tight",
            pad_inches=0.02,
        )
    plt.close(figure)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run clean Qwen2.5-VL/LingoQA inference and audit whether "
            "orthogonal projection preserves inter-layer relationships."
        )
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=REPOSITORY_ROOT / "configs/models/qwen25_vl.yaml",
    )
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=REPOSITORY_ROOT / "configs/datasets/lingoqa.yaml",
    )
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "analysis/qwen_lingoqa_projection_preservation.json"
        ),
    )
    parser.add_argument(
        "--figure-output",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "figures/qwen_lingoqa_projection_preservation.png"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.projection_dim <= 0:
        parser.error("--projection-dim must be positive")
    if args.bootstrap_replicates <= 0:
        parser.error("--bootstrap-replicates must be positive")
    return args


def main() -> int:
    args = _parse_args()
    summary_output = args.summary_output.resolve()
    figure_output = args.figure_output.resolve()
    outputs = [
        summary_output,
        figure_output,
        figure_output.with_suffix(".pdf"),
    ]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        paths = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Outputs already exist; use --overwrite: {paths}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model_adapter = load_model_adapter(args.model_config.resolve())
    dataset_adapter = load_dataset_adapter(args.dataset_config.resolve())
    collector = ProjectionRelationCollector(
        seed=args.seed,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    profiler = None
    start_time = time.monotonic()
    completed_samples = 0
    try:
        model_adapter.load(args.device)
        profiler = Profiler(
            model_adapter.model,
            proj_dim=args.projection_dim,
            proj_method="project",
            seed=args.seed,
            projection_observer=collector.observe,
        )
        profiler.register()
        if profiler.shared_proj_mat is None:
            raise RuntimeError("Profiler did not initialize a projection matrix")
        orthogonality_error, orthogonality_max_abs = (
            _orthogonality_metrics(profiler.shared_proj_mat)
        )

        for sample_id, sample in enumerate(
            dataset_adapter.iter_samples(max_samples=args.max_samples)
        ):
            collector.start_sample(sample_id, sample.orig_id)
            model_adapter.generate(
                sample.question,
                sample.image,
                max_new_tokens=args.max_new_tokens,
            )
            profiler.finalize()
            collector.finish_sample()
            profiler.reset(clear_stats=True)
            completed_samples += 1
            print(
                (
                    "\r[projection-audit] "
                    f"samples={completed_samples}/{args.max_samples} "
                    f"decode_steps={collector.step_count}"
                ),
                end="",
                flush=True,
            )
        print()
    finally:
        if profiler is not None:
            profiler.unregister()
        model_adapter.close()

    elapsed_seconds = time.monotonic() - start_time
    configuration = {
        "model_config": str(args.model_config.resolve()),
        "dataset_config": str(args.dataset_config.resolve()),
        "device": args.device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "max_samples": args.max_samples,
        "max_new_tokens": args.max_new_tokens,
        "projection_method": "project",
        "projection_dim": args.projection_dim,
        "seed": args.seed,
        "monitored_module": "language_model.layers[*].self_attn.o_proj",
        "fault_injection": False,
        "decode_only": True,
    }
    summary = collector.build_summary(
        orthogonality_error=orthogonality_error,
        orthogonality_max_abs=orthogonality_max_abs,
        configuration=configuration,
        elapsed_seconds=elapsed_seconds,
    )
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_summary = summary_output.with_name(
        f"{summary_output.name}.tmp"
    )
    temporary_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_summary.replace(summary_output)
    _plot_results(
        summary,
        figure_output,
        dpi=args.dpi,
    )
    print(
        json.dumps(
            {
                "summary": str(summary_output),
                "figure": str(figure_output),
                "figure_pdf": str(figure_output.with_suffix(".pdf")),
                "samples": summary["samples"],
                "decode_steps": summary["decode_steps"],
                "global_metrics": summary["global_metrics"],
                "mean_rsm_metrics": summary["mean_rsm_metrics"],
                "sample_level_rsa": summary["sample_level_rsa"],
                "orthogonality": summary["orthogonality"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
