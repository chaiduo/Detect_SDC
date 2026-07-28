"""Compatibility API for the historical Qwen mapping trainer."""

from __future__ import annotations

import argparse
from typing import Any

from .training import (
    InterLayerJsonlTensorDataset,
    LayerAwareResidualMLP,
    ResidualMLPBlock,
    evaluate_final_split,
    evaluate_model,
    regression_loss,
    split_mapping_dataset,
    train_model as _train_model,
)


def split_dataset_sequential_valid(
    dataset,
    valid_ratio: float = 0.2,
    test_ratio_in_train: float = 0.2,
    seed: int = 42,
):
    splits = split_mapping_dataset(
        dataset,
        strategy="sequential",
        valid_ratio=valid_ratio,
        test_ratio=0.1,
        test_ratio_in_train=test_ratio_in_train,
        seed=seed,
    )
    return splits.train, splits.selection, splits.final


def train_model(
    jsonl_path: str,
    save_best_path: str = "best_model.pt",
    num_layers: int = 28,
    batch_size: int = 2048,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 200,
    num_workers: int = 4,
    valid_ratio: float = 0.15,
    test_ratio_in_train: float = 0.1,
    cosine_weight: float = 0.1,
    early_stop_patience: int = 5,
    scheduler_patience: int = 5,
    scheduler_factor: float = 0.5,
    min_lr: float = 1e-6,
    seed: int = 42,
    device: str | None = None,
    **kwargs: Any,
):
    return _train_model(
        jsonl_path,
        save_best_path,
        model_kwargs={
            "x_dim": 64,
            "num_layers": num_layers,
            "layer_emb_dim": 16,
            "hidden_dim": 64,
            "num_blocks": 4,
            "dropout": 0.1,
        },
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs,
        num_workers=num_workers,
        split_strategy="sequential",
        valid_ratio=valid_ratio,
        test_ratio_in_train=test_ratio_in_train,
        cosine_weight=cosine_weight,
        early_stop_patience=early_stop_patience,
        scheduler_patience=scheduler_patience,
        scheduler_factor=scheduler_factor,
        min_lr=min_lr,
        seed=seed,
        device=device,
        **kwargs,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_path", required=True)
    parser.add_argument("--save_best_path", required=True)
    parser.add_argument("--num_layers", type=int, default=28)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device")
    args = parser.parse_args()
    train_model(**vars(args))


__all__ = [
    "InterLayerJsonlTensorDataset",
    "LayerAwareResidualMLP",
    "ResidualMLPBlock",
    "evaluate_final_split",
    "evaluate_model",
    "main",
    "regression_loss",
    "split_dataset_sequential_valid",
    "train_model",
]
