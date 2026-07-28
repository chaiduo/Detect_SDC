"""Compatibility API for the historical LLaVA mapping trainer."""

from __future__ import annotations

import argparse
from typing import Any

from .training import (
    InterLayerJsonlTensorDataset,
    LayerAwareResidualMLP as SharedLayerAwareResidualMLP,
    ResidualMLPBlock,
    evaluate_model,
    split_mapping_dataset,
    train_model as _train_model,
)


class LayerAwareResidualMLP(SharedLayerAwareResidualMLP):
    def __init__(
        self,
        x_dim: int = 64,
        num_layers: int = 32,
        layer_emb_dim: int = 16,
        hidden_dim: int = 256,
        num_blocks: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__(
            x_dim=x_dim,
            num_layers=num_layers,
            layer_emb_dim=layer_emb_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            dropout=dropout,
        )


def split_dataset_sequential_valid(
    dataset,
    valid_ratio: float = 0.15,
    test_ratio_in_train: float = 0.15,
):
    splits = split_mapping_dataset(
        dataset,
        strategy="sequential",
        valid_ratio=valid_ratio,
        test_ratio=0.1,
        test_ratio_in_train=test_ratio_in_train,
        seed=42,
    )
    return splits.train, splits.selection, splits.final


def train_model(
    jsonl_path: str,
    save_best_path: str,
    num_layers: int = 32,
    batch_size: int = 2048,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    epochs: int = 500,
    num_workers: int = 4,
    valid_ratio: float = 0.15,
    test_ratio_in_train: float = 0.15,
    cosine_weight: float = 1.0,
    early_stop_patience: int = 5,
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
            "hidden_dim": 256,
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
        scheduler_enabled=False,
        use_amp=False,
        pin_memory=False,
        persistent_workers=False,
        final_metrics="loss",
        device=device,
        **kwargs,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_path", required=True)
    parser.add_argument("--save_best_path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_layers", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    train_model(**vars(args))


__all__ = [
    "InterLayerJsonlTensorDataset",
    "LayerAwareResidualMLP",
    "ResidualMLPBlock",
    "evaluate_model",
    "main",
    "split_dataset_sequential_valid",
    "train_model",
]
