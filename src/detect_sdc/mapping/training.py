"""Shared mapping-model architecture and training engine."""

from __future__ import annotations

import json
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm


class InterLayerJsonlTensorDataset(Dataset):
    """Load adjacent-layer projection pairs from profiler JSONL output."""

    def __init__(self, file_path: str | Path):
        xs: list[Any] = []
        ys: list[Any] = []
        src_layers: list[int] = []
        tgt_layers: list[int] = []
        steps: list[int] = []

        print(f"[Info] Start reading: {file_path}")
        with Path(file_path).open("r", encoding="utf-8") as stream:
            for line in tqdm(stream, desc="Reading jsonl", unit="lines"):
                if not line.strip():
                    continue
                record = json.loads(line)
                xs.append(record["x"])
                ys.append(record["y"])
                src_layers.append(int(record["src_layer"]))
                tgt_layers.append(int(record["tgt_layer"]))
                steps.append(int(record["step"]))

        self.x = torch.tensor(xs, dtype=torch.float32)
        self.y = torch.tensor(ys, dtype=torch.float32)
        self.src_layer = torch.tensor(src_layers, dtype=torch.long)
        self.tgt_layer = torch.tensor(tgt_layers, dtype=torch.long)
        self.step = torch.tensor(steps, dtype=torch.long)
        if self.x.ndim != 2 or self.y.shape != self.x.shape:
            raise ValueError(
                "Mapping data must contain equally shaped 2D x/y vectors"
            )
        print(f"[Info] Dataset loaded: {self.x.size(0)} samples")

    def __len__(self) -> int:
        return self.x.size(0)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "x": self.x[index],
            "y": self.y[index],
            "src_layer": self.src_layer[index],
            "tgt_layer": self.tgt_layer[index],
            "step": self.step[index],
        }


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim * 2, dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.norm(inputs)
        hidden = self.fc1(hidden)
        hidden = self.act(hidden)
        hidden = self.dropout(hidden)
        hidden = self.fc2(hidden)
        hidden = self.dropout(hidden)
        return inputs + hidden


class LayerAwareResidualMLP(nn.Module):
    """Residual projection predictor shared by Qwen and LLaVA jobs."""

    def __init__(
        self,
        x_dim: int = 64,
        num_layers: int = 28,
        layer_emb_dim: int = 16,
        hidden_dim: int = 64,
        num_blocks: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.src_emb = nn.Embedding(num_layers, layer_emb_dim)
        self.tgt_emb = nn.Embedding(num_layers, layer_emb_dim)
        self.norm_x = nn.LayerNorm(x_dim)
        self.input_proj = nn.Linear(
            x_dim + 2 * layer_emb_dim,
            hidden_dim,
        )
        self.blocks = nn.Sequential(
            *[
                ResidualMLPBlock(hidden_dim, dropout=dropout)
                for _ in range(num_blocks)
            ]
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, x_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        src_layer: torch.Tensor,
        tgt_layer: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.norm_x(inputs)
        hidden = torch.cat(
            [
                normalized,
                self.src_emb(src_layer),
                self.tgt_emb(tgt_layer),
            ],
            dim=-1,
        )
        hidden = self.input_proj(hidden)
        hidden = self.blocks(hidden)
        hidden = self.out_norm(hidden)
        return inputs + self.out_proj(hidden)


@dataclass(frozen=True)
class MappingSplits:
    train: Subset
    selection: Subset
    final: Subset
    selection_name: str
    final_name: str


def split_mapping_dataset(
    dataset: Dataset,
    *,
    strategy: str,
    valid_ratio: float,
    test_ratio: float,
    test_ratio_in_train: float,
    seed: int,
) -> MappingSplits:
    """Create the two historical split layouts without duplicating trainers."""

    total = len(dataset)
    if total <= 0:
        raise ValueError("Mapping dataset is empty")
    for name, ratio in (
        ("valid_ratio", valid_ratio),
        ("test_ratio", test_ratio),
        ("test_ratio_in_train", test_ratio_in_train),
    ):
        if not 0.0 < ratio < 1.0:
            raise ValueError(f"{name} must be between 0 and 1")

    if strategy == "sequential":
        final_size = int(total * valid_ratio)
        training_pool_size = total - final_size
        selection_size = int(training_pool_size * test_ratio_in_train)
        train_size = training_pool_size - selection_size
        _validate_split_sizes(train_size, selection_size, final_size)
        train_indices = list(range(train_size))
        selection_indices = list(range(train_size, training_pool_size))
        final_indices = list(range(training_pool_size, total))
        selection_name, final_name = "Test", "Valid"
    elif strategy == "random":
        selection_size = int(total * valid_ratio)
        final_size = int(total * test_ratio)
        train_size = total - selection_size - final_size
        _validate_split_sizes(train_size, selection_size, final_size)
        indices = list(range(total))
        random.Random(seed).shuffle(indices)
        final_indices = indices[:final_size]
        selection_indices = indices[
            final_size : final_size + selection_size
        ]
        train_indices = indices[final_size + selection_size :]
        selection_name, final_name = "Valid", "Test"
    elif strategy == "partition":
        selection_size = int(total * valid_ratio)
        final_size = int(total * test_ratio)
        train_size = total - selection_size - final_size
        _validate_split_sizes(train_size, selection_size, final_size)
        train_indices = list(range(train_size))
        selection_indices = list(
            range(train_size, train_size + selection_size)
        )
        final_indices = list(
            range(train_size + selection_size, total)
        )
        selection_name, final_name = "Valid", "Test"
    else:
        raise ValueError(
            "mapping split strategy must be sequential, random, or partition"
        )

    print(
        f"[Split] strategy={strategy} total={total} "
        f"train={len(train_indices)} "
        f"{selection_name.lower()}={len(selection_indices)} "
        f"{final_name.lower()}={len(final_indices)}"
    )
    return MappingSplits(
        train=Subset(dataset, train_indices),
        selection=Subset(dataset, selection_indices),
        final=Subset(dataset, final_indices),
        selection_name=selection_name,
        final_name=final_name,
    )


def regression_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    cosine_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    mse = F.mse_loss(prediction, target)
    cosine_loss = (
        1.0
        - F.cosine_similarity(prediction, target, dim=-1).mean()
    )
    total = mse + cosine_weight * cosine_loss
    return total, {
        "mse": mse.item(),
        "cos_loss": cosine_loss.item(),
        "cos_sim": 1.0 - cosine_loss.item(),
        "total": total.item(),
    }


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    *,
    device: str,
    cosine_weight: float,
    amp_enabled: bool,
) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "mse": 0.0, "cos_loss": 0.0}
    count = 0
    with torch.no_grad():
        for batch in data_loader:
            inputs, target, src_layer, tgt_layer = _move_batch(batch, device)
            with _autocast(amp_enabled):
                prediction = model(inputs, src_layer, tgt_layer)
                loss, metrics = regression_loss(
                    prediction,
                    target,
                    cosine_weight=cosine_weight,
                )
            batch_size = inputs.size(0)
            totals["loss"] += loss.item() * batch_size
            totals["mse"] += metrics["mse"] * batch_size
            totals["cos_loss"] += metrics["cos_loss"] * batch_size
            count += batch_size

    average = {
        key: value / max(count, 1)
        for key, value in totals.items()
    }
    average["cos_sim"] = 1.0 - average["cos_loss"]
    return average


def evaluate_final_split(
    model: nn.Module,
    data_loader: DataLoader,
    *,
    device: str,
    split_name: str,
    amp_enabled: bool,
) -> dict[str, float]:
    model.eval()
    total_mse = 0.0
    total_cosine = 0.0
    total_mean_target = 0.0
    total_mean_prediction = 0.0
    total_hamming = 0.0
    count = 0
    with torch.no_grad():
        for batch in data_loader:
            inputs, target, src_layer, tgt_layer = _move_batch(batch, device)
            with _autocast(amp_enabled):
                prediction = model(inputs, src_layer, tgt_layer)
            total_mse += (
                torch.mean((prediction - target) ** 2, dim=-1)
                .sum()
                .item()
            )
            total_cosine += (
                F.cosine_similarity(prediction, target, dim=-1).sum().item()
            )
            total_mean_target += target.mean(dim=-1).sum().item()
            total_mean_prediction += (
                prediction.mean(dim=-1).sum().item()
            )
            target_order = torch.argsort(target, dim=-1, stable=True)
            prediction_order = torch.argsort(
                prediction,
                dim=-1,
                stable=True,
            )
            total_hamming += (
                (prediction_order != target_order).sum(dim=-1).sum().item()
            )
            count += inputs.size(0)

    mse = total_mse / max(count, 1)
    metrics = {
        "rmse": mse**0.5,
        "mse": mse,
        "cosine_similarity": total_cosine / max(count, 1),
        "mean_y": total_mean_target / max(count, 1),
        "mean_y_hat": total_mean_prediction / max(count, 1),
        "hamming_rank": total_hamming / max(count, 1),
    }
    print(
        f"[{split_name}] RMSE={metrics['rmse']:.6f} "
        f"cosine_similarity={metrics['cosine_similarity']:.6f}"
    )
    return metrics


def train_model(
    jsonl_path: str,
    save_best_path: str,
    *,
    model_kwargs: Mapping[str, Any] | None = None,
    batch_size: int = 2048,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    epochs: int = 500,
    num_workers: int = 4,
    split_strategy: str = "sequential",
    valid_ratio: float = 0.15,
    test_ratio: float = 0.1,
    test_ratio_in_train: float = 0.15,
    cosine_weight: float = 1.0,
    early_stop_patience: int = 5,
    scheduler_enabled: bool = True,
    scheduler_patience: int = 5,
    scheduler_factor: float = 0.5,
    min_lr: float = 1e-6,
    use_amp: bool = True,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    final_metrics: str = "detailed",
    seed: int = 42,
    device: str | None = None,
) -> tuple[nn.Module, dict[str, float]]:
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if batch_size <= 0 or epochs <= 0 or num_workers < 0:
        raise ValueError("Invalid mapping training batch/epoch/worker values")
    if cosine_weight < 0:
        raise ValueError("cosine_weight must be non-negative")
    if final_metrics not in {"detailed", "loss"}:
        raise ValueError("final_metrics must be 'detailed' or 'loss'")

    dataset = InterLayerJsonlTensorDataset(jsonl_path)
    splits = split_mapping_dataset(
        dataset,
        strategy=split_strategy,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
        test_ratio_in_train=test_ratio_in_train,
        seed=seed,
    )
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers and num_workers > 0,
    }
    train_loader = DataLoader(
        splits.train,
        shuffle=True,
        **loader_kwargs,
    )
    selection_loader = DataLoader(
        splits.selection,
        shuffle=False,
        **loader_kwargs,
    )
    final_loader = DataLoader(
        splits.final,
        shuffle=False,
        **loader_kwargs,
    )

    model = LayerAwareResidualMLP(**dict(model_kwargs or {})).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_factor,
            patience=scheduler_patience,
            threshold=1e-4,
            min_lr=min_lr,
        )
        if scheduler_enabled
        else None
    )
    amp_enabled = use_amp and str(device).startswith("cuda")
    scaler = _gradient_scaler() if amp_enabled else None

    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    no_improve = 0
    for epoch in range(1, epochs + 1):
        train_loss = _train_epoch(
            model,
            train_loader,
            optimizer,
            scaler=scaler,
            device=device,
            cosine_weight=cosine_weight,
            amp_enabled=amp_enabled,
        )
        selection_metrics = evaluate_model(
            model,
            selection_loader,
            device=device,
            cosine_weight=cosine_weight,
            amp_enabled=amp_enabled,
        )
        if scheduler is not None:
            scheduler.step(selection_metrics["loss"])
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d} | lr={current_lr:.6e} | "
            f"train_loss={train_loss:.6f} | "
            f"{splits.selection_name.lower()}_loss="
            f"{selection_metrics['loss']:.6f}"
        )

        if selection_metrics["loss"] < best_loss - 1e-4:
            best_loss = selection_metrics["loss"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            torch.save(best_state, save_best_path)
            no_improve = 0
            print(f"[Info] Best model saved to: {save_best_path}")
        else:
            no_improve += 1
            if no_improve >= early_stop_patience:
                print(f"[Info] Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    if final_metrics == "loss":
        metrics = evaluate_model(
            model,
            final_loader,
            device=device,
            cosine_weight=cosine_weight,
            amp_enabled=amp_enabled,
        )
    else:
        metrics = evaluate_final_split(
            model,
            final_loader,
            device=device,
            split_name=splits.final_name,
            amp_enabled=amp_enabled,
        )
    return model, metrics


def _train_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    scaler: Any,
    device: str,
    cosine_weight: float,
    amp_enabled: bool,
) -> float:
    model.train()
    loss_sum = 0.0
    count = 0
    for batch in data_loader:
        inputs, target, src_layer, tgt_layer = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(amp_enabled):
            prediction = model(inputs, src_layer, tgt_layer)
            loss, _ = regression_loss(
                prediction,
                target,
                cosine_weight=cosine_weight,
            )
        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        loss_sum += loss.item() * inputs.size(0)
        count += inputs.size(0)
    return loss_sum / max(count, 1)


def _move_batch(
    batch: Mapping[str, torch.Tensor],
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        batch["x"].to(device, non_blocking=True),
        batch["y"].to(device, non_blocking=True),
        batch["src_layer"].to(device, non_blocking=True),
        batch["tgt_layer"].to(device, non_blocking=True),
    )


def _autocast(enabled: bool):
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda", enabled=True)
    legacy_autocast = getattr(torch.cuda.amp, "autocast")
    return legacy_autocast(enabled=True)


def _gradient_scaler():
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=True)
    legacy_scaler = getattr(torch.cuda.amp, "GradScaler")
    return legacy_scaler(enabled=True)


def _validate_split_sizes(
    train_size: int,
    selection_size: int,
    final_size: int,
) -> None:
    if min(train_size, selection_size, final_size) <= 0:
        raise ValueError(
            "Mapping train/selection/final split must all be non-empty"
        )
