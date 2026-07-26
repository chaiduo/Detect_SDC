import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, Subset


# =========================
# 1. Dataset
# =========================
class InterLayerJsonlTensorDataset(Dataset):
    def __init__(self, file_path: str):
        xs = []
        ys = []
        src_layers = []
        tgt_layers = []
        steps = []

        print(f"[Info] Start reading: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Reading jsonl", unit="lines"):
                line = line.strip()
                if not line:
                    continue

                obj = json.loads(line)

                xs.append(obj["x"])
                ys.append(obj["y"])
                src_layers.append(obj["src_layer"])
                tgt_layers.append(obj["tgt_layer"])
                steps.append(obj["step"])

        print("[Info] Converting to torch tensors...")

        self.x = torch.tensor(xs, dtype=torch.float32)
        self.y = torch.tensor(ys, dtype=torch.float32)
        self.src_layer = torch.tensor(src_layers, dtype=torch.long)
        self.tgt_layer = torch.tensor(tgt_layers, dtype=torch.long)
        self.step = torch.tensor(steps, dtype=torch.long)

        print(f"[Info] Dataset loaded: {self.x.size(0)} samples")

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return {
            "x": self.x[idx],
            "y": self.y[idx],
            "src_layer": self.src_layer[idx],
            "tgt_layer": self.tgt_layer[idx],
            "step": self.step[idx],
        }


# =========================
# 2. Split
# =========================
def split_dataset_sequential_valid(
    dataset,
    valid_ratio=0.2,
    test_ratio_in_train=0.2,
    seed=42,
):
    """
    切分规则（全顺序切分）：
    - valid = 整个数据集最后 valid_ratio
    - train_full = 前面部分
    - test = train_full 的后 test_ratio_in_train
    - train = train_full 的前面部分
    """
    n_total = len(dataset)
    n_valid = int(n_total * valid_ratio)
    n_train_full = n_total - n_valid

    if n_total <= 0:
        raise ValueError("Dataset is empty.")

    if n_valid <= 0:
        raise ValueError("valid set is empty. Please increase dataset size or valid_ratio.")

    if n_train_full <= 0:
        raise ValueError("train_full is empty. Please reduce valid_ratio.")

    valid_indices = list(range(n_train_full, n_total))
    train_full_indices = list(range(0, n_train_full))

    n_test = int(n_train_full * test_ratio_in_train)
    if n_test <= 0:
        raise ValueError("test set is empty. Please increase dataset size or test_ratio_in_train.")

    n_train = n_train_full - n_test
    if n_train <= 0:
        raise ValueError("train set is empty after split. Please reduce test_ratio_in_train.")

    train_indices = list(range(0, n_train))
    test_indices = list(range(n_train, n_train_full))

    train_set = Subset(dataset, train_indices)
    test_set = Subset(dataset, test_indices)
    valid_set = Subset(dataset, valid_indices)

    print(f"[Split] Total samples : {n_total}")
    print(f"[Split] Train samples : {len(train_set)}")
    print(f"[Split] Test samples  : {len(test_set)}")
    print(f"[Split] Valid samples : {len(valid_set)}")
    print(f"[Split] Train index range: [0, {n_train})")
    print(f"[Split] Test index range : [{n_train}, {n_train_full})")
    print(f"[Split] Valid index range: [{n_train_full}, {n_total})")
    print("[Split] Verified: train/test/valid are mutually disjoint.")

    return train_set, test_set, valid_set


# =========================
# 3. Model
# =========================
class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim * 2, dim)

    def forward(self, x):
        h = self.norm(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.dropout(h)
        return x + h


class LayerAwareResidualMLP(nn.Module):
    def __init__(
        self,
        x_dim=64,
        num_layers=28,
        layer_emb_dim=16,
        hidden_dim=64,
        num_blocks=4,
        dropout=0.1,
    ):
        super().__init__()

        self.src_emb = nn.Embedding(num_layers, layer_emb_dim)
        self.tgt_emb = nn.Embedding(num_layers, layer_emb_dim)

        in_dim = x_dim + 2 * layer_emb_dim

        self.norm_x = nn.LayerNorm(x_dim)
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.blocks = nn.Sequential(
            *[ResidualMLPBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)]
        )

        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, x_dim)

    def forward(self, x, src_layer, tgt_layer):
        x_norm = self.norm_x(x)
        src_e = self.src_emb(src_layer)
        tgt_e = self.tgt_emb(tgt_layer)

        h = torch.cat([x_norm, src_e, tgt_e], dim=-1)
        h = self.input_proj(h)
        h = self.blocks(h)
        h = self.out_norm(h)

        delta = self.out_proj(h)
        y_hat = x + delta
        return y_hat


# =========================
# 4. Loss
# =========================
def regression_loss(pred, target, cosine_weight=0.1):
    mse = F.mse_loss(pred, target)
    cos_loss = 1.0 - F.cosine_similarity(pred, target, dim=-1).mean()
    total = mse + cosine_weight * cos_loss

    return total, {
        "mse": mse.item(),
        "cos_loss": cos_loss.item(),
        "cos_sim": 1.0 - cos_loss.item(),
        "total": total.item(),
    }


# =========================
# 5. Eval
# =========================
def evaluate_model(model, data_loader, device, cosine_weight=0.1, use_amp=False, device_type="cpu"):
    model.eval()

    loss_sum = 0.0
    mse_sum = 0.0
    cos_loss_sum = 0.0
    count = 0

    with torch.no_grad():
        for batch in data_loader:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            src_layer = batch["src_layer"].to(device, non_blocking=True)
            tgt_layer = batch["tgt_layer"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                pred = model(x, src_layer, tgt_layer)
                loss, metrics = regression_loss(pred, y, cosine_weight=cosine_weight)

            bs = x.size(0)
            loss_sum += loss.item() * bs
            mse_sum += metrics["mse"] * bs
            cos_loss_sum += metrics["cos_loss"] * bs
            count += bs

    avg_loss = loss_sum / max(count, 1)
    avg_mse = mse_sum / max(count, 1)
    avg_cos_loss = cos_loss_sum / max(count, 1)
    avg_cos_sim = 1.0 - avg_cos_loss

    return {
        "loss": avg_loss,
        "mse": avg_mse,
        "cos_loss": avg_cos_loss,
        "cos_sim": avg_cos_sim,
    }


# def evaluate_final_split(model, data_loader, device, split_name="Valid", use_amp=False, device_type="cpu"):
#     model.eval()

#     total_mse = 0.0
#     total_cos = 0.0
#     total_samples = 0
#     total_mean_y = 0.0
#     total_mean_y_hat = 0.0

#     with torch.no_grad():
#         for batch in data_loader:
#             x = batch["x"].to(device, non_blocking=True)
#             y = batch["y"].to(device, non_blocking=True)
#             src_layer = batch["src_layer"].to(device, non_blocking=True)
#             tgt_layer = batch["tgt_layer"].to(device, non_blocking=True)

#             with torch.amp.autocast(device_type=device_type, enabled=use_amp):
#                 y_hat = model(x, src_layer, tgt_layer)

#             mean_yhat = torch.mean(y_hat, dim=-1)
#             mean_y = torch.mean(y, dim=-1)
#             mse = torch.mean((y_hat - y) ** 2, dim=-1)
#             cos = F.cosine_similarity(y_hat, y, dim=-1)

#             total_mean_y += mean_y.sum().item()
#             total_mean_y_hat += mean_yhat.sum().item()
#             total_mse += mse.sum().item()
#             total_cos += cos.sum().item()
#             total_samples += x.size(0)

#     avg_mse = total_mse / max(total_samples, 1)
#     rmse = avg_mse ** 0.5
#     avg_cos = total_cos / max(total_samples, 1)
#     avg_mean_y = total_mean_y / max(total_samples, 1)
#     avg_mean_y_hat = total_mean_y_hat / max(total_samples, 1)

#     print(f"[{split_name}] RMSE: {rmse:.6f}")
#     print(f"[{split_name}] Cosine Similarity: {avg_cos:.6f}")
#     print(f"[{split_name}] Mean y: {avg_mean_y:.6f}")
#     print(f"[{split_name}] Mean y_hat: {avg_mean_y_hat:.6f}")

#     return {
#         "rmse": rmse,
#         "mse": avg_mse,
#         "cosine_similarity": avg_cos,
#         "mean_y": avg_mean_y,
#         "mean_y_hat": avg_mean_y_hat,
#     }
def evaluate_final_split(model, data_loader, device, split_name="Valid", use_amp=False, device_type="cpu"):
    model.eval()

    total_mse = 0.0
    total_cos = 0.0
    total_samples = 0
    total_mean_y = 0.0
    total_mean_y_hat = 0.0
    total_hamming = 0.0  # 新增

    with torch.no_grad():
        for batch in data_loader:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            src_layer = batch["src_layer"].to(device, non_blocking=True)
            tgt_layer = batch["tgt_layer"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                y_hat = model(x, src_layer, tgt_layer)

            mean_yhat = torch.mean(y_hat, dim=-1)
            mean_y = torch.mean(y, dim=-1)
            mse = torch.mean((y_hat - y) ** 2, dim=-1)
            cos = F.cosine_similarity(y_hat, y, dim=-1)

            # ===== 排序汉明距离（稳定排序）=====
            idx_y_hat = torch.argsort(y_hat, dim=-1, stable=True)
            idx_y = torch.argsort(y, dim=-1, stable=True)

            hamming = (idx_y_hat != idx_y).sum(dim=-1)  # 每个样本
            total_hamming += hamming.sum().item()
            # ==================================

            total_mean_y += mean_y.sum().item()
            total_mean_y_hat += mean_yhat.sum().item()
            total_mse += mse.sum().item()
            total_cos += cos.sum().item()
            total_samples += x.size(0)

    avg_mse = total_mse / max(total_samples, 1)
    rmse = avg_mse ** 0.5
    avg_cos = total_cos / max(total_samples, 1)
    avg_mean_y = total_mean_y / max(total_samples, 1)
    avg_mean_y_hat = total_mean_y_hat / max(total_samples, 1)
    avg_hamming = total_hamming / max(total_samples, 1)

    print(f"[{split_name}] RMSE: {rmse:.6f}")
    print(f"[{split_name}] Cosine Similarity: {avg_cos:.6f}")
    print(f"[{split_name}] Mean y: {avg_mean_y:.6f}")
    print(f"[{split_name}] Mean y_hat: {avg_mean_y_hat:.6f}")
    print(f"[{split_name}] Hamming (rank): {avg_hamming:.6f}")

    return {
        "rmse": rmse,
        "mse": avg_mse,
        "cosine_similarity": avg_cos,
        "mean_y": avg_mean_y,
        "mean_y_hat": avg_mean_y_hat,
        "hamming_rank": avg_hamming,
    }

# =========================
# 6. Train
# =========================
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
    device: str = "cuda:4" if torch.cuda.is_available() else "cpu",
):
    device_type = "cuda" if "cuda" in device else "cpu"
    use_amp = (device_type == "cuda")

    # 1) dataset
    dataset = InterLayerJsonlTensorDataset(jsonl_path)

    train_set, test_set, valid_set = split_dataset_sequential_valid(
        dataset,
        valid_ratio=valid_ratio,
        test_ratio_in_train=test_ratio_in_train,
        seed=seed,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    valid_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    # 2) model
    model = LayerAwareResidualMLP(
        x_dim=64,
        num_layers=num_layers,
        layer_emb_dim=16,
        hidden_dim=64,
        num_blocks=4,
        dropout=0.1,
    ).to(device)

    # # 3) optimizer / scheduler / amp
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_factor,
        patience=scheduler_patience,
        threshold=1e-4,
        min_lr=min_lr,
    )

    scaler = torch.amp.GradScaler(device_type, enabled=use_amp)

    # 4) early stopping based on test
    best_test_loss = float("inf")
    best_state = None
    no_improve_count = 0

    # 5) training loop
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            src_layer = batch["src_layer"].to(device, non_blocking=True)
            tgt_layer = batch["tgt_layer"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                pred = model(x, src_layer, tgt_layer)
                loss, _ = regression_loss(pred, y, cosine_weight=cosine_weight)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = x.size(0)
            train_loss_sum += loss.item() * bs
            train_count += bs

        train_loss = train_loss_sum / max(train_count, 1)

        # evaluate only on test
        test_metrics = evaluate_model(
            model,
            test_loader,
            device=device,
            cosine_weight=cosine_weight,
            use_amp=use_amp,
            device_type=device_type,
        )

        # scheduler on test
        scheduler.step(test_metrics["loss"])
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d} | "
            f"lr={current_lr:.6e} | "
            f"train_loss={train_loss:.6f} | "
            f"test_loss={test_metrics['loss']:.6f} | "
            f"test_mse={test_metrics['mse']:.6f} | "
            f"test_cos_loss={test_metrics['cos_loss']:.6f} | "
            f"test_cos_sim={test_metrics['cos_sim']:.6f}"
        )

        # save best by test loss
        if test_metrics["loss"] < best_test_loss - 1e-4:
            best_test_loss = test_metrics["loss"]
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            torch.save(best_state, save_best_path)
            no_improve_count = 0
            print(f"[Info] Best model updated and saved to: {save_best_path}")
        else:
            no_improve_count += 1

        # early stopping
        if no_improve_count >= early_stop_patience:
            print(f"[Info] Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # final valid evaluation
    print("[Info] Training finished. Running final evaluation on valid set...")
    final_valid_metrics = evaluate_final_split(
        model,
        valid_loader,
        device=device,
        split_name="Valid",
        use_amp=use_amp,
        device_type=device_type,
    )
    
    return model, final_valid_metrics

# =========================
# 7. Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_path", type=str, default="./json/attn_proj_mapping_64_project.jsonl")
    parser.add_argument("--save_best_path", type=str, default="best_model.pt")
    parser.add_argument("--num_layers", type=int, default=28)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--valid_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio_in_train", type=float, default=0.15)
    parser.add_argument("--cosine_weight", type=float, default=1)
    parser.add_argument("--early_stop_patience", type=int, default=5)
    parser.add_argument("--scheduler_patience", type=int, default=5)
    parser.add_argument("--scheduler_factor", type=float, default=0.5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model, valid_metrics = train_model(
        jsonl_path=args.jsonl_path,
        save_best_path=args.save_best_path,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        num_workers=args.num_workers,
        valid_ratio=args.valid_ratio,
        test_ratio_in_train=args.test_ratio_in_train,
        cosine_weight=args.cosine_weight,
        early_stop_patience=args.early_stop_patience,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor,
        min_lr=args.min_lr,
        seed=args.seed,
        device=args.device,
    )

    print("[Info] Final valid metrics:")
    print(valid_metrics)


if __name__ == "__main__":
    main()
