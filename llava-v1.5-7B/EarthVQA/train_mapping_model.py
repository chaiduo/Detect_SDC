import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm


class InterLayerJsonlTensorDataset(Dataset):
    def __init__(self, file_path: str):
        xs, ys, src_layers, tgt_layers, steps = [], [], [], [], []
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


def split_dataset_sequential_valid(dataset, valid_ratio=0.15, test_ratio_in_train=0.15):
    n_total = len(dataset)
    n_valid = int(n_total * valid_ratio)
    n_train_full = n_total - n_valid
    n_test = int(n_train_full * test_ratio_in_train)
    n_train = n_train_full - n_test
    if min(n_valid, n_test, n_train) <= 0:
        raise ValueError("Invalid split. Please increase dataset size or adjust ratios.")

    train_set = Subset(dataset, list(range(0, n_train)))
    test_set = Subset(dataset, list(range(n_train, n_train_full)))
    valid_set = Subset(dataset, list(range(n_train_full, n_total)))
    print(f"[Split] Total: {n_total}, Train: {len(train_set)}, Test: {len(test_set)}, Valid: {len(valid_set)}")
    return train_set, test_set, valid_set


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
    def __init__(self, x_dim=64, num_layers=32, layer_emb_dim=16, hidden_dim=256, num_blocks=4, dropout=0.1):
        super().__init__()
        self.src_emb = nn.Embedding(num_layers, layer_emb_dim)
        self.tgt_emb = nn.Embedding(num_layers, layer_emb_dim)
        self.norm_x = nn.LayerNorm(x_dim)
        self.input_proj = nn.Linear(x_dim + 2 * layer_emb_dim, hidden_dim)
        self.blocks = nn.Sequential(*[ResidualMLPBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)])
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, x_dim)

    def forward(self, x, src_layer, tgt_layer):
        x_norm = self.norm_x(x)
        h = torch.cat([x_norm, self.src_emb(src_layer), self.tgt_emb(tgt_layer)], dim=-1)
        h = self.input_proj(h)
        h = self.blocks(h)
        h = self.out_norm(h)
        return x + self.out_proj(h)


def regression_loss(pred, target, cosine_weight=1.0):
    mse = F.mse_loss(pred, target)
    cos_loss = 1.0 - F.cosine_similarity(pred, target, dim=-1).mean()
    total = mse + cosine_weight * cos_loss
    return total, {"mse": mse.item(), "cos_loss": cos_loss.item(), "cos_sim": 1.0 - cos_loss.item()}


def evaluate_model(model, data_loader, device, cosine_weight=1.0):
    model.eval()
    loss_sum, count = 0.0, 0
    with torch.no_grad():
        for batch in data_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            src_layer = batch["src_layer"].to(device)
            tgt_layer = batch["tgt_layer"].to(device)
            pred = model(x, src_layer, tgt_layer)
            loss, _ = regression_loss(pred, y, cosine_weight=cosine_weight)
            loss_sum += loss.item() * x.size(0)
            count += x.size(0)
    return {"loss": loss_sum / max(count, 1)}


def train_model(
    jsonl_path,
    save_best_path,
    num_layers=32,
    batch_size=2048,
    lr=5e-4,
    weight_decay=1e-4,
    epochs=500,
    num_workers=4,
    valid_ratio=0.15,
    test_ratio_in_train=0.15,
    cosine_weight=1.0,
    early_stop_patience=5,
    device="cuda:0",
):
    dataset = InterLayerJsonlTensorDataset(jsonl_path)
    train_set, test_set, valid_set = split_dataset_sequential_valid(dataset, valid_ratio, test_ratio_in_train)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = LayerAwareResidualMLP(num_layers=num_layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            src_layer = batch["src_layer"].to(device)
            tgt_layer = batch["tgt_layer"].to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x, src_layer, tgt_layer)
            loss, _ = regression_loss(pred, y, cosine_weight=cosine_weight)
            loss.backward()
            optimizer.step()

        metrics = evaluate_model(model, test_loader, device, cosine_weight=cosine_weight)
        print(f"Epoch {epoch:03d} | test_loss={metrics['loss']:.6f}")
        if metrics["loss"] < best_loss - 1e-4:
            best_loss = metrics["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, save_best_path)
            no_improve = 0
            print(f"[Info] Best model saved to: {save_best_path}")
        else:
            no_improve += 1
            if no_improve >= early_stop_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    valid_metrics = evaluate_model(model, valid_loader, device, cosine_weight=cosine_weight)
    print("[Info] Final valid metrics:", valid_metrics)
    return model, valid_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_path", type=str, default="/data1/home/dataset_share/cd_data/llava-v1.5-7B/EarthVQA/final/attn_proj_mapping_64_project.jsonl")
    parser.add_argument("--save_best_path", type=str, default="./model/best_mapping_model.pt")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_layers", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    train_model(
        jsonl_path=args.jsonl_path,
        save_best_path=args.save_best_path,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        epochs=args.epochs,
        num_workers=args.num_workers,
        device=args.device,
    )


if __name__ == "__main__":
    main()
