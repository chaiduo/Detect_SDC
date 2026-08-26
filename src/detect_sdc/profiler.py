import json
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

ProjectionObserver = Callable[
    [torch.Tensor, torch.Tensor, tuple[int, ...]],
    None,
]


class Profiler:
    def __init__(
        self,
        model,
        proj_dim: Optional[int] = None,
        proj_method: str = "project",
        seed: int = 42,
        projection_observer: Optional[ProjectionObserver] = None,
    ):
        self.model = model

        self.handles = []
        self.current_step = 0
        self._step_hook_handle = None

        # 降维后向量（最终存 CPU）
        # layer_idx -> list of step vectors, each vector shape [K]
        self.decode_attn_proj_by_layer = defaultdict(list)

        # 当前 step 的临时缓存（尽量保留在 GPU）
        # layer_idx -> Tensor[D]
        self._current_step_attn_gpu = {}

        self.proj_dim = proj_dim
        self.proj_method = proj_method  # "max", "min", "mean", "project"
        self.seed = seed
        self.projection_observer = projection_observer

        # 投影矩阵（仅当 proj_method="project" 时使用）
        self.shared_proj_mat = None

        try:
            self.compute_device = next(model.parameters()).device
        except StopIteration:
            self.compute_device = torch.device("cpu")

    # ================= public API =================

    def register(self):
        layers = self._get_layers()
        if layers is None:
            layers = self.model.model.layers

        num_layers = len(layers)

        # 如果使用投影降维，初始化投影矩阵
        if self.proj_method == "project":
            self._init_shared_orthogonal_mat(layers)

        for layer_idx in range(num_layers):
            h = layers[layer_idx].self_attn.o_proj.register_forward_hook(
                self._make_decode_layer_hook(layer_idx)
            )
            self.handles.append(h)

        # lm_head forward 次数作为 decode step 计数器
        # 同时把它作为“当前 step 结束”的边界
        self._step_hook_handle = self.model.lm_head.register_forward_hook(
            self._make_step_hook()
        )

    def reset(self, clear_stats: bool = False):
        self.current_step = 0
        self._current_step_attn_gpu.clear()

        if clear_stats:
            self.decode_attn_proj_by_layer.clear()

    def unregister(self):
        # 先 flush 最后一步，避免遗漏
        self.finalize()

        for h in self.handles:
            h.remove()
        self.handles = []

        if self._step_hook_handle is not None:
            self._step_hook_handle.remove()
            self._step_hook_handle = None

        self.current_step = 0
        self._current_step_attn_gpu.clear()

    def finalize(self):
        """
        generation 结束后手动调用一次，
        防止最后一个 step 的缓存尚未写入。
        """
        if self.current_step > 0 and self._current_step_attn_gpu:
            self._finalize_current_step_projection()

    def save_attn_proj_interlayer_jsonl(
        self,
        file_path: str,
        sample_id: Optional[int] = None,
    ) -> int:
        """
        将降维后的 attn 向量保存为相邻层监督样本 JSONL。
        
        Args:
            file_path: 输出文件路径
            sample_id: 样本 ID，可选。如果提供，会保存到每条记录中
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        storage = self.decode_attn_proj_by_layer
        layer_ids = sorted(storage.keys())

        if len(layer_ids) < 2:
            raise ValueError("Need at least 2 layers in decode_attn_proj_by_layer.")

        num_steps = min(len(storage[layer_idx]) for layer_idx in layer_ids)

        rows_written = 0
        with path.open("a", encoding="utf-8") as f:
            for step_idx in range(num_steps):
                for i in range(len(layer_ids) - 1):
                    src_layer = layer_ids[i]
                    tgt_layer = layer_ids[i + 1]

                    x = storage[src_layer][step_idx]
                    y = storage[tgt_layer][step_idx]

                    record = {
                        "sample_id": sample_id,
                        "step": step_idx,
                        "src_layer": src_layer,
                        "tgt_layer": tgt_layer,
                        "x": x.detach().cpu().tolist(),
                        "y": y.detach().cpu().tolist(),
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    rows_written += 1
        return rows_written

    def _compare_pred_target_stats(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        eps: float = 1e-12,
    ):
        pred = pred.detach().float().cpu().reshape(-1)
        target = target.detach().float().cpu().reshape(-1)

        pred_mean = pred.mean().item()
        target_mean = target.mean().item()
        mean_diff = pred_mean - target_mean

        pred_std = pred.std(unbiased=False).item() if pred.numel() > 1 else 0.0
        target_std = target.std(unbiased=False).item() if target.numel() > 1 else 0.0
        std_diff = pred_std - target_std

        cos_sim = F.cosine_similarity(
            pred.unsqueeze(0),
            target.unsqueeze(0),
            dim=-1,
            eps=eps,
        ).item()

        diff = pred - target

        l2_distance = torch.norm(diff, p=2).item()
        l1_distance = torch.norm(diff, p=1).item()
        chebyshev_distance = torch.norm(diff, p=float("inf")).item()

        return {
            "pred_mean": pred_mean,
            "target_mean": target_mean,
            "mean_diff": mean_diff,
            "pred_std": pred_std,
            "target_std": target_std,
            "std_diff": std_diff,
            "cos_sim": cos_sim,
            "l2_distance": l2_distance,
            "l1_distance": l1_distance,
            "chebyshev_distance": chebyshev_distance,
        }

    def get_attn_proj_model_compare_result(
        self,
        predictor_model,
        device: str | None = None,
        include_vectors: bool = True,
        max_steps: Optional[int] = None,
    ):
        if device is None:
            device = str(self.compute_device)
        storage = self.decode_attn_proj_by_layer
        layer_ids = sorted(storage.keys())

        if len(layer_ids) < 2:
            return {
                "num_steps": 0,
                "num_layer_pairs": 0,
                "records": [],
            }

        num_steps = min(len(storage[layer_idx]) for layer_idx in layer_ids)
        if max_steps is not None:
            num_steps = min(num_steps, max_steps)

        predictor_model.eval()
        predictor_model.to(device)

        records = []

        with torch.no_grad():
            for step_idx in range(num_steps):
                for i in range(len(layer_ids) - 1):
                    src_layer_id = layer_ids[i]
                    tgt_layer_id = layer_ids[i + 1]

                    x_vec = storage[src_layer_id][step_idx]
                    target_vec = storage[tgt_layer_id][step_idx]

                    x = x_vec.unsqueeze(0).to(device=device, dtype=torch.float32)
                    src_layer = torch.tensor(
                        [src_layer_id],
                        dtype=torch.long,
                        device=device,
                    )
                    tgt_layer = torch.tensor(
                        [tgt_layer_id],
                        dtype=torch.long,
                        device=device,
                    )
                    step = torch.tensor(
                        [step_idx],
                        dtype=torch.long,
                        device=device,
                    )

                    try:
                        pred = predictor_model(x, src_layer, tgt_layer, step)
                    except TypeError:
                        pred = predictor_model(x, src_layer, tgt_layer)

                    pred_vec = pred[0].detach().float().cpu()
                    target_vec = target_vec.detach().float().cpu()

                    stats = self._compare_pred_target_stats(pred_vec, target_vec)

                    record = {
                        "step": step_idx,
                        "src_layer": src_layer_id,
                        "tgt_layer": tgt_layer_id,
                        **stats,
                    }

                    if include_vectors:
                        record["x"] = x_vec.detach().cpu().tolist()
                        record["pred"] = pred_vec.tolist()
                        record["target"] = target_vec.tolist()

                    records.append(record)

        return {
            "num_steps": num_steps,
            "num_layer_pairs": max(len(layer_ids) - 1, 0),
            "records": records,
        }

    # ================= internal helpers =================

    def _get_layers(self):
        lm = getattr(self.model, "language_model", None)
        if lm is None:
            inner = getattr(self.model, "model", None)
            lm = (
                getattr(inner, "language_model", None)
                if inner is not None
                else None
            )
        if lm is None:
            return None
        return getattr(lm, "layers", None)

    def _to_1d(self, out: torch.Tensor):
        """
        保持当前 device，不在这里强制搬到 CPU。
        """
        if out is None:
            return None

        if isinstance(out, (tuple, list)):
            if len(out) == 0:
                return None
            out = out[0]

        out = out.detach().to(dtype=torch.float32)

        if out.dim() == 3:
            v = out[0, -1, :]
        elif out.dim() == 2:
            v = out[-1, :]
        elif out.dim() == 1:
            v = out
        else:
            return None

        return v.reshape(-1)

    def _make_step_hook(self):
        def hook(_module, _input, output):
            # 当前 decode step 结束时，统一处理这个 step 的缓存
            if self.current_step > 0:
                self._finalize_current_step_projection()

            self.current_step += 1
            return output

        return hook

    def _make_decode_layer_hook(self, layer_idx: int):
        def hook(_module, _input, out):
            # skip prefill, only save decode-stage last-token vector
            if self.current_step == 0:
                return

            v = self._to_1d(out)
            if v is None:
                return

            # 当前 step 临时缓存，保留在当前设备
            self._current_step_attn_gpu[layer_idx] = v

        return hook

    def _finalize_current_step_projection(self):
        """
        在一个 decode step 结束时，统一把该 step 的各层向量
        根据 self.proj_method 做降维。
        """
        if not self._current_step_attn_gpu:
            return

        layer_ids = sorted(self._current_step_attn_gpu.keys())
        vecs = [self._current_step_attn_gpu[layer_idx] for layer_idx in layer_ids]

        # [L, D]
        x = torch.stack(vecs, dim=0)

        # 批量降维 [L, D] -> [L, K]
        y = self._reduce_dimension(x, self.proj_dim)

        if self.projection_observer is not None:
            self.projection_observer(
                x.detach(),
                y.detach(),
                tuple(layer_ids),
            )

        # 存回 CPU
        y_cpu = y.detach().float().cpu()

        for i, layer_idx in enumerate(layer_ids):
            self.decode_attn_proj_by_layer[layer_idx].append(y_cpu[i].clone())

        self._current_step_attn_gpu.clear()

    def _init_shared_orthogonal_mat(self, layers):
        """
        初始化共享的正交投影矩阵 [D, K]
        """
        if self.proj_dim is None:
            return

        # 从第一层获取输入维度 D
        first_layer = layers[0]
        with torch.no_grad():
            # 通过 hook 或其他方式获取维度，这里假设 D 为 4096（Llama-7B 的隐藏层维度）
            # 实际使用时可以根据模型自动推断
            D = first_layer.self_attn.o_proj.in_features
        
        # 生成随机正交矩阵 (QR 分解)
        torch.manual_seed(self.seed)
        random_mat = torch.randn(D, self.proj_dim, device=self.compute_device)
        Q, _ = torch.linalg.qr(random_mat)
        self.shared_proj_mat = Q  # [D, K]

    def _project_batch(self, x: torch.Tensor, out_dim: Optional[int]):
        """
        使用正交投影矩阵批量降维
        x: [B, D]
        return: [B, K]
        """
        if self.shared_proj_mat is None:
            raise ValueError(
                "Projection matrix not initialized. "
                "Call _init_shared_orthogonal_mat first."
            )

        if x.dim() != 2:
            raise ValueError(f"x must be 2D [B, D], got shape {tuple(x.shape)}")

        input_dim = x.shape[1]
        if out_dim is None or out_dim >= input_dim:
            return x

        # x [B, D] @ W [D, K] = [B, K]
        return x @ self.shared_proj_mat

    def _sliding_pooling_batch(
        self,
        x: torch.Tensor,
        out_dim: Optional[int],
        pool_type: str = "max",
    ):
        """
        对 batch 向量做滑动窗口池化降维。
        x: [B, D]
        pool_type: "max", "min", "mean"
        return: [B, K]
        """
        if x.dim() != 2:
            raise ValueError(f"x must be 2D [B, D], got shape {tuple(x.shape)}")

        input_dim = x.shape[1]

        if out_dim is None or out_dim >= input_dim:
            return x

        if out_dim <= 0:
            raise ValueError(f"out_dim must be positive, got {out_dim}")

        window_size = max(input_dim // out_dim, 1)

        starts = torch.linspace(
            0,
            input_dim - window_size,
            steps=out_dim,
            device=x.device,
        ).long()

        pooled = []
        for s in starts:
            chunk = x[:, s:s + window_size]  # [B, W]
            if pool_type == "max":
                pooled.append(chunk.max(dim=-1).values)
            elif pool_type == "min":
                pooled.append(chunk.min(dim=-1).values)
            elif pool_type == "mean":
                pooled.append(chunk.mean(dim=-1))
            else:
                raise ValueError(
                    "pool_type must be 'max', 'min', or 'mean', "
                    f"got {pool_type}"
                )

        return torch.stack(pooled, dim=-1)  # [B, K]

    def _reduce_dimension(self, x: torch.Tensor, out_dim: Optional[int]):
        """
        统一的降维接口，根据 self.proj_method 选择降维方式
        x: [B, D]
        return: [B, K]
        """
        if x.dim() != 2:
            raise ValueError(f"x must be 2D [B, D], got shape {tuple(x.shape)}")

        if self.proj_method == "project":
            return self._project_batch(x, out_dim)
        elif self.proj_method in ["max", "min", "mean"]:
            return self._sliding_pooling_batch(x, out_dim, self.proj_method)
        raise ValueError(
            f"Unknown proj_method: {self.proj_method}. "
            "Must be 'max', 'min', 'mean', or 'project'"
        )
