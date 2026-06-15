import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

class Profiler:
    def __init__(self, model, proj_dim: Optional[int] = None, proj_method: str = "max", seed: int = 42):
        self.model = model

        self.handles = []
        self.current_step = 0
        self._step_hook_handle = None

        # 原始向量（最终存 CPU）
        # layer_idx -> list of step vectors, each vector shape [D]
        self.decode_attn_by_layer = defaultdict(list)

        # 降维后向量（最终存 CPU）
        # layer_idx -> list of step vectors, each vector shape [K]
        self.decode_attn_proj_by_layer = defaultdict(list)

        # 当前 step 的临时缓存（尽量保留在 GPU）
        # layer_idx -> Tensor[D]
        self._current_step_attn_gpu = {}

        self.proj_dim = proj_dim
        self.proj_method = proj_method  # "max", "min", "mean", "project"
        self.seed = seed

        # 投影矩阵（仅当 proj_method="project" 时使用）
        self.shared_proj_mat = None

        try:
            self.compute_device = next(model.parameters()).device
        except StopIteration:
            self.compute_device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

    # ================= public API =================

    def register(self):
        layers = self._get_layers("language_model")
        if layers is None:
            layers = self.model.model.layers

        num_layers = len(layers)

        # 如果使用投影降维，初始化投影矩阵
        if self.proj_method == "project":
            self._init_shared_orthogonal_mat(layers)

        for layer_idx in range(num_layers):
            h = layers[layer_idx].self_attn.o_proj.register_forward_hook(
                self._make_decode_layer_hook(layer_idx, branch="attn")
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
            self.decode_attn_by_layer.clear()
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

    def get_steps(self):
        return self.current_step

    # =========================
    # 1) 同一 step 下，相邻 layer 的统计
    # =========================
    def get_interlayer_transition(self, branch: str = "attn", eps: float = 1e-12):
        storage = self._get_branch_storage(branch)
        layer_ids = sorted(storage.keys())

        if len(layer_ids) < 2:
            return []

        num_steps = min(len(storage[layer_idx]) for layer_idx in layer_ids)
        results = []

        for step_idx in range(num_steps):
            step_result = {
                "layer_pairs": [],
                "cos_sim": [],
                "rmse": [],
                "diff_mean": [],
                "diff_std": [],
            }

            for i in range(len(layer_ids) - 1):
                l1 = layer_ids[i]
                l2 = layer_ids[i + 1]

                v1 = storage[l1][step_idx]
                v2 = storage[l2][step_idx]

                m = self._pairwise_metrics(v1, v2, eps=eps)

                step_result["layer_pairs"].append([l1, l2])
                step_result["cos_sim"].append(m["cos_sim"])
                step_result["rmse"].append(m["rmse"])
                step_result["diff_mean"].append(m["diff_mean"])
                step_result["diff_std"].append(m["diff_std"])

            results.append(step_result)

        return results

    # =========================
    # 2) 不同 step 下，相同 layer 的统计
    # =========================
    def get_layerwise_temporal_transition(self, branch: str = "attn", eps: float = 1e-12):
        storage = self._get_branch_storage(branch)
        result = {}

        for layer_idx in sorted(storage.keys()):
            vec_list = storage[layer_idx]
            layer_result = {
                "step_pairs": [],
                "cos_sim": [],
                "rmse": [],
                "diff_mean": [],
                "diff_std": [],
            }

            if len(vec_list) < 2:
                result[layer_idx] = layer_result
                continue

            for t in range(len(vec_list) - 1):
                v1 = vec_list[t]
                v2 = vec_list[t + 1]

                m = self._pairwise_metrics(v1, v2, eps=eps)

                layer_result["step_pairs"].append([t, t + 1])
                layer_result["cos_sim"].append(m["cos_sim"])
                layer_result["rmse"].append(m["rmse"])
                layer_result["diff_mean"].append(m["diff_mean"])
                layer_result["diff_std"].append(m["diff_std"])

            result[layer_idx] = layer_result

        return result

    # =========================
    # 3) 同一 step 下，各 layer 自身向量的统计
    # =========================
    def get_layerwise_vector_stats(self, branch: str = "attn", unbiased: bool = False):
        storage = self._get_branch_storage(branch)
        layer_ids = sorted(storage.keys())

        if len(layer_ids) == 0:
            return []

        num_steps = min(len(storage[layer_idx]) for layer_idx in layer_ids)
        results = []

        for step_idx in range(num_steps):
            step_result = {
                "layers": [],
                "mean": [],
                "std": [],
                "norm": [],
            }

            for layer_idx in layer_ids:
                v = storage[layer_idx][step_idx]
                m = self._vector_stats(v, unbiased=unbiased)

                step_result["layers"].append(layer_idx)
                step_result["mean"].append(m["mean"])
                step_result["std"].append(m["std"])
                step_result["norm"].append(m["norm"])

            results.append(step_result)

        return results

    # =========================
    # 4) 比较原始向量与降维后向量的统计
    # =========================
    def get_projected_vs_original_stats(self, unbiased: bool = False):
        orig_storage = self.decode_attn_by_layer
        proj_storage = self.decode_attn_proj_by_layer

        layer_ids = sorted(set(orig_storage.keys()) & set(proj_storage.keys()))
        if len(layer_ids) == 0:
            return []

        num_steps = min(
            min(len(orig_storage[layer_idx]) for layer_idx in layer_ids),
            min(len(proj_storage[layer_idx]) for layer_idx in layer_ids),
        )

        results = []

        for step_idx in range(num_steps):
            step_result = {
                "layers": [],
                "orig_mean": [],
                "orig_std": [],
                "orig_norm": [],
                "proj_mean": [],
                "proj_std": [],
                "proj_norm": [],
                "mean_abs_diff": [],
                "std_abs_diff": [],
                "norm_abs_diff": [],
            }

            for layer_idx in layer_ids:
                v_orig = orig_storage[layer_idx][step_idx]
                v_proj = proj_storage[layer_idx][step_idx]

                m_orig = self._vector_stats(v_orig, unbiased=unbiased)
                m_proj = self._vector_stats(v_proj, unbiased=unbiased)

                step_result["layers"].append(layer_idx)
                step_result["orig_mean"].append(m_orig["mean"])
                step_result["orig_std"].append(m_orig["std"])
                step_result["orig_norm"].append(m_orig["norm"])
                step_result["proj_mean"].append(m_proj["mean"])
                step_result["proj_std"].append(m_proj["std"])
                step_result["proj_norm"].append(m_proj["norm"])
                step_result["mean_abs_diff"].append(abs(m_orig["mean"] - m_proj["mean"]))
                step_result["std_abs_diff"].append(abs(m_orig["std"] - m_proj["std"]))
                step_result["norm_abs_diff"].append(abs(m_orig["norm"] - m_proj["norm"]))

            results.append(step_result)

        return results

    def save_attn_proj_interlayer_jsonl(self, file_path: str, sample_id: Optional[int] = None):
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

    def save_attn_proj_temporal_jsonl(self, file_path: str):
        """
        将降维后的 attn 向量保存为 temporal 监督样本 JSONL。
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        storage = self.decode_attn_proj_by_layer
        layer_ids = sorted(storage.keys())

        if len(layer_ids) == 0:
            raise ValueError("decode_attn_proj_by_layer is empty, nothing to save.")

        with path.open("a", encoding="utf-8") as f:
            for layer_idx in layer_ids:
                vec_list = storage[layer_idx]
                for t in range(len(vec_list) - 1):
                    x = vec_list[t]
                    y = vec_list[t + 1]

                    record = {
                        "layer": layer_idx,
                        "src_step": t,
                        "tgt_step": t + 1,
                        "x": x.detach().cpu().tolist(),
                        "y": y.detach().cpu().tolist(),
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _compare_pred_target_stats(self, pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12):
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
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        include_vectors: bool = True,
        max_steps: Optional[int] = None,
    ):
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
                    src_layer = torch.tensor([src_layer_id], dtype=torch.long, device=device)
                    tgt_layer = torch.tensor([tgt_layer_id], dtype=torch.long, device=device)
                    step = torch.tensor([step_idx], dtype=torch.long, device=device)

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

    def get_attn_proj_model_diff_vector_result(
        self,
        predictor_model,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        include_pred_target_vectors: bool = False,
        max_steps: Optional[int] = None,
    ):
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
                    src_layer = torch.tensor([src_layer_id], dtype=torch.long, device=device)
                    tgt_layer = torch.tensor([tgt_layer_id], dtype=torch.long, device=device)
                    step = torch.tensor([step_idx], dtype=torch.long, device=device)

                    try:
                        pred = predictor_model(x, src_layer, tgt_layer, step)
                    except TypeError:
                        pred = predictor_model(x, src_layer, tgt_layer)

                    pred_vec = pred[0].detach().float().cpu()
                    target_vec = target_vec.detach().float().cpu()
                    diff_vec = pred_vec - target_vec

                    record = {
                        "step": step_idx,
                        "src_layer": src_layer_id,
                        "tgt_layer": tgt_layer_id,
                        "diff_vec": diff_vec.tolist(),
                    }

                    if include_pred_target_vectors:
                        record["pred_vec"] = pred_vec.tolist()
                        record["target_vec"] = target_vec.tolist()

                    records.append(record)

        return {
            "num_steps": num_steps,
            "num_layer_pairs": max(len(layer_ids) - 1, 0),
            "records": records,
        }

    # ================= internal helpers =================

    def _get_branch_storage(self, branch: str):
        if branch == "attn":
            return self.decode_attn_by_layer
        elif branch == "attn_proj":
            return self.decode_attn_proj_by_layer
        else:
            raise ValueError("branch must be 'attn' or 'attn_proj'")

    def _get_layers(self, name: str = "language_model"):
        lm = getattr(self.model, name, None)
        if lm is None:
            inner = getattr(self.model, "model", None)
            lm = getattr(inner, name, None) if inner is not None else None
        if lm is None:
            return None

        if name == "language_model":
            return getattr(lm, "layers", None)
        return None

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
        def hook(module, input, output):
            # 当前 decode step 结束时，统一处理这个 step 的缓存
            if self.current_step > 0:
                self._finalize_current_step_projection()

            self.current_step += 1
            return output

        return hook

    def _make_decode_layer_hook(self, layer_idx: int, branch: str):
        def hook(module, inp, out):
            # skip prefill, only save decode-stage last-token vector
            if self.current_step == 0:
                return

            v = self._to_1d(out)
            if v is None:
                return

            if branch == "attn":
                # 当前 step 临时缓存，保留在当前设备
                self._current_step_attn_gpu[layer_idx] = v

                # 原始向量存 CPU
                self.decode_attn_by_layer[layer_idx].append(
                    v.detach().float().cpu().clone()
                )
            else:
                raise ValueError(f"Unknown branch: {branch}")

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

        # 存回 CPU
        y_cpu = y.detach().float().cpu()

        for i, layer_idx in enumerate(layer_ids):
            self.decode_attn_proj_by_layer[layer_idx].append(y_cpu[i].clone())

        self._current_step_attn_gpu.clear()

    def _sliding_max_1d(self, v: torch.Tensor, out_dim: Optional[int]):
        """
        对单个 1D 向量做滑动窗口最大值降维。
        v: [D]
        return: [K]
        """
        v = v.reshape(-1)
        n = v.shape[0]

        if out_dim is None or out_dim >= n:
            return v

        if out_dim <= 0:
            raise ValueError(f"out_dim must be positive, got {out_dim}")

        window_size = max(n // out_dim, 1)

        starts = torch.linspace(
            0,
            n - window_size,
            steps=out_dim,
            device=v.device
        ).long()

        chunks = [v[s:s + window_size].max() for s in starts]
        return torch.stack(chunks, dim=0)

    def _sliding_max_batch(self, x: torch.Tensor, out_dim: Optional[int]):
        """
        对 batch 向量做滑动窗口最大值降维。
        x: [B, D]
        return: [B, K]

        为了兼顾简单性和速度，这里按窗口切片后在 dim=-1 取 max。
        """
        if x.dim() != 2:
            raise ValueError(f"x must be 2D [B, D], got shape {tuple(x.shape)}")

        bsz, n = x.shape

        if out_dim is None or out_dim >= n:
            return x

        if out_dim <= 0:
            raise ValueError(f"out_dim must be positive, got {out_dim}")

        window_size = max(n // out_dim, 1)

        starts = torch.linspace(
            0,
            n - window_size,
            steps=out_dim,
            device=x.device
        ).long()

        pooled = []
        for s in starts:
            chunk = x[:, s:s + window_size]          # [B, W]
            pooled.append(chunk.max(dim=-1).values) # [B]

        return torch.stack(pooled, dim=-1)          # [B, K]

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

    def _project_1d(self, v: torch.Tensor, out_dim: Optional[int]):
        """
        使用正交投影矩阵降维
        v: [D]
        return: [K]
        """
        if self.shared_proj_mat is None:
            raise ValueError("Projection matrix not initialized. Call _init_shared_orthogonal_mat first.")
        
        v = v.reshape(-1)
        D = v.shape[0]
        
        if out_dim is None or out_dim >= D:
            return v
        
        # v [D] @ W [D, K] = [K]
        return v @ self.shared_proj_mat

    def _project_batch(self, x: torch.Tensor, out_dim: Optional[int]):
        """
        使用正交投影矩阵批量降维
        x: [B, D]
        return: [B, K]
        """
        if self.shared_proj_mat is None:
            raise ValueError("Projection matrix not initialized. Call _init_shared_orthogonal_mat first.")
        
        if x.dim() != 2:
            raise ValueError(f"x must be 2D [B, D], got shape {tuple(x.shape)}")
        
        B, D = x.shape
        
        if out_dim is None or out_dim >= D:
            return x
        
        # x [B, D] @ W [D, K] = [B, K]
        return x @ self.shared_proj_mat

    def _sliding_pooling_1d(self, v: torch.Tensor, out_dim: Optional[int], pool_type: str = "max"):
        """
        对单个 1D 向量做滑动窗口池化降维。
        v: [D]
        pool_type: "max", "min", "mean"
        return: [K]
        """
        v = v.reshape(-1)
        n = v.shape[0]

        if out_dim is None or out_dim >= n:
            return v

        if out_dim <= 0:
            raise ValueError(f"out_dim must be positive, got {out_dim}")

        window_size = max(n // out_dim, 1)

        starts = torch.linspace(
            0,
            n - window_size,
            steps=out_dim,
            device=v.device
        ).long()

        if pool_type == "max":
            chunks = [v[s:s + window_size].max() for s in starts]
        elif pool_type == "min":
            chunks = [v[s:s + window_size].min() for s in starts]
        elif pool_type == "mean":
            chunks = [v[s:s + window_size].mean() for s in starts]
        else:
            raise ValueError(f"pool_type must be 'max', 'min', or 'mean', got {pool_type}")
        
        return torch.stack(chunks, dim=0)

    def _sliding_pooling_batch(self, x: torch.Tensor, out_dim: Optional[int], pool_type: str = "max"):
        """
        对 batch 向量做滑动窗口池化降维。
        x: [B, D]
        pool_type: "max", "min", "mean"
        return: [B, K]
        """
        if x.dim() != 2:
            raise ValueError(f"x must be 2D [B, D], got shape {tuple(x.shape)}")

        B, n = x.shape

        if out_dim is None or out_dim >= n:
            return x

        if out_dim <= 0:
            raise ValueError(f"out_dim must be positive, got {out_dim}")

        window_size = max(n // out_dim, 1)

        starts = torch.linspace(
            0,
            n - window_size,
            steps=out_dim,
            device=x.device
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
                raise ValueError(f"pool_type must be 'max', 'min', or 'mean', got {pool_type}")

        return torch.stack(pooled, dim=-1)  # [B, K]

    def _reduce_dimension(self, x: torch.Tensor, out_dim: Optional[int]):
        """
        统一的降维接口，根据 self.proj_method 选择降维方式
        x: [B, D] 或 [D]
        return: [B, K] 或 [K]
        """
        is_batch = (x.dim() == 2)
        
        if self.proj_method == "project":
            if is_batch:
                return self._project_batch(x, out_dim)
            else:
                return self._project_1d(x, out_dim)
        elif self.proj_method in ["max", "min", "mean"]:
            if is_batch:
                return self._sliding_pooling_batch(x, out_dim, self.proj_method)
            else:
                return self._sliding_pooling_1d(x, out_dim, self.proj_method)
        else:
            raise ValueError(f"Unknown proj_method: {self.proj_method}. Must be 'max', 'min', 'mean', or 'project'")

    def _pairwise_metrics(self, v1: torch.Tensor, v2: torch.Tensor, eps: float = 1e-12):
        v1 = v1.detach().float().cpu().reshape(-1)
        v2 = v2.detach().float().cpu().reshape(-1)

        diff = v2 - v1

        if diff.numel() == 0:
            return {
                "cos_sim": None,
                "rmse": None,
                "diff_mean": None,
                "diff_std": None,
            }

        cos_sim = F.cosine_similarity(
            v1.unsqueeze(0), v2.unsqueeze(0), dim=-1, eps=eps
        ).item()

        rmse = torch.sqrt((diff ** 2).mean()).item()
        diff_mean = diff.mean().item()
        diff_std = diff.std(unbiased=False).item() if diff.numel() > 1 else 0.0

        return {
            "cos_sim": cos_sim,
            "rmse": rmse,
            "diff_mean": diff_mean,
            "diff_std": diff_std,
        }

    def _vector_stats(self, v: torch.Tensor, unbiased: bool = False):
        """
        对单个 1D 向量计算自身统计：
        - mean
        - std
        - norm
        """
        v = v.detach().float().cpu().reshape(-1)

        if v.numel() == 0:
            return {
                "mean": None,
                "std": None,
                "norm": None,
            }

        mean = v.mean().item()
        std = v.std(unbiased=unbiased).item() if v.numel() > 1 else 0.0
        norm = torch.norm(v, p=2).item()

        return {
            "mean": mean,
            "std": std,
            "norm": norm,
        }