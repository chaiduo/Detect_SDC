import torch
import torch.nn.functional as F
from collections import defaultdict, Counter
from typing import Any, Dict, Optional, List


class Profiler:
    def __init__(self, model):
        self.model = model

        self.handles = []
        self.current_step = 0
        self._step_hook_handle = None

        # 下面这些结构主要服务于“旧版多层 top-k 统计”
        # layer_idx -> rank(0/1/2) -> Counter(idx)
        self.topk_counter_mlp = defaultdict(lambda: defaultdict(Counter))
        self.total_count_mlp = defaultdict(lambda: defaultdict(int))

        self.topk_counter_attn = defaultdict(lambda: defaultdict(Counter))
        self.total_count_attn = defaultdict(lambda: defaultdict(int))

        # 保存每一步每个 rank 的 idx 序列
        self.topk_seq_mlp = defaultdict(lambda: defaultdict(list))
        self.topk_seq_attn = defaultdict(lambda: defaultdict(list))

        # 旧版：保存指定层 decode 阶段的向量序列（当前 register 中默认未启用）
        self.attn_o_vecs = defaultdict(list)
        self.mlp_down_vecs = defaultdict(list)

        # ================= prefill/decode projection =================
        # prefill 阶段保存整段输出矩阵 [T_prefill, D]
        self.prefill_merger = None
        self.prefill_mlp27_down_proj = None
        self.prefill_attn27_o_proj = None

        # decode 阶段每步只取“最后一个 token”的向量 [D]
        self.attn_list = []
        self.mlp_list = []

    # ================= public API =================

    def register(self):
        layers = self._get_layers("language_model")
        if layers is None:
            layers = self.model.model.layers

        # 旧版：对多层进行 top-k / vec 序列统计
        # 如果你后面需要恢复多层分析，可以把下面两段打开
        # for layer_idx in (25, 26, 27):
        #     h = layers[layer_idx].mlp.down_proj.register_forward_hook(
        #         self._make_hook(layer_idx, "mlp_down")
        #     )
        #     self.handles.append(h)

        # for layer_idx in (25, 26, 27):
        #     h = layers[layer_idx].self_attn.o_proj.register_forward_hook(
        #         self._make_hook(layer_idx, "attn_o")
        #     )
        #     self.handles.append(h)

        # ================= prefill/decode 专用 hook =================
        # merger：只在 prefill 阶段抓完整 visual token 矩阵
        merger = self._get_visual_merger()
        if merger is not None:
            h = merger.register_forward_hook(self._make_prefill_merger_hook())
            self.handles.append(h)

        # layer 27: prefill 保存整段矩阵；decode 保存每步最后 token 向量
        h = layers[27].mlp.down_proj.register_forward_hook(
            self._make_prefill_decode_mlp27_hook()
        )
        self.handles.append(h)

        h = layers[27].self_attn.o_proj.register_forward_hook(
            self._make_prefill_decode_attn27_hook()
        )
        self.handles.append(h)

        # 用 lm_head 的 forward 次数近似记录 decode step
        # current_step == 0 视为 prefill；>0 视为 decode
        self._step_hook_handle = self.model.lm_head.register_forward_hook(
            self._make_step_hook()
        )

    def reset(self, clear_stats: bool = False):
        self.current_step = 0
        if clear_stats:
            self.topk_counter_mlp.clear()
            self.total_count_mlp.clear()
            self.topk_counter_attn.clear()
            self.total_count_attn.clear()
            self.topk_seq_mlp.clear()
            self.topk_seq_attn.clear()
            self.attn_o_vecs.clear()
            self.mlp_down_vecs.clear()

            self.prefill_merger = None
            self.prefill_mlp27_down_proj = None
            self.prefill_attn27_o_proj = None

            self.attn_list.clear()
            self.mlp_list.clear()

    def unregister(self):
        for h in self.handles:
            h.remove()
        self.handles = []

        self.current_step = 0

        if self._step_hook_handle is not None:
            self._step_hook_handle.remove()
            self._step_hook_handle = None

    def get_steps(self):
        return self.current_step

    # ================= top-k stats =================

    def get_stats_mlp(self) -> Dict[str, Dict[int, Dict[str, List[Any]]]]:
        return self._compute_running_mode_stats(self.topk_seq_mlp)

    def get_stats_attn(self) -> Dict[str, Dict[int, Dict[str, List[Any]]]]:
        return self._compute_running_mode_stats(self.topk_seq_attn)

    # ================= existing vector stats =================

    def get_cos_sim(self, eps: float = 1e-12):
        results = self._compute_pairwise_metrics_between_dicts(
            self.attn_o_vecs,
            self.mlp_down_vecs,
            metrics=("cos", "dot"),
            eps=eps,
        )
        return results["cos"], results["dot"]

    def get_attn_mean_std(self, unbiased: bool = False):
        return self._compute_vecs_mean_std(self.attn_o_vecs, unbiased=unbiased)

    def get_mlp_down_mean_std(self, unbiased: bool = False):
        return self._compute_vecs_mean_std(self.mlp_down_vecs, unbiased=unbiased)

    # ================= adjacent metrics =================

    def get_mlp_down_adjacent_dot(self):
        return self._compute_adjacent_metrics(self.mlp_down_vecs, metrics=("dot",))["dot"]

    def get_attn_o_adjacent_dot(self):
        return self._compute_adjacent_metrics(self.attn_o_vecs, metrics=("dot",))["dot"]

    def get_mlp_down_adjacent_diff_mean(self):
        return self._compute_adjacent_metrics(self.mlp_down_vecs, metrics=("diff_mean",))["diff_mean"]

    def get_attn_o_adjacent_diff_mean(self):
        return self._compute_adjacent_metrics(self.attn_o_vecs, metrics=("diff_mean",))["diff_mean"]

    def get_mlp_down_adjacent_rmse(self):
        return self._compute_adjacent_metrics(self.mlp_down_vecs, metrics=("rmse",))["rmse"]

    def get_attn_o_adjacent_rmse(self):
        return self._compute_adjacent_metrics(self.attn_o_vecs, metrics=("rmse",))["rmse"]

    def get_attn_mlp_pair_rmse(self):
        return self._compute_pairwise_metrics_between_dicts(
            self.attn_o_vecs,
            self.mlp_down_vecs,
            metrics=("rmse",),
        )["rmse"]

    def get_adjacent_summary(self):
        return {
            "mlp_down": self._compute_adjacent_metrics(
                self.mlp_down_vecs,
                metrics=("dot", "diff_mean", "rmse"),
            ),
            "attn_o": self._compute_adjacent_metrics(
                self.attn_o_vecs,
                metrics=("dot", "diff_mean", "rmse"),
            ),
        }

    def get_attn_mlp_pair_summary(self, eps: float = 1e-12):
        return self._compute_pairwise_metrics_between_dicts(
            self.attn_o_vecs,
            self.mlp_down_vecs,
            metrics=("cos", "dot", "rmse"),
            eps=eps,
        )

    # ================= prefill/decode projection =================

    def get_prefill_decode_projection_stats(
        self,
        unbiased: bool = False,
        topk: int = 3,
        eps: float = 1e-12,
    ):
        # 这里比较的是：
        # decode 向量（attn_list / mlp_list）与 prefill 阶段矩阵（merger / o_proj / down_proj）的关系
        return {
            "attn_x_merger": self._matvec_stats(
                self.prefill_merger, self.attn_list, unbiased=unbiased, topk=topk, eps=eps
            ),
            "attn_x_o_proj": self._matvec_stats(
                self.prefill_attn27_o_proj, self.attn_list, unbiased=unbiased, topk=topk, eps=eps
            ),

            "mlp_x_merger": self._matvec_stats(
                self.prefill_merger, self.mlp_list, unbiased=unbiased, topk=topk, eps=eps
            ),
            "mlp_x_down_proj": self._matvec_stats(
                self.prefill_mlp27_down_proj, self.mlp_list, unbiased=unbiased, topk=topk, eps=eps
            ),

            "meta": {
                "prefill_merger_shape": list(self.prefill_merger.shape) if self.prefill_merger is not None else None,
                "prefill_mlp27_down_proj_shape": list(self.prefill_mlp27_down_proj.shape) if self.prefill_mlp27_down_proj is not None else None,
                "prefill_attn27_o_proj_shape": list(self.prefill_attn27_o_proj.shape) if self.prefill_attn27_o_proj is not None else None,
                "num_decode_attn": len(self.attn_list),
                "num_decode_mlp": len(self.mlp_list),
            }
        }

    # ================= decode dynamics =================

    def get_decode_dynamics_stats(self, eps: float = 1e-12):
        # 对 decode 阶段 layer27 的 attn/mlp 向量序列做动态统计
        return {
            "attn": self._vec_sequence_dynamics(self.attn_list, eps=eps),
            "mlp": self._vec_sequence_dynamics(self.mlp_list, eps=eps),
        }

    # ================= internal helpers =================

    def _topk_idx(self, v: torch.Tensor, k: int):
        # 加一个极小的 idx 偏置，避免相同值时 topk tie-breaking 不稳定
        eps = torch.finfo(v.dtype).eps
        idx_bias = torch.arange(v.numel(), dtype=v.dtype, device=v.device)
        v_adj = v - idx_bias * eps
        _, idx = torch.topk(v_adj, k=k, largest=True)
        return idx.tolist()

    def _get_layers(self, name: str = "language_model"):
        # 兼容 self.model.language_model 和 self.model.model.language_model 两种挂载方式
        lm = getattr(self.model, name, None)
        if lm is None:
            inner = getattr(self.model, "model", None)
            lm = getattr(inner, name, None) if inner is not None else None
        if lm is None:
            return None

        if name == "language_model":
            return getattr(lm, "layers", None)
        return None

    def _get_visual_merger(self):
        # 兼容 self.model.visual 和 self.model.model.visual
        visual = getattr(self.model, "visual", None)
        if visual is None:
            inner = getattr(self.model, "model", None)
            visual = getattr(inner, "visual", None) if inner is not None else None

        if visual is None:
            return None

        return getattr(visual, "merger", None)

    def _to_1d(self, out: torch.Tensor):
        # 统一把 hook 输出转成 1D 向量 [D]
        # - 3D: [B, T, D] -> 取 batch0 的最后 token
        # - 2D: [T, D] -> 取最后 token
        # - 1D: 直接返回
        if out is None:
            return None

        if isinstance(out, (tuple, list)):
            if len(out) == 0:
                return None
            out = out[0]

        out = out.detach().to(dtype=torch.float32, device="cpu")

        if out.dim() == 3:
            v = out[0, -1, :]
        elif out.dim() == 2:
            v = out[-1, :]
        elif out.dim() == 1:
            v = out
        else:
            return None

        return v.reshape(-1)

    def _to_2d(self, out: torch.Tensor):
        # 统一把 hook 输出转成矩阵 [T, D]
        # - 3D: [B, T, D] -> 取 batch0
        # - 2D: 直接返回
        # - 1D: 升成 [1, D]
        if out is None:
            return None

        if isinstance(out, (tuple, list)):
            if len(out) == 0:
                return None
            out = out[0]

        out = out.detach().to(dtype=torch.float32, device="cpu")

        if out.dim() == 3:
            mat = out[0, :, :]
        elif out.dim() == 2:
            mat = out
        elif out.dim() == 1:
            mat = out.unsqueeze(0)
        else:
            return None

        return mat

    def _extract_topk_idx(self, v: torch.Tensor, k: int = 3) -> Optional[list]:
        if v is None:
            return None
        k = min(k, v.numel())
        idx = self._topk_idx(v, k)
        return idx

    def _make_step_hook(self):
        # lm_head 每 forward 一次，就认为 decode 步数 +1
        def hook(module, input, output):
            self.current_step += 1
            return output
        return hook

    def _make_hook(self, layer_idx: int, layer_name: str):
        # 旧版多层 hook：只在 decode 阶段记录
        def hook(m, inp, out):
            if self.current_step == 0:
                return

            v = self._to_1d(out)
            if v is None:
                return

            topk = self._extract_topk_idx(v, k=3)
            if not topk:
                return

            if layer_name == "mlp_down":
                self.mlp_down_vecs[layer_idx].append(v.clone())
                for rank, idx in enumerate(topk):
                    self.topk_counter_mlp[layer_idx][rank][idx] += 1
                    self.total_count_mlp[layer_idx][rank] += 1
                    self.topk_seq_mlp[layer_idx][rank].append(int(idx))

            elif layer_name == "attn_o":
                self.attn_o_vecs[layer_idx].append(v.clone())
                for rank, idx in enumerate(topk):
                    self.topk_counter_attn[layer_idx][rank][idx] += 1
                    self.total_count_attn[layer_idx][rank] += 1
                    self.topk_seq_attn[layer_idx][rank].append(int(idx))

        return hook

    def _make_prefill_merger_hook(self):
        # 只在 prefill 阶段抓 merger 的完整输出矩阵
        def hook(module, inp, out):
            if self.current_step != 0:
                return

            mat = self._to_2d(out)
            if mat is None:
                return

            self.prefill_merger = mat.clone()

        return hook

    def _make_prefill_decode_mlp27_hook(self):
        # layer27 mlp.down_proj:
        # - prefill: 保存整段矩阵 [T_prefill, D]
        # - decode: 每步保存最后 token 向量 [D]
        def hook(module, inp, out):
            if self.current_step == 0:
                mat = self._to_2d(out)
                if mat is not None:
                    self.prefill_mlp27_down_proj = mat.clone()
            else:
                v = self._to_1d(out)
                if v is not None:
                    self.mlp_list.append(v.clone())

        return hook

    def _make_prefill_decode_attn27_hook(self):
        # layer27 self_attn.o_proj:
        # - prefill: 保存整段矩阵 [T_prefill, D]
        # - decode: 每步保存最后 token 向量 [D]
        def hook(module, inp, out):
            if self.current_step == 0:
                mat = self._to_2d(out)
                if mat is not None:
                    self.prefill_attn27_o_proj = mat.clone()
            else:
                v = self._to_1d(out)
                if v is not None:
                    self.attn_list.append(v.clone())

        return hook

    def _stack_vec_list(self, vec_list: List[torch.Tensor]) -> Optional[torch.Tensor]:
        # list[[D], [D], ...] -> [T, D]
        if not vec_list:
            return None

        out = []
        for v in vec_list:
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v)
            v = v.detach().float().cpu().reshape(-1)
            out.append(v)

        if len(out) == 0:
            return None

        return torch.stack(out, dim=0)

    def _compute_vecs_mean_std(self, vec_dict, unbiased: bool = False):
        # 对每个向量本身，计算其内部维度上的 mean/std
        stats = {}

        for key, vec_list in vec_dict.items():
            mean_list = []
            std_list = []

            for v in vec_list:
                if not isinstance(v, torch.Tensor):
                    v = torch.tensor(v)

                v = v.detach().float().cpu().reshape(-1)

                mean_list.append(v.mean().item())
                std_list.append(v.std(unbiased=unbiased).item())

            stats[key] = {
                "mean": mean_list,
                "std": std_list,
            }

        return stats

    def _compute_adjacent_metrics(self, vec_dict, metrics=("dot", "diff_mean", "rmse"), eps: float = 1e-12):
        # 对相邻时间步向量 v_t, v_{t+1} 计算指标
        results = {metric: {} for metric in metrics}

        for key, vec_list in vec_dict.items():
            mat = self._stack_vec_list(vec_list)
            if mat is None or mat.size(0) < 2:
                for metric in metrics:
                    results[metric][key] = []
                continue

            v1 = mat[:-1]
            v2 = mat[1:]
            diff = v2 - v1

            if "dot" in metrics:
                dot = (v1 * v2).sum(dim=-1)
                results["dot"][key] = dot.tolist()

            if "diff_mean" in metrics:
                diff_mean = diff.mean(dim=-1)
                results["diff_mean"][key] = diff_mean.tolist()

            if "rmse" in metrics:
                rmse = torch.sqrt((diff ** 2).mean(dim=-1))
                results["rmse"][key] = rmse.tolist()

            if "cos" in metrics:
                cos = F.cosine_similarity(v1, v2, dim=-1, eps=eps)
                results["cos"][key] = cos.tolist()

        return results

    def _compute_pairwise_metrics_between_dicts(
        self,
        dict1,
        dict2,
        metrics=("cos", "dot", "rmse"),
        eps: float = 1e-12,
    ):
        # 对两个 dict 中相同 key 的向量序列按时间步一一对应比较
        common_keys = set(dict1.keys()) & set(dict2.keys())
        if not common_keys:
            return {metric: {} for metric in metrics}

        results = {metric: {} for metric in metrics}

        for key in sorted(common_keys):
            list1 = dict1[key]
            list2 = dict2[key]

            assert len(list1) == len(list2), \
                f"Layer {key}: length mismatch ({len(list1)} vs {len(list2)})"

            mat1 = self._stack_vec_list(list1)
            mat2 = self._stack_vec_list(list2)

            if mat1 is None or mat2 is None:
                for metric in metrics:
                    results[metric][key] = []
                continue

            diff = mat2 - mat1

            if "cos" in metrics:
                cos = F.cosine_similarity(mat1, mat2, dim=-1, eps=eps)
                results["cos"][key] = cos.tolist()

            if "dot" in metrics:
                dot = (mat1 * mat2).sum(dim=-1)
                results["dot"][key] = dot.tolist()

            if "rmse" in metrics:
                rmse = torch.sqrt((diff ** 2).mean(dim=-1))
                results["rmse"][key] = rmse.tolist()

            if "diff_mean" in metrics:
                diff_mean = diff.mean(dim=-1)
                results["diff_mean"][key] = diff_mean.tolist()

        return results

    def _compute_running_mode_stats(
        self,
        seq_dict
    ) -> Dict[int, Dict[str, Dict[str, List[Any]]]]:
        # 对 top-k idx 序列做“运行众数”统计：
        # 看随着 step 增加，当前最常出现的 idx 是谁，以及其占比
        results: Dict[int, Dict[str, Dict[str, List[Any]]]] = {}

        for layer_idx, rank_map in seq_dict.items():
            results[layer_idx] = {}

            for rank, seq in rank_map.items():
                counter = Counter()
                mode_idx_list: List[int] = []
                mode_rate_list: List[float] = []

                current_mode: Optional[int] = None
                current_mode_count = 0

                for step, idx in enumerate(seq, start=1):
                    counter[idx] += 1
                    cnt = counter[idx]

                    if current_mode is None:
                        current_mode = idx
                        current_mode_count = cnt
                    else:
                        if cnt > current_mode_count:
                            current_mode = idx
                            current_mode_count = cnt
                        elif cnt == current_mode_count and idx < current_mode:
                            current_mode = idx
                            current_mode_count = cnt

                    mode_idx_list.append(int(current_mode))
                    mode_rate_list.append(current_mode_count / step)

                results[layer_idx][f"top{rank + 1}"] = {
                    "mode_idx": mode_idx_list,
                    "mode_rate": mode_rate_list,
                }

        return results

    def _scalar_seq_summary(self, x, unbiased: bool = False, early_k: int = 3):
        # 把一条标量序列压成一组聚合统计
        if x is None:
            x = []
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.detach().float().cpu().reshape(-1)

        if x.numel() == 0:
            return {
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "first": None,
                "last": None,
                "early_mean": None,
                "late_mean": None,
                "slope": None,
            }

        n = x.numel()
        k = min(early_k, n)

        out = {
            "mean": x.mean().item(),
            "std": x.std(unbiased=unbiased).item() if n > 1 else 0.0,
            "min": x.min().item(),
            "max": x.max().item(),
            "first": x[0].item(),
            "last": x[-1].item(),
            "early_mean": x[:k].mean().item(),
            "late_mean": x[-k:].mean().item(),
            "slope": 0.0,
        }

        if n >= 2:
            t = torch.arange(n, dtype=torch.float32)
            t_mean = t.mean()
            x_mean = x.mean()
            denom = ((t - t_mean) ** 2).sum()
            if denom.item() > 0:
                slope = (((t - t_mean) * (x - x_mean)).sum() / denom).item()
                out["slope"] = slope

        return out

    def _aggregate_feature_dict(self, feat_dict: Dict[str, List[Any]], early_k: int = 3):
        # 对 dict 中每条 list 型特征做 summary
        out = {}
        for k, v in feat_dict.items():
            if isinstance(v, list):
                try:
                    summary = self._scalar_seq_summary(v, early_k=early_k)
                    for sk, sv in summary.items():
                        out[f"{k}_{sk}"] = sv
                except Exception:
                    pass
        return out

    def _vec_summary_stats(self, x: torch.Tensor, unbiased: bool = False):
        # 对单个向量做 max/min/mean/std 统计
        x = x.detach().float().cpu().reshape(-1)

        if x.numel() == 0:
            return {
                "max": None,
                "max_pos": None,
                "min": None,
                "min_pos": None,
                "mean": None,
                "std": None,
            }

        max_val, max_idx = torch.max(x, dim=0)
        min_val, min_idx = torch.min(x, dim=0)

        return {
            "max": max_val.item(),
            "max_pos": int(max_idx.item()),
            "min": min_val.item(),
            "min_pos": int(min_idx.item()),
            "mean": x.mean().item(),
            "std": x.std(unbiased=unbiased).item() if x.numel() > 1 else 0.0,
        }

    def _topk_mean(self, x: torch.Tensor, k: int = 3):
        # 取向量中最大的 k 个值的平均
        x = x.detach().float().cpu().reshape(-1)
        if x.numel() == 0:
            return None
        k = min(k, x.numel())
        vals, _ = torch.topk(x, k=k, largest=True)
        return vals.mean().item()

    def _softmax_entropy(self, x: torch.Tensor, dim: int = -1, eps: float = 1e-12):
        # 对一维 score 分布做 softmax 后计算熵
        p = torch.softmax(x, dim=dim)
        ent = -(p * torch.log(p.clamp_min(eps))).sum(dim=dim)
        return ent

    def _cosine_with_mean_vec(self, mat: torch.Tensor, v: torch.Tensor, eps: float = 1e-12):
        # v 与 mat 的行均值向量做 cosine
        mean_vec = mat.mean(dim=0)
        mean_vec = mean_vec.reshape(1, -1)
        v = v.reshape(1, -1)
        return F.cosine_similarity(mean_vec, v, dim=-1, eps=eps).item()

    def _subspace_projection_stats(self, mat: torch.Tensor, v: torch.Tensor):
        # 把 v 投影到 mat 张成的子空间，返回投影与残差大小
        mat = mat.detach().float().cpu()
        v = v.detach().float().cpu().reshape(-1)

        if mat.dim() != 2:
            raise ValueError(f"mat must be 2D, got {tuple(mat.shape)}")

        try:
            U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
        except RuntimeError:
            return {
                "proj_norm": None,
                "residual_norm": None,
                "residual_ratio": None,
            }

        if S.numel() == 0:
            return {
                "proj_norm": 0.0,
                "residual_norm": torch.norm(v).item(),
                "residual_ratio": 1.0,
            }

        tol = max(mat.shape) * torch.finfo(S.dtype).eps * S.max()
        rank = int((S > tol).sum().item())

        if rank == 0:
            proj = torch.zeros_like(v)
        else:
            Vr = Vh[:rank, :].T
            proj = Vr @ (Vr.T @ v)

        residual = v - proj
        v_norm = torch.norm(v).item()
        proj_norm = torch.norm(proj).item()
        residual_norm = torch.norm(residual).item()
        residual_ratio = residual_norm / (v_norm + 1e-12)

        return {
            "proj_norm": proj_norm,
            "residual_norm": residual_norm,
            "residual_ratio": residual_ratio,
        }

    def _matvec_stats(
        self,
        mat: torch.Tensor,
        vec_list: List[torch.Tensor],
        unbiased: bool = False,
        topk: int = 3,
        eps: float = 1e-12,
    ):
        # 对每个向量 v，计算它与矩阵 mat 中每一行的相似度/得分分布
        # 这里 score 使用归一化后的点积，近似 cosine，相比原始 mat @ v 更不容易数值饱和
        result = {
            "max": [],
            "max_pos": [],
            "min": [],
            "min_pos": [],
            "mean": [],
            "std": [],
            "topk_mean": [],
            "entropy": [],
            "score_l2": [],
            "cosine_with_mat_mean": [],
        }

        if mat is None or len(vec_list) == 0:
            return result

        if not isinstance(mat, torch.Tensor):
            mat = torch.tensor(mat)
        mat = mat.detach().float().cpu()

        if mat.dim() != 2:
            raise ValueError(f"mat must be 2D, got shape={tuple(mat.shape)}")

        D = mat.size(1)

        # 先把 mat 的每一行归一化，避免在循环里重复算
        mat_norm = F.normalize(mat, dim=-1)

        for v in vec_list:
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v)

            v = v.detach().float().cpu().reshape(-1)

            if v.numel() != D:
                raise ValueError(
                    f"dimension mismatch: mat.shape={tuple(mat.shape)}, vec.shape={tuple(v.shape)}"
                )

            # 归一化后的 score 范围更稳定，entropy 也更有可读性
            v_norm = F.normalize(v.unsqueeze(0), dim=-1).squeeze(0)
            score = mat_norm @ v_norm

            base_stats = self._vec_summary_stats(score, unbiased=unbiased)
            ent = self._softmax_entropy(score, dim=0, eps=eps).item()

            result["max"].append(base_stats["max"])
            result["max_pos"].append(base_stats["max_pos"])
            result["min"].append(base_stats["min"])
            result["min_pos"].append(base_stats["min_pos"])
            result["mean"].append(base_stats["mean"])
            result["std"].append(base_stats["std"])
            result["topk_mean"].append(self._topk_mean(score, k=topk))
            result["entropy"].append(ent)
            result["score_l2"].append(torch.norm(score).item())
            result["cosine_with_mat_mean"].append(self._cosine_with_mean_vec(mat, v, eps=eps))

        return result

    def _vec_sequence_dynamics(self, vec_list: List[torch.Tensor], eps: float = 1e-12):
        # 对向量时间序列 [v1, v2, ...] 做动态统计
        # - mean_seq/std_seq: 每个向量自身内部维度上的均值/标准差
        # - norm_seq: 每个向量的 L2 范数
        # - adjacent_*: 相邻向量之间的变化
        # - second_order_l2_seq: 二阶差分强度
        result = {
            "mean_seq": [],
            "std_seq": [],
            "norm_seq": [],
            "adjacent_l2_seq": [],
            "adjacent_l1_seq": [],
            "adjacent_cos_seq": [],
            "second_order_l2_seq": [],
        }

        mat = self._stack_vec_list(vec_list)
        if mat is None or mat.size(0) == 0:
            return result

        mean_seq = mat.mean(dim=-1)
        std_seq = mat.std(dim=-1, unbiased=False)
        norm_seq = torch.norm(mat, dim=-1)

        result["mean_seq"] = mean_seq.tolist()
        result["std_seq"] = std_seq.tolist()
        result["norm_seq"] = norm_seq.tolist()

        if mat.size(0) >= 2:
            v1 = mat[:-1]
            v2 = mat[1:]
            diff = v2 - v1

            adjacent_l2 = torch.norm(diff, dim=-1)
            adjacent_l1 = diff.abs().sum(dim=-1)
            adjacent_cos = F.cosine_similarity(v1, v2, dim=-1, eps=eps)

            result["adjacent_l2_seq"] = adjacent_l2.tolist()
            result["adjacent_l1_seq"] = adjacent_l1.tolist()
            result["adjacent_cos_seq"] = adjacent_cos.tolist()

        if mat.size(0) >= 3:
            d1 = mat[1:] - mat[:-1]
            d2 = d1[1:] - d1[:-1]
            second_order_l2 = torch.norm(d2, dim=-1)
            result["second_order_l2_seq"] = second_order_l2.tolist()

        return result
