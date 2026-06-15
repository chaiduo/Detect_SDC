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

        # layer_idx -> rank(0/1/2) -> Counter(idx)
        self.topk_counter_mlp = defaultdict(lambda: defaultdict(Counter))
        self.total_count_mlp = defaultdict(lambda: defaultdict(int))

        self.topk_counter_attn = defaultdict(lambda: defaultdict(Counter))
        self.total_count_attn = defaultdict(lambda: defaultdict(int))

        # 保存每一步每个 rank 的 idx 序列
        self.topk_seq_mlp = defaultdict(lambda: defaultdict(list))
        self.topk_seq_attn = defaultdict(lambda: defaultdict(list))

        self.attn_o_vecs = defaultdict(list)
        self.mlp_down_vecs = defaultdict(list)

        # ================= 新增: prefill/decode projection 分析 =================
        self.prefill_A = None   # merger 输出矩阵 [T_prefill, D]
        self.prefill_B = None   # layer27 mlp.down_proj 在 prefill 的输出矩阵 [T_prefill, D]

        self.attn_list = []     # decode阶段 layer27 attn_o 每步最后token向量 [D]
        self.mlp_list = []      # decode阶段 layer27 mlp_down 每步最后token向量 [D]

    # ================= public API =================

    def register(self):
        layers = self._get_layers("language_model")
        if layers is None:
            layers = self.model.model.layers

        # for layer_idx in (23, 24, 25, 26, 27):
        #     h = layers[layer_idx].mlp.down_proj.register_forward_hook(
        #         self._make_hook(layer_idx, "mlp_down")
        #     )
        #     self.handles.append(h)

        # for layer_idx in (23, 24, 25, 26, 27):
        #     h = layers[layer_idx].self_attn.o_proj.register_forward_hook(
        #         self._make_hook(layer_idx, "attn_o")
        #     )
        #     self.handles.append(h)

        # ================= 新增: prefill/decode 专用 hook =================
        merger = self._get_visual_merger()
        if merger is not None:
            h = merger.register_forward_hook(self._make_prefill_A_hook())
            self.handles.append(h)

        h = layers[27].mlp.down_proj.register_forward_hook(
            self._make_prefill_decode_mlp27_hook()
        )
        self.handles.append(h)

        h = layers[27].self_attn.o_proj.register_forward_hook(
            self._make_decode_attn27_hook()
        )
        self.handles.append(h)

        # 用 lm_head forward 作为 step 计数
        self._step_hook_handle = self.model.lm_head.register_forward_hook(
            self._make_step_counter_hook()
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

            self.prefill_A = None
            self.prefill_B = None
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
        """
        返回每个 layer 的滑动统计结果：
        - mode_idx: 到当前步为止出现最多的 idx
        - mode_rate: 该 idx 到当前步为止的占比
        """
        return self._compute_running_mode_stats(self.topk_seq_mlp)

    def get_stats_attn(self) -> Dict[str, Dict[int, Dict[str, List[Any]]]]:
        """
        返回每个 layer 的滑动统计结果：
        - mode_idx: 到当前步为止出现最多的 idx
        - mode_rate: 该 idx 到当前步为止的占比
        """
        return self._compute_running_mode_stats(self.topk_seq_attn)

    # ================= existing vector stats =================

    def get_cos_sim(self, eps: float = 1e-12):
        """
        计算 attn_o_vecs 与 mlp_down_vecs 在相同 layer 下、对应位置向量的：
        1. cosine similarity
        2. dot product
        """
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

    # ================= new APIs: adjacent metrics =================

    def get_mlp_down_adjacent_dot(self):
        """
        计算 mlp_down_vecs 中相邻两项的点积
        返回: {layer_id: [dot(v0,v1), dot(v1,v2), ...]}
        """
        return self._compute_adjacent_metrics(self.mlp_down_vecs, metrics=("dot",))["dot"]

    def get_attn_o_adjacent_dot(self):
        """
        计算 attn_o_vecs 中相邻两项的点积
        返回: {layer_id: [dot(v0,v1), dot(v1,v2), ...]}
        """
        return self._compute_adjacent_metrics(self.attn_o_vecs, metrics=("dot",))["dot"]

    def get_mlp_down_adjacent_diff_mean(self):
        """
        计算 mlp_down_vecs 中相邻两项差向量的均值
        返回: {layer_id: [mean(v1-v0), mean(v2-v1), ...]}
        """
        return self._compute_adjacent_metrics(self.mlp_down_vecs, metrics=("diff_mean",))["diff_mean"]

    def get_attn_o_adjacent_diff_mean(self):
        """
        计算 attn_o_vecs 中相邻两项差向量的均值
        返回: {layer_id: [mean(v1-v0), mean(v2-v1), ...]}
        """
        return self._compute_adjacent_metrics(self.attn_o_vecs, metrics=("diff_mean",))["diff_mean"]

    def get_mlp_down_adjacent_rmse(self):
        """
        计算 mlp_down_vecs 中相邻两项的 RMSE
        返回: {layer_id: [rmse(v0,v1), rmse(v1,v2), ...]}
        """
        return self._compute_adjacent_metrics(self.mlp_down_vecs, metrics=("rmse",))["rmse"]

    def get_attn_o_adjacent_rmse(self):
        """
        计算 attn_o_vecs 中相邻两项的 RMSE
        返回: {layer_id: [rmse(v0,v1), rmse(v1,v2), ...]}
        """
        return self._compute_adjacent_metrics(self.attn_o_vecs, metrics=("rmse",))["rmse"]

    def get_attn_mlp_pair_rmse(self):
        """
        计算 attn_o_vecs 和 mlp_down_vecs 中对应项的 RMSE
        返回: {layer_id: [rmse(attn[0], mlp[0]), rmse(attn[1], mlp[1]), ...]}
        """
        return self._compute_pairwise_metrics_between_dicts(
            self.attn_o_vecs,
            self.mlp_down_vecs,
            metrics=("rmse",),
        )["rmse"]

    # ================= optional aggregated APIs =================

    def get_adjacent_summary(self):
        """
        一次性返回两类向量的相邻项统计
        """
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
        """
        一次性返回 attn_o 与 mlp_down 对应项之间的统计
        """
        return self._compute_pairwise_metrics_between_dicts(
            self.attn_o_vecs,
            self.mlp_down_vecs,
            metrics=("cos", "dot", "rmse"),
            eps=eps,
        )

    # ================= 新增 public API: prefill/decode projection =================

    def get_prefill_decode_projection_stats(self, unbiased: bool = False):
        """
        返回四组统计:
        - attn_list x A
        - attn_list x B
        - mlp_list  x A
        - mlp_list  x B

        其中:
        A = prefill 阶段 merger 输出矩阵 [T_prefill, D]
        B = prefill 阶段 layer27 mlp.down_proj 输出矩阵 [T_prefill, D]
        """
        return {
            "attn_x_A": self._matvec_stats(self.prefill_A, self.attn_list, unbiased=unbiased),
            "attn_x_B": self._matvec_stats(self.prefill_B, self.attn_list, unbiased=unbiased),
            "mlp_x_A": self._matvec_stats(self.prefill_A, self.mlp_list, unbiased=unbiased),
            "mlp_x_B": self._matvec_stats(self.prefill_B, self.mlp_list, unbiased=unbiased),
            "meta": {
                "prefill_A_shape": list(self.prefill_A.shape) if self.prefill_A is not None else None,
                "prefill_B_shape": list(self.prefill_B.shape) if self.prefill_B is not None else None,
                "num_decode_attn": len(self.attn_list),
                "num_decode_mlp": len(self.mlp_list),
            }
        }

    # ================= internal helpers =================

    def _make_step_counter_hook(self):
        def hook(module, input, output):
            self.current_step += 1
            return output
        return hook

    def _topk_idx(self, v: torch.Tensor, k: int):
        """
        返回稳定 topk 的 idx(list[int])
        数值相同则下标小的优先
        """
        eps = torch.finfo(v.dtype).eps
        idx_bias = torch.arange(v.numel(), dtype=v.dtype, device=v.device)

        # 数值相同时，较小下标优先
        v_adj = v - idx_bias * eps
        _, idx = torch.topk(v_adj, k=k, largest=True)
        return idx.tolist()

    def _get_layers(self, name: str = "language_model"):
        """
        兼容：
        - model.language_model.layers
        - model.model.language_model.layers
        """
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
        """
        兼容:
        - model.visual.merger
        - model.model.visual.merger
        """
        visual = getattr(self.model, "visual", None)
        if visual is None:
            inner = getattr(self.model, "model", None)
            visual = getattr(inner, "visual", None) if inner is not None else None

        if visual is None:
            return None

        return getattr(visual, "merger", None)

    def _to_1d(self, out: torch.Tensor):
        """
        把 hook 的输出统一转成最后一个 token 对应的一维向量
        """
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
        """
        把 hook 的输出统一转成二维矩阵 [T, D]
        - [B, T, D] -> 取 batch0 => [T, D]
        - [T, D] -> 原样
        - [D] -> [1, D]
        """
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
        """
        返回最后 token hidden 上的 top-k index
        """
        if v is None:
            return None
        k = min(k, v.numel())
        idx = self._topk_idx(v, k)
        return idx

    def _make_hook(self, layer_idx: int, layer_name: str):
        def hook(m, inp, out):
            # step = 0 通常是 prompt / prefill 阶段，原统计逻辑中忽略
            if self.current_step == 0:
                return

            v = self._to_1d(out)
            if v is None:
                return

            topk = self._extract_topk_idx(v, k=3)
            if not topk:
                return

            # 只保存 1D 向量，避免后续统计时重复做 shape 处理
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

    # ================= 新增 hook: prefill/decode 保存逻辑 =================

    def _make_prefill_A_hook(self):
        def hook(module, inp, out):
            # prefill 阶段：current_step == 0
            if self.current_step != 0:
                return

            mat = self._to_2d(out)
            if mat is None:
                return

            self.prefill_A = mat.clone()

            # A_norm = torch.norm(self.prefill_A, dim=1)

            # max_val, max_pos = torch.max(A_norm, dim=0)
            # print(self.prefill_A.shape)
            # print("A_norm max value:", max_val.item())
            # print("A_norm max pos:", max_pos.item())

        return hook

    def _make_prefill_decode_mlp27_hook(self):
        def hook(module, inp, out):
            if self.current_step == 0:
                # prefill: 保存二维矩阵 B
                mat = self._to_2d(out)
                if mat is not None:
                    self.prefill_B = mat.clone()
            else:
                # decode: 保存最后token的一维向量
                v = self._to_1d(out)
                if v is not None:
                    self.mlp_list.append(v.clone())

        return hook

    def _make_decode_attn27_hook(self):
        def hook(module, inp, out):
            # 仅 decode 阶段记录
            if self.current_step == 0:
                return

            v = self._to_1d(out)
            if v is None:
                return

            self.attn_list.append(v.clone())

        return hook

    def _stack_vec_list(self, vec_list: List[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        把 list[Tensor] 堆叠成 [N, D]
        """
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
        """
        计算每层每个向量自身的 mean/std
        返回:
            {layer_id: {"mean": [...], "std": [...]} }
        """
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
        """
        计算单个 dict 内部相邻两项的指标。
        vec_dict: {layer_id: [v0, v1, v2, ...]}

        支持 metrics:
        - "dot"
        - "diff_mean"
        - "rmse"
        - "cos"
        """
        results = {metric: {} for metric in metrics}

        for key, vec_list in vec_dict.items():
            mat = self._stack_vec_list(vec_list)
            if mat is None or mat.size(0) < 2:
                for metric in metrics:
                    results[metric][key] = []
                continue

            v1 = mat[:-1]   # [N-1, D]
            v2 = mat[1:]    # [N-1, D]
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
        """
        计算两个 defaultdict(list) 中对应 key、对应位置向量之间的指标。
        dict1: {layer_id: [v0, v1, ...]}
        dict2: {layer_id: [u0, u1, ...]}

        支持 metrics:
        - "cos"
        - "dot"
        - "rmse"
        - "diff_mean"
        """
        common_keys = set(dict1.keys()) & set(dict2.keys())
        if not common_keys:
            print("Warning: No common layers found between the two vector dicts")
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
        """
        对 seq_dict 做滑动累计统计。

        输入:
            seq_dict:
                layer_idx -> rank -> [idx_1, idx_2, ..., idx_T]

        输出:
            {
                23: {
                    "top1": {
                        "mode_idx": [...],
                        "mode_rate": [...]
                    },
                    "top2": {...},
                    "top3": {...}
                },
                24: {...}
            }
        """
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

                    # 更新当前 mode
                    # 若并列，取 idx 更小的，保证结果稳定
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

    # ================= 新增: projection internal helpers =================

    def _vec_summary_stats(self, x: torch.Tensor, unbiased: bool = False):
        """
        x: [T]
        返回:
        - max
        - max_pos
        - min
        - min_pos
        - mean
        - std
        """
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


    def _matvec_stats(self, mat: torch.Tensor, vec_list: List[torch.Tensor], unbiased: bool = False):
        """
        mat: [T, D]
        vec_list: list[[D]]

        对每个向量 v 做:
            score = mat @ v   -> [T]

        然后对 score 统计:
        - max
        - max_pos
        - min
        - min_pos
        - mean
        - std

        返回:
        {
            "max": [...],
            "max_pos": [...],
            "min": [...],
            "min_pos": [...],
            "mean": [...],
            "std": [...]
        }
        """
        result = {
            "max": [],
            "max_pos": [],
            "min": [],
            "min_pos": [],
            "mean": [],
            "std": [],
        }

        if mat is None or len(vec_list) == 0:
            return result

        if not isinstance(mat, torch.Tensor):
            mat = torch.tensor(mat)

        mat = mat.detach().float().cpu()

        if mat.dim() != 2:
            raise ValueError(f"mat must be 2D, got shape={tuple(mat.shape)}")

        D = mat.size(1)

        for v in vec_list:
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v)

            v = v.detach().float().cpu().reshape(-1)

            if v.numel() != D:
                raise ValueError(
                    f"dimension mismatch: mat.shape={tuple(mat.shape)}, vec.shape={tuple(v.shape)}"
                )

            score = mat @ v   # [T]
            stats = self._vec_summary_stats(score, unbiased=unbiased)

            result["max"].append(stats["max"])
            result["max_pos"].append(stats["max_pos"])
            result["min"].append(stats["min"])
            result["min_pos"].append(stats["min_pos"])
            result["mean"].append(stats["mean"])
            result["std"].append(stats["std"])

        return result

