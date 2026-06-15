import torch
from collections import defaultdict, Counter
from typing import Optional, Tuple, List


class Detecter:
    def __init__(self, model, fixed_layer_idx: int = 27):
        self.model = model
        self.fixed_layer_idx = fixed_layer_idx

        self.handles = []
        self.current_step = 0
        self._step_hook_handle = None

        # 统计结构（不再关心 layer_idx）
        # self.idx_counter[layer_name] = Counter({idx: count, ...})
        self.idx_counter = defaultdict(Counter)
        # self.total_count[layer_name] = total samples counted
        self.total_count = defaultdict(int)

    # ----------------- public API -----------------

    def register(self):
        layers = self._get_layers("language_model")
        if layers is None:
            raise RuntimeError("Cannot find language_model.layers")

        li = self.fixed_layer_idx

        # 只对固定层注册两个钩子
        h1 = layers[li].input_layernorm.register_forward_hook(
            self._make_hook("input_layernorm")
        )
        h2 = layers[li].mlp.down_proj.register_forward_hook(
            self._make_hook("mlp.down_proj")
        )
        self.handles.extend([h1, h2])

        # 用 lm_head 的 forward 作为“每生成一个 token 的 step 计数”
        self._step_hook_handle = self.model.lm_head.register_forward_hook(
            self._make_step_counter_hook()
        )

    def reset(self, clear_stats: bool = False):
        self.current_step = 0
        if clear_stats:
            self.idx_counter.clear()
            self.total_count.clear()

    def unregister(self):
        for h in self.handles:
            h.remove()
        self.handles = []

        self.current_step = 0

        if self._step_hook_handle is not None:
            self._step_hook_handle.remove()
            self._step_hook_handle = None

    def is_error_by_thresholds(
        self,
        k_by_name: dict,
        default_k: float = 0.0,
        min_total: int = 1,
    ) -> bool:
        """
        对不同 layer_name 使用不同阈值 k。

        k_by_name: 例如 {"input_layernorm": 0.15, "mlp.down_proj": 0.35}
        default_k: 某个 layer_name 没配置时用这个阈值
        min_total: 样本数太少时跳过
        """
        for layer_name, c in self.idx_counter.items():
            _, _, total, diff_ratio = self._mode_and_diff_ratio_from_counter(c)
            if total < min_total:
                continue

            k = k_by_name.get(layer_name, default_k)
            if diff_ratio > k:
                return True

        return False


    def error_report_by_thresholds(
        self,
        k_by_name: dict,
        default_k: float = 0.0,
        min_total: int = 1,
    ):
        """
        报告每个 layer_name 的 diff_ratio、阈值k、是否超阈值。
        返回列表： (is_bad, diff_ratio, k, total, layer_name, mode_idx)
        按“更危险”的排前面：先 bad，再 diff_ratio 降序。
        """
        rows = []
        for layer_name, c in self.idx_counter.items():
            mode_idx, _, total, diff_ratio = self._mode_and_diff_ratio_from_counter(c)
            if total < min_total:
                continue

            k = k_by_name.get(layer_name, default_k)
            is_bad = diff_ratio > k
            rows.append((is_bad, diff_ratio, k, total, layer_name, mode_idx))

        rows.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return rows

    def error_report(self, min_total: int = 1):
        """
        输出每个 layer_name 的 (diff_ratio, total, layer_name, mode_idx)，便于你调 k。
        """
        rows = []
        for layer_name, c in self.idx_counter.items():
            mode_idx, _, total, diff_ratio = self._mode_and_diff_ratio_from_counter(c)
            if total < min_total:
                continue
            rows.append((diff_ratio, total, layer_name, mode_idx))
        rows.sort(reverse=True)
        return rows

    # ----------------- internal helpers -----------------

    def _make_step_counter_hook(self):
        def hook(module, input, output):
            self.current_step += 1
            return output
        return hook

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
        elif name == "visual":
            return getattr(lm, "blocks", None)
        return None

    def _extract_top1_idx(self, out: torch.Tensor) -> Optional[int]:
        """
        默认统计：batch=0、最后一个 token 的 hidden 向量上的 argmax。
        out 常见形状：[B, T, D] 或 [T, D] 或 [D]
        """
        if out is None:
            return None

        # 有的模块可能返回 tuple/list（保险起见）
        if isinstance(out, (tuple, list)):
            if len(out) == 0:
                return None
            out = out[0]

        if not torch.is_tensor(out):
            return None

        # 转到 cpu/float32 方便统计
        out = out.detach().to(dtype=torch.float32, device="cpu")

        if out.dim() == 3:
            # [B, T, D] -> 取 batch0、最后 token
            v = out[0, -1, :]
        elif out.dim() == 2:
            # [T, D] -> 取最后 token
            v = out[-1, :]
        elif out.dim() == 1:
            # [D]
            v = out
        else:
            return None

        return int(torch.argmax(v).item())

    def _make_hook(self, layer_name: str):
        def hook(m, inp, out):
            # 只在 step!=0 时统计（你要求的行为）
            if self.current_step == 0:
                return

            idx = self._extract_top1_idx(out)
            if idx is None:
                return

            self.idx_counter[layer_name][idx] += 1
            self.total_count[layer_name] += 1

        return hook

    def _mode_and_diff_ratio_from_counter(self, c: Counter) -> Tuple[Optional[int], int, int, float]:
        """
        返回：mode_idx, mode_count, total, diff_ratio
        diff_ratio = 1 - mode_count/total
        """
        total = sum(c.values())
        if total == 0:
            return None, 0, 0, 0.0
        mode_idx, mode_count = c.most_common(1)[0]
        diff_ratio = 1.0 - (mode_count / total)
        return mode_idx, mode_count, total, diff_ratio
