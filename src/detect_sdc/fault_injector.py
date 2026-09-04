import random
import re

import numpy as np
import torch


class FaultInjector:
    """
    故障注入器，支持两种模式：
    1. activation: 在 forward 过程中对某层输出做 bit flip
    2. weight:     在推理前直接对某层 weight 参数做 bit flip

    说明：
    - 统一只保留多 bit flip 接口
    - 单 bit flip 可通过 bit_positions=[k] 实现
    - 实际支持任意多 bit flip（通过 bit_positions 指定）
    """

    # 有符号整数类型对应的无符号 numpy 类型
    # 用于按位异或 bit flip
    _int_to_uint = {
        torch.int8: np.uint8,
        torch.int16: np.uint16,
        torch.int32: np.uint32,
        torch.int64: np.uint64
    }
    BIT_POLICIES = (
        "random",
        "mantissa_only",
        "low_mantissa",
        "low_exponent",
    )
    _mantissa_bits = {
        torch.float16: tuple(range(10)),
        torch.bfloat16: tuple(range(7)),
        torch.float32: tuple(range(23)),
    }
    _low_mantissa_bits = {
        torch.float16: tuple(range(5)),
        torch.bfloat16: tuple(range(4)),
        torch.float32: tuple(range(11)),
    }
    _low_exponent_bits = {
        torch.float16: tuple(range(15)),
        torch.bfloat16: tuple(range(12)),
        torch.float32: tuple(range(28)),
    }

    def __init__(self, model, mode="activation"):
        """
        Args:
            model: PyTorch 模型
            mode:  "activation" 或 "weight"
        """
        if mode not in ("activation", "weight"):
            raise ValueError("mode must be 'activation' or 'weight'")
        self.model = model
        self.mode = mode

        # 注入位置：张量 flatten 后的元素索引
        self.idx = -1

        # 多 bit 模式，例如 [3, 13]
        self.bit_positions = None

        # 随机注入时翻转几个 bit，默认 1 bit
        # 即便只保留多 bit 接口，num_bits=1 仍然是合法的
        self.num_bits = 1
        self.bit_policy = "random"
        self._manual_bit_positions = False

        # 目标模块名，例如 "model.layers.10.mlp.down_proj"
        self.module_name = None

        # 实际选中的注入目标
        self.select_inject_target = None

        # 是否已经完成注入
        self.select_target_has_injected = False

        # 当前 forward step（用于 activation 模式）
        self.current_step = 0

        # 在第几个 step 注入激活值故障
        self.inject_step = -1

        # 故障记录信息
        self.fault_info = None

        # hook 句柄
        self.fault_hook_handle = None
        self.step_hook_handle = None
        self.save_hook_handles = []

        # weight 模式专用：是否已经注入过权重故障
        self.weight_fault_injected = False

        # weight 模式专用：备份原始权重值，便于恢复
        self.weight_backup = None

    # ============================================================
    # 基础工具函数
    # ============================================================
    def _get_bit_width(self, dtype):
        """
        根据 tensor dtype 返回 bit 宽度。
        """
        return {
            torch.int8: 8,
            torch.uint8: 8,
            torch.int16: 16,
            torch.float16: 16,
            torch.bfloat16: 16,
            torch.int32: 32,
            torch.float32: 32,
            torch.int64: 64
        }.get(dtype, None)

    def _normalize_bit_positions(self, bit_positions):
        """
        统一把 bit_positions 规格化为 list[int]。

        支持：
        - int，例如 13
        - list / tuple，例如 [3, 13]

        说明：
        - 虽然外部统一推荐传 list，但这里保留 int 容错，方便调用
        """
        if isinstance(bit_positions, int):
            return [bit_positions]
        elif isinstance(bit_positions, (list, tuple)):
            if len(bit_positions) == 0:
                raise ValueError("bit_positions list/tuple is empty")
            return [int(b) for b in bit_positions]
        else:
            raise TypeError("bit_positions must be int, list, or tuple")

    def _build_bit_mask(self, bit_positions, bit_width):
        """
        根据多个 bit 位置构造 xor mask，并检查越界。
        返回：
        - mask
        - 去重并排序后的 bit_positions
        """
        mask = 0
        used = set()

        for b in bit_positions:
            if b < 0 or b >= bit_width:
                raise ValueError(f"bit position {b} out of range for bit width {bit_width}")
            if b in used:
                continue
            used.add(b)
            mask |= (1 << b)

        return mask, sorted(list(used))

    def _set_fault_info(
        self,
        module_name,
        dtype,
        idx,
        before,
        after,
        step,
        mode,
        bit_positions=None
    ):
        """
        记录本次故障注入的元信息。
        """

        self.fault_info = {
            "mode": mode,                    # activation / weight
            "module": module_name,          # 模块名
            "component": self._module_component(module_name),
            "layer_index": self._module_layer_index(module_name),
            "op_type": self._module_op_type(module_name),
            "forward": step,                # 注入发生的 forward step
            "dtype": dtype,                 # 张量类型
            "idx": idx,                     # flatten 后元素索引
            "bit_positions": bit_positions, # 多 bit 列表
            "bit_categories": self._bit_categories(
                dtype,
                bit_positions or [],
            ),
            "bit_policy": (
                "manual" if self._manual_bit_positions else self.bit_policy
            ),
            "before": before,               # 翻转前数值
            "after": after                  # 翻转后数值
        }

    @staticmethod
    def _module_component(module_name):
        name = str(module_name)
        if "mm_projector" in name or "merger" in name:
            return "projector"
        if "vision" in name or ".visual." in name:
            return "vision"
        if name == "lm_head" or name.endswith(".lm_head"):
            return "lm_head"
        if "language_model.layers." in name or name.startswith("model.layers."):
            return "language"
        return "other"

    @classmethod
    def _module_layer_index(cls, module_name):
        if cls._module_component(module_name) != "language":
            return None
        match = re.search(r"(?:language_model\.)?layers\.(\d+)", str(module_name))
        return None if match is None else int(match.group(1))

    @staticmethod
    def _module_op_type(module_name):
        name = str(module_name)
        known = (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "qkv",
            "fc1",
            "fc2",
            "lm_head",
        )
        for candidate in known:
            if name == candidate or name.endswith(f".{candidate}"):
                return candidate
        return name.rsplit(".", maxsplit=1)[-1]

    @staticmethod
    def _bit_categories(dtype, bit_positions):
        layouts = {
            torch.float16: (10, 15),
            torch.bfloat16: (7, 15),
            torch.float32: (23, 31),
            "torch.float16": (10, 15),
            "torch.bfloat16": (7, 15),
            "torch.float32": (23, 31),
        }
        layout = layouts.get(dtype)
        if layout is None:
            return ["integer"] * len(bit_positions)
        exponent_start, sign_bit = layout
        return [
            (
                "sign"
                if bit == sign_bit
                else "exponent"
                if bit >= exponent_start
                else "mantissa"
            )
            for bit in bit_positions
        ]

    def _candidate_bit_positions(self, dtype):
        bit_width = self._get_bit_width(dtype)
        if bit_width is None:
            raise ValueError(f"Unsupported dtype for bit flip: {dtype}")
        if self.bit_policy == "random":
            return tuple(range(bit_width))
        if self.bit_policy == "mantissa_only":
            layouts = self._mantissa_bits
        elif self.bit_policy == "low_mantissa":
            layouts = self._low_mantissa_bits
        elif self.bit_policy == "low_exponent":
            layouts = self._low_exponent_bits
        else:
            raise ValueError(f"Unsupported bit_policy: {self.bit_policy!r}")
        try:
            return layouts[dtype]
        except KeyError as error:
            raise ValueError(
                f"bit_policy={self.bit_policy!r} requires a floating dtype, "
                f"got {dtype}"
            ) from error

    # ============================================================
    # bit flip 核心函数
    # ============================================================
    def flip_bits(self, tensor: torch.Tensor, idx: int, bit_positions):
        """
        对 tensor.flatten()[idx] 的一个或多个 bit 做 bit flip，
        返回一个新的 tensor（不会原地修改输入 tensor）。

        参数：
        - bit_positions:
          1. int，例如 13
          2. list / tuple，例如 [3, 13]

        注意：
        - 对 float16 / float32 / bfloat16 采用按底层 bit 表示翻转
        - 对 int 类型采用无符号 view 后异或翻转
        """
        if not isinstance(tensor, torch.Tensor):
            print("[FaultInjector] input is not a tensor")
            return tensor

        flat = tensor.flatten()
        if idx < 0 or idx >= flat.numel():
            raise IndexError(f"idx {idx} out of range, numel={flat.numel()}")

        elem = flat[idx]
        dtype = tensor.dtype
        device = tensor.device

        bit_width = self._get_bit_width(dtype)
        if bit_width is None:
            print(f"[FaultInjector] unsupported dtype: {dtype}")
            return tensor

        bit_positions = self._normalize_bit_positions(bit_positions)
        mask, bit_positions = self._build_bit_mask(bit_positions, bit_width)

        # 克隆一份，避免原地修改传入 tensor
        new_flat = flat.clone()

        # 拿到 CPU 上的单个元素，方便用 numpy 做 view / xor
        x_cpu = elem.detach().cpu()

        # ---------- float16 ----------
        if dtype == torch.float16:
            x_uint = x_cpu.numpy().view(np.uint16)
            x_flipped = (x_uint ^ np.uint16(mask))
            x_flipped = np.expand_dims(x_flipped, 0).view(np.float16)

            new_tensor = torch.from_numpy(x_flipped).to(device)
            new_flat[idx] = new_tensor[0]

            self._set_fault_info(
                self.module_name,
                str(dtype),
                idx,
                elem.item(),
                new_tensor[0].item(),
                self.inject_step,
                self.mode,
                bit_positions=bit_positions
            )
            return new_flat.view_as(tensor)

        # ---------- bfloat16 ----------
        elif dtype == torch.bfloat16:
            # bfloat16 没有直接 numpy dtype，这里用 float32 的高 16 位模拟
            x_fp32 = x_cpu.to(torch.float32).numpy()
            x_uint32 = x_fp32.view(np.uint32)
            x_bf16_uint16 = (x_uint32 >> 16).astype(np.uint16)

            x_flipped_uint16 = x_bf16_uint16 ^ np.uint16(mask)
            x_flipped_uint32 = x_flipped_uint16.astype(np.uint32) << 16
            x_flipped_fp32 = x_flipped_uint32.view(np.float32)
            x_flipped_fp32 = np.array([x_flipped_fp32], dtype=np.float32)

            new_tensor = torch.from_numpy(x_flipped_fp32).to(torch.bfloat16).to(device)
            new_flat[idx] = new_tensor[0]

            self._set_fault_info(
                self.module_name,
                str(dtype),
                idx,
                elem.item(),
                new_tensor[0].item(),
                self.inject_step,
                self.mode,
                bit_positions=bit_positions
            )
            return new_flat.view_as(tensor)

        # ---------- float32 ----------
        elif dtype == torch.float32:
            x_uint = x_cpu.numpy().view(np.uint32)
            x_flipped = (x_uint ^ np.uint32(mask))
            x_flipped = np.expand_dims(x_flipped, 0).view(np.float32)

            new_tensor = torch.from_numpy(x_flipped).to(device)
            new_flat[idx] = new_tensor[0]

            self._set_fault_info(
                self.module_name,
                str(dtype),
                idx,
                elem.item(),
                new_tensor[0].item(),
                self.inject_step,
                self.mode,
                bit_positions=bit_positions
            )
            return new_flat.view_as(tensor)

        # ---------- int 类型 ----------
        elif dtype in self._int_to_uint:
            x_uint = x_cpu.numpy().view(self._int_to_uint[dtype])
            x_flipped = x_uint ^ np.array(mask, dtype=self._int_to_uint[dtype])

            original_np_dtype = x_cpu.numpy().dtype
            x_flipped = np.array([x_flipped], dtype=self._int_to_uint[dtype])

            new_tensor = torch.from_numpy(x_flipped.view(original_np_dtype)).to(device)
            new_flat[idx] = new_tensor[0]

            self._set_fault_info(
                self.module_name,
                str(dtype),
                idx,
                elem.item(),
                new_tensor[0].item(),
                self.inject_step,
                self.mode,
                bit_positions=bit_positions
            )
            return new_flat.view_as(tensor)

        else:
            print(f"[FaultInjector] unsupported dtype: {dtype}")
            return tensor

    def random_bitflip(self, tensor: torch.Tensor, num_bits: int = None):
        """
        在 tensor 中随机选一个元素、随机选若干个 bit 做翻转。
        如果 idx / bit_positions 已经提前设定，则使用指定值。

        Args:
            tensor: 输入张量
            num_bits: 随机翻转 bit 个数
                      若为 None，则使用 self.num_bits
        """
        if not isinstance(tensor, torch.Tensor):
            print("[FaultInjector] input is not a tensor")
            return tensor

        flat = tensor.flatten()
        if flat.numel() == 0:
            print("[FaultInjector] tensor is empty")
            return tensor

        if self.idx == -1:
            self.idx = random.randint(0, flat.numel() - 1)

        bit_width = self._get_bit_width(tensor.dtype)
        if bit_width is None:
            print("[FaultInjector] unsupported bit width")
            return tensor

        if num_bits is None:
            num_bits = self.num_bits

        if num_bits <= 0:
            raise ValueError("num_bits must be >= 1")

        # 优先使用手动指定的 bit_positions
        if self.bit_positions is not None:
            bit_positions = self.bit_positions
        else:
            candidates = self._candidate_bit_positions(tensor.dtype)
            if num_bits > len(candidates):
                raise ValueError(
                    f"num_bits={num_bits} exceeds {self.bit_policy} "
                    f"candidate count={len(candidates)} for {tensor.dtype}"
                )
            bit_positions = random.sample(candidates, num_bits)
            self.bit_positions = bit_positions

        return self.flip_bits(tensor, self.idx, bit_positions)

    # ============================================================
    # step 计数相关（activation 模式使用）
    # ============================================================
    def make_step_counter_hook(self):
        """
        构造 step 计数 hook。
        每当 lm_head forward 一次，就把 current_step + 1。
        """
        def hook(module, inputs, output):
            self.current_step += 1
            return output
        return hook

    def register_step_hooks(self):
        """
        在 lm_head 上注册 step 计数 hook。
        用于 activation 模式下控制“在第几个 forward step 注入”。
        """
        self.step_hook_handle = self.model.lm_head.register_forward_hook(
            self.make_step_counter_hook()
        )

    # ============================================================
    # 目标模块选择
    # ============================================================
    def _collect_eligible_modules(self):
        """
        收集所有可注入模块。
        当前仅选择 torch.nn.Linear 层。
        """
        return [
            (name, mod)
            for name, mod in self.model.named_modules()
            if isinstance(mod, torch.nn.Linear)
        ]

    def _get_target_module(self):
        """
        获取目标模块：
        - 如果 self.module_name 已指定，则按名字寻找
        - 否则从所有 Linear 层中随机选一个
        """
        eligible = self._collect_eligible_modules()

        if len(eligible) == 0:
            raise ValueError("No eligible Linear modules found.")

        # 未指定模块名：随机选一个 Linear 层
        if self.module_name is None:
            name, module = random.choice(eligible)
            self.module_name = name
            self.select_inject_target = name
            return name, module

        # 已指定模块名：找到对应模块
        matches = [(n, m) for n, m in eligible if n == self.module_name]
        if len(matches) == 0:
            raise ValueError(f"Specified module_name '{self.module_name}' not found.")

        name, module = matches[0]
        self.select_inject_target = name
        return name, module

    # ============================================================
    # activation 模式：在 forward 输出上做 bit flip
    # ============================================================
    def make_fault_hook(self):
        """
        构造 activation 注入 hook。

        逻辑：
        - 只在 mode == "activation" 时生效
        - 只在 current_step == inject_step 时注入一次
        - 对该层输出 output 做一次 bit flip
        """
        def hook(module, inputs, output):
            if self.mode != "activation":
                return output

            if self.select_target_has_injected:
                return output

            # 当前只处理 output 是单个 Tensor 的情况
            if not isinstance(output, torch.Tensor):
                return output

            if self.current_step != self.inject_step:
                return output

            out = output.clone()
            out = self.random_bitflip(out, num_bits=self.num_bits)

            self.select_target_has_injected = True

            # 注入完成后移除 hook，避免重复注入
            if self.fault_hook_handle is not None:
                self.fault_hook_handle.remove()
                self.fault_hook_handle = None

            return out

        return hook

    def register_fault_hooks(self):
        """
        注册 activation 故障 hook。

        执行流程：
        1. 选定一个目标 Linear 层
        2. 若 inject_step 未指定，则随机选一个 step
        3. 在该层 forward 输出上挂 hook
        """
        if self.mode != "activation":
            return

        name, module = self._get_target_module()

        # 如果没有指定注入 step，则随机选一个
        if self.inject_step == -1:
            self.inject_step = 0

        self.fault_hook_handle = module.register_forward_hook(self.make_fault_hook())

    # ============================================================
    # weight 模式：直接修改模块参数
    # ============================================================
    def inject_weight_fault(self):
        """
        对目标模块的 weight 参数做一次 bit flip。

        逻辑：
        1. 找到目标模块
        2. 取 module.weight.data
        3. 随机选一个权重元素 + 随机若干个 bit
        4. 做 bit flip
        5. 把修改后的结果写回 parameter.data

        注意：
        - 这是“持久性”修改，后续 forward 都会使用这个坏掉的权重
        - 若想恢复，需要调用 restore_weight_fault()
        """
        if self.mode != "weight":
            return

        if self.weight_fault_injected:
            return

        name, module = self._get_target_module()

        if not hasattr(module, "weight") or module.weight is None:
            raise ValueError(f"Module '{name}' has no weight parameter.")

        weight = module.weight.data
        flat = weight.flatten()

        if flat.numel() == 0:
            raise ValueError(f"Module '{name}' weight is empty.")

        if self.idx == -1:
            self.idx = random.randint(0, flat.numel() - 1)

        bit_width = self._get_bit_width(weight.dtype)
        if bit_width is None:
            raise ValueError(f"Unsupported weight dtype: {weight.dtype}")

        # 如果没有手工指定 bit_positions，则随机生成
        if self.bit_positions is None:
            if self.num_bits <= 0:
                raise ValueError("self.num_bits must be >= 1")
            candidates = self._candidate_bit_positions(weight.dtype)
            if self.num_bits > len(candidates):
                raise ValueError(
                    f"num_bits={self.num_bits} exceeds {self.bit_policy} "
                    f"candidate count={len(candidates)} for {weight.dtype}"
                )
            self.bit_positions = random.sample(candidates, self.num_bits)

        # 只备份被修改的那个元素，节省空间
        original_value = flat[self.idx].item()
        self.weight_backup = {
            "module_name": name,
            "idx": self.idx,
            "value": original_value
        }

        # 对整个 weight 张量返回一个翻转后的新张量
        new_weight = self.flip_bits(weight, self.idx, self.bit_positions)

        # 写回权重参数
        module.weight.data.copy_(new_weight)

        self.weight_fault_injected = True
        self.select_target_has_injected = True

        # weight 注入没有明确 forward step，这里可记录为 -1
        if self.fault_info is not None:
            self.fault_info["forward"] = -1
            self.fault_info["forward_norm"] = None

    def restore_weight_fault(self):
        """
        恢复之前注入过的单点权重故障。
        只恢复被修改的那个权重元素。

        注意：
        - 当前只恢复“被修改前的数值”
        - 不关心当时翻转了几个 bit，因为最终恢复的是原始值
        """
        if self.weight_backup is None:
            return

        backup_name = self.weight_backup["module_name"]
        idx = self.weight_backup["idx"]
        value = self.weight_backup["value"]

        if self.module_name != backup_name:
            raise ValueError("Backup module_name mismatch.")

        _, module = self._get_target_module()
        flat = module.weight.data.flatten()
        flat[idx] = torch.tensor(
            value,
            dtype=module.weight.data.dtype,
            device=module.weight.data.device
        )

        self.weight_fault_injected = False
        self.weight_backup = None
        self.select_target_has_injected = False

    # ============================================================
    # 外部接口
    # ============================================================
    def inject(self):
        """
        统一注入接口：
        - weight 模式：立即修改权重
        - activation 模式：注册 forward hook，等待推理时触发
        """
        if self.mode == "weight":
            self.inject_weight_fault()
        elif self.mode == "activation":
            self.register_fault_hooks()
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def unregister_hooks(self):
        """
        移除所有已经注册的 hook。
        activation 模式推理结束后建议调用。
        """
        if self.fault_hook_handle is not None:
            self.fault_hook_handle.remove()

        if self.step_hook_handle is not None:
            self.step_hook_handle.remove()

        for h in self.save_hook_handles:
            h.remove()

        self.fault_hook_handle = None
        self.step_hook_handle = None
        self.save_hook_handles = []

    def reset(self):
        """
        重置故障注入器内部状态。
        不会自动恢复已经改坏的权重。
        如果之前做过 weight 注入，先调用 restore_weight_fault()。
        """
        self.idx = -1
        self.bit_positions = None
        self.num_bits = 1
        self.bit_policy = "random"
        self._manual_bit_positions = False
        self.inject_step = -1
        self.current_step = 0
        self.module_name = None
        self.fault_info = None
        self.select_inject_target = None
        self.select_target_has_injected = False
        self.weight_fault_injected = False
        self.weight_backup = None

    def set_num_bits(self, num_bits: int):
        """
        设置随机注入时翻转的 bit 个数。
        例如：
        - 1: 单 bit flip（通过 bit_positions=[k] 也可实现）
        - 2: 双 bit flip
        """
        if num_bits <= 0:
            raise ValueError("num_bits must be >= 1")
        self.num_bits = num_bits

    def set_bit_policy(self, bit_policy: str):
        policy = str(bit_policy).strip().lower()
        if policy not in self.BIT_POLICIES:
            raise ValueError(
                f"bit_policy must be one of {self.BIT_POLICIES}, got {bit_policy!r}"
            )
        self.bit_policy = policy

    def set_inject_info(
        self,
        idx: int,
        module_name: str,
        inject_step: int = -1,
        bit_positions=None
    ):
        """
        手动指定注入参数。

        Args:
            idx:           flatten 后元素索引
            module_name:   目标模块名
            inject_step:   activation 模式下注入 step；weight 模式可设为 -1
            bit_positions: 多 bit 模式使用，例如 [3, 13]
                           单 bit 也写成 [13]
        """
        self.idx = idx
        self.inject_step = inject_step
        self.module_name = module_name
        self.select_inject_target = module_name
        self.bit_positions = bit_positions

        if bit_positions is not None:
            normalized = self._normalize_bit_positions(bit_positions)
            self.bit_positions = normalized
            self.num_bits = len(normalized)
            self._manual_bit_positions = True
