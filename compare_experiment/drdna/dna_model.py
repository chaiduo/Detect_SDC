import torch
import torch.nn as nn
import timm
from scipy.stats import wasserstein_distance
import os
import random
import numpy as np
import logging
from collections import defaultdict

from utils.help_func import set_random_seed, get_latest_timestamp
from models.data_module import DnaDataModule


class DnaModel():
    def __init__(self, config):
        """
        初始化 DNA 模型。

        Args:
            config (dict): 模型配置，包括模型路径、设备等信息。
        """

        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # 1. 设置 CUDA 环境变量
        self._setup_environment()

        # 2. 设置随机种子
        self.random_seed = self.config["environment"].get("random_seed", -1)
        if self.random_seed >= 0:
            set_random_seed(self.random_seed, self.cuda_device_index)
        else:
            # 内置地设置一个随机种子
            self.random_seed = random.randint(0, 10000)
            self.logger.info(f"未设置随机种子，将使用随机种子: {self.random_seed}")
            self.config["environment"]["random_seed"] = self.random_seed
            set_random_seed(self.random_seed, self.cuda_device_index)

        # 3. 加载数据集
        self.data_module = DnaDataModule(self.config)
        self.label_names = self.data_module.label_names
        self.img_key = self.data_module.img_key
        self.current_label_key = self.data_module.current_label_key
        if self.label_names:
            self.current_num_classes = len(self.label_names)
        else:
            # Fallback if label_names not directly available
            if self.config["dataset"].get("choice", "cifar100") == "cifar10":
                self.current_num_classes = 10
            elif self.config["dataset"].get("choice", "cifar100") == "cifar100":
                self.current_num_classes = 100
            elif self.config["dataset"].get("choice", "cifar100") == "imagenette2":
                self.current_num_classes = 10  # ImageNette has 10 classes
            else:
                self.logger.warning("无法从数据集确定类别数量。将使用默认值 1000 作为模型头部的类别数。")
                self.current_num_classes = 1000  # Default to ImageNet classes


        # 4. 加载模型
        self.model, self.model_status = self._load_model()
        self.activation_hook_handles = []

        # 5. 应用数据集转换并创建Dataloader
        # self.dataset = self._apply_transforms(raw_dataset)

        # 6. 初始化profile信息
        self.activations = defaultdict(list)
        self.profiling_sites = defaultdict(list)
        self.target_layer_names = []
        self.target_layer_output_shapes = {}  # 新增：记录目标层输出shape

    def _setup_environment(self):
        """
        设置 CUDA 环境变量并确定 PyTorch 计算设备。

        此私有方法从配置中获取 CUDA 设备索引，并据此初始化 PyTorch 的计算设备。
        如果 CUDA 不可用，则回退到使用 CPU。

        Returns:
            torch.device: PyTorch 计算设备对象 (例如 'cuda:0' 或 'cpu')。
        """
        self.cuda_device_index = self.config["environment"].get(
            "cuda_device_index", 0)
        self.device = torch.device(
            f"cuda:{self.cuda_device_index}" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"PyTorch 计算设备已设置为: {self.device}")

    def _load_model(self) -> tuple:
        """
        加载预训练模型。

        此私有方法根据配置中的模型路径加载预训练的 PyTorch 模型。
        如果模型文件不存在，则抛出异常。

        Returns:
            tuple: 包含以下元素的元组：
                - model (nn.Module): 加载的 PyTorch 模型对象。
                - model_status (str): 模型状态，指示是加载了微调模型还是原始预训练模型。
        """
        model_name = self.config["model"].get(
            "name", "vit_tiny_patch16_224.augreg_in21k_ft_in1k")
        load_finetuned_model = self.config["model"].get(
            "load_finetuned", False)
        finetuned_model_path = self.config["model"].get(
            "finetuned_path", "./finetuned_vit_imagenette2.pth")

        model = None
        model_status = "unknown"

        if load_finetuned_model and os.path.exists(finetuned_model_path):
            try:
                model = timm.create_model(
                    model_name, pretrained=False, num_classes=self.current_num_classes)
                model.load_state_dict(torch.load(
                    finetuned_model_path, map_location=self.device))
                self.logger.info(f"成功加载微调模型: {finetuned_model_path}")
                model_status = "finetuned"
            except Exception as e:
                self.logger.warning(f"加载微调模型失败: {e}。将回退到加载原始预训练模型。")
                model = timm.create_model(
                    model_name, pretrained=True, num_classes=self.current_num_classes)
                model_status = "pretrained"
        else:
            self.logger.info("未开启加载微调模型或未找到已微调模型文件。将使用原始预训练模型。")
            model = timm.create_model(
                model_name, pretrained=True, num_classes=self.current_num_classes)
            model_status = "pretrained"

        model.eval()
        model = model.to(self.device)

        return model, model_status

    def _create_get_activation_hook(self, name):
        """
        创建激活hook以捕获指定模块的输出。

        Args:
            name (str): 模块名称，用于标识激活数据。

        Returns:
            function: 一个hook函数，当指定模块被调用时会执行，捕获其输出。
        """
        def _get_activation_hook(module, input, output):
            # 将激活值从 GPU 转移到 CPU，并转换为 numpy 数组，以减少 GPU 内存占用
            # 对于线性层，output 通常是 (batch_size, num_features)
            # 如果是多维的（例如某些注意力机制的中间输出），将其展平到 (batch_size, -1)
            current_batch_activations = output.detach().cpu().numpy()
            self.logger.debug(f"捕获层 '{name}' 的激活值，形状: {current_batch_activations.shape}")
            if current_batch_activations.ndim > 2:
                current_batch_activations = current_batch_activations.reshape(
                    current_batch_activations.shape[0], -1)
            self.activations[name].append(current_batch_activations)

        return _get_activation_hook

    @staticmethod
    def _get_str_module_name_map():
        return {
            "linear": nn.Linear,
            "conv2d": nn.Conv2d,
            "identity": nn.Identity,
            "dropout": nn.Dropout,
            "norm": nn.LayerNorm,
            "gelu": nn.GELU,
            'block': timm.models.vision_transformer.Block,  # 针对 ViT 的 Block
            'block_attn': timm.models.vision_transformer.Attention,  # 针对 ViT 的 Attention
        }

    def _get_potential_target_layers(self):
        """
        获取模型中所有目标层的名称。

        此私有方法遍历模型的所有子模块，并返回一个包含目标层名称的列表。
        比如可以指定所有的卷积层或者线性层。

        Returns:
            list: 包含目标层名称的列表。
        例如：['layer1.0.conv1', 'layer1.0.conv2', 'layer2.0.conv1', ...]
        """
        target_layer_names = []

        specified_module_type = self.config.get(
            "profiling", {}).get("specified_module_type", "")
        if not specified_module_type:
            self.logger.warning("未指定目标层类型，将选择默认的linear层")
            specified_module_type = "linear"
        str_module_name_map = self._get_str_module_name_map()

        specified_layer_names = self.config.get(
            "profiling", {}).get("specified_layer_names", {})
        if specified_layer_names:
            self.logger.info(f"将从 {specified_layer_names} 中选择采样目标层")
        else:
            self.logger.warning(
                f"未指定采样层名称限制，将从所有 {specified_module_type} 层中进行采样")

        # 遍历模型的所有子模块
        for name, module in self.model.named_modules():
            if isinstance(module, str_module_name_map[specified_module_type]):
                # 将层名称添加到目标层列表中
                if specified_layer_names:
                    if any(specified_name_str in name for specified_name_str in specified_layer_names):
                        target_layer_names.append(name)
                else:
                    target_layer_names.append(name)

        self.logger.info(f"已识别 {len(target_layer_names)} 个目标层进行 Profiling。")
        return target_layer_names

    def get_fi_available_layers(self):
        """
        输出可以进行注错的layers
        """
        avaiable_layers_name = defaultdict(list)
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                avaiable_layers_name["linear"].append(name)
            elif isinstance(module, nn.Conv2d):
                avaiable_layers_name["conv2d"].append(name)
            elif isinstance(module, nn.Identity):
                avaiable_layers_name["identity"].append(name)
            elif isinstance(module, nn.Dropout):
                avaiable_layers_name["dropout"].append(name)
            elif isinstance(module, nn.LayerNorm):
                avaiable_layers_name["norm"].append(name)
            elif isinstance(module, nn.GELU):
                avaiable_layers_name["gelu"].append(name)
            else:
                avaiable_layers_name["else"].append(name)
                self.logger.warning(f"未识别的层类型: {type(module)}，名称: {name}")
        return avaiable_layers_name

    def _register_profiling_hooks(self):
        """
        注册用于分析模型性能的hook。

        Args:
            num_layers_to_sample (int): 要采样的层数。如果为 -1，则采样所有层。
            num_neurons_per_layer (int): 每层要采样的神经元数量。
        """

        self.logger.info("正在注册分析hook...")
        self._remove_activation_hooks()  # 清理之前的hook
        num_neurons_per_layer = self.config["profiling"].get(
            "num_neurons_per_layer", 5)
        num_layers_to_sample = self.config["profiling"].get(
            "num_layers_to_sample", -1)
        potential_target_layer_names = self._get_potential_target_layers()

        # 如果指定了采样层数，则从目标层中随机选择
        if num_layers_to_sample > 0 and num_layers_to_sample < len(potential_target_layer_names):
            layer_sampling_strategy = self.config["profiling"].get(
                "layer_sampling_strategy", "random")
            if layer_sampling_strategy == "random":
                self.target_layer_names = random.sample(
                    potential_target_layer_names, num_layers_to_sample)
                self.logger.info(
                    f"已随机选择 {num_layers_to_sample} 个目标层进行 Profiling。")
            elif layer_sampling_strategy == "avg_step":
                if num_layers_to_sample == 1:
                    # 如果只采样1层，则选择中间层（或近似中间层）
                    self.target_layer_names = [
                        potential_target_layer_names[len(potential_target_layer_names) // 2]]
                    self.logger.info(f"已通过平均步长采样选择 1 个目标层进行 Profiling。")
                else:
                    step = len(potential_target_layer_names) / \
                        num_layers_to_sample
                    sampled_indices = [int(i * step)
                                       for i in range(num_layers_to_sample)]
                    sampled_indices = [
                        min(idx, len(potential_target_layer_names) - 1) for idx in sampled_indices]
                    sampled_indices = sorted(list(set(sampled_indices)))
                    self.target_layer_names = [
                        potential_target_layer_names[i] for i in sampled_indices]
                    self.logger.info(
                        f"已通过平均步长采样选择 {len(self.target_layer_names)} 个目标层进行 Profiling。")
            else:
                self.target_layer_names = random.sample(
                    potential_target_layer_names, num_layers_to_sample)
                self.logger.warning(
                    f"未知采样策略: {layer_sampling_strategy}。将默认使用随机采样。")
        else:
            self.target_layer_names = potential_target_layer_names
            self.logger.info("将对所有目标层进行 Profiling。")

        # 1. 先推理一批，记录每个目标层的输出shape
        self._infer_and_record_layer_output_shapes()

        # 2. 注册hook并采样token_num*out_feature空间
        self.profiling_sites = defaultdict(list)
        for name in self.target_layer_names:
            layer = dict(self.model.named_modules()).get(name, None)
            shape = self.target_layer_output_shapes.get(name, None)
            if layer is not None and shape is not None:
                # shape: (token_num, out_feature) 或 (out_feature,)
                if len(shape) == 1:
                    token_num, out_feature = 1, shape[0]
                elif len(shape) >= 2:
                    token_num, out_feature = shape[0], shape[1]
                else:
                    self.logger.warning(f"层 '{name}' 输出shape异常: {shape}")
                    continue
                total_neurons = token_num * out_feature
                if total_neurons == 0:
                    self.logger.warning(f"层 '{name}' 的输出特征数为0，无法Profiling。")
                    continue
                num_neurons = min(num_neurons_per_layer, total_neurons)
                sampled_flat_indices = random.sample(
                    range(total_neurons), num_neurons)
                # 保存为(flat_index, token_idx, feature_idx)
                sampled_tuples = [(idx, idx // out_feature, idx % out_feature)
                                  for idx in sampled_flat_indices]
                self.profiling_sites[name] = sampled_tuples
                self.logger.info(
                    f"将对层 '{name}' 采样 {num_neurons} 个神经元(token维度支持)。")
                hook_fn = self._create_get_activation_hook(name)
                handle = layer.register_forward_hook(hook_fn)
                self.activation_hook_handles.append(handle)
            else:
                self.logger.warning(f"目标层 {name} 不存在或未能获取输出shape，无法注册hook。")
        if not self.profiling_sites:
            self.logger.warning("没有有效的目标层进行 Profiling。请检查模型结构或配置。")
        self.logger.info(f"已注册 {len(self.activation_hook_handles)} 个hook。")

    def _remove_activation_hooks(self):
        """
        移除所有注册的分析hook。

        此私有方法遍历所有注册的hook并将其移除，以清理模型状态。
        """
        for handle in self.activation_hook_handles:
            handle.remove()
        self.activation_hook_handles.clear()
        self.logger.info("已移除所有激活值hook。")

    def run_profiling_inference(self):
        """
        运行 Profiling 推理。

        此方法执行一次前向推理，并收集所有注册hook的激活数据。
        在 Profiling 过程中，模型将处于评估模式。
        """
        self.logger.info("正在运行 Profiling 推理...")

        # 清空之前的激活数据
        self.activations.clear()

        # 注册 Profiling hook
        self._register_profiling_hooks()
        if not self.profiling_sites:
            self.logger.error("没有有效的 Profiling 站点可供采样。请检查模型结构或配置。")
            return {}

        # 获取数据加载器
        dataloader = self._get_dataloader()

        # 存储记录的神经元值
        # key: (layer_name, neuron_index) -> list of values
        recorded_neuron_values = defaultdict(list)

        # 执行一次前向推理
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloader):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)

                # 记录每个采样的神经元的激活值
                for layer_name, activation_list in self.activations.items():
                    if activation_list:
                        current_batch_activations = activation_list.pop(
                            0)  # 获取最后一个批次的激活值
                        self.logger.debug(
                            f"层 '{layer_name}' 的当前批次激活值形状: {current_batch_activations.shape}")
                        shape = self.target_layer_output_shapes.get(
                            layer_name, None)
                        if shape is None:
                            self.logger.warning(
                                f"未找到层 '{layer_name}' 的输出shape，跳过采样。")
                            continue
                        if len(shape) == 1:
                            token_num, out_feature = 1, shape[0]
                        else:
                            token_num, out_feature = shape[-2], shape[-1]
                        # 采样点为(flat_index, token_idx, feature_idx)
                        for flat_idx, token_idx, feat_idx in self.profiling_sites[layer_name]:
                            try:
                                if current_batch_activations.ndim == 2:
                                    values = current_batch_activations[:, flat_idx]
                                else:
                                    self.logger.warning(
                                        f"激活shape异常: {current_batch_activations.shape}")
                                    continue
                                recorded_neuron_values[(
                                    layer_name, flat_idx, token_idx, feat_idx)].append(values)
                                self.logger.debug(
                                    f"记录层 '{layer_name}' 的采样点(flat_idx={flat_idx}, token={token_idx}, feat={feat_idx}) 激活值。shape: {values.shape}")
                            except Exception as e:
                                self.logger.warning(
                                    f"采样点(flat_idx={flat_idx}, token={token_idx}, feat={feat_idx}) 采集激活异常: {e}")
                # 清空self.activations以释放内存，因为当前批次的所有激活值都已经处理
                self.activations.clear()

                if (i + 1) % self.config["data_loader"].get("log_interval", 10) == 0:
                    self.logger.info(f"已处理 {i + 1}/{len(dataloader)} 批次。")

                if i + 1 >= self.config["data_loader"].get("iteration_limit", -1) > 0:
                    self.logger.info(
                        f"已达到迭代限制 {self.config['data_loader']['iteration_limit']}，停止 Profiling。")
                    break

        self.logger.info("Profiling 推理完成。")

        self._remove_activation_hooks()  # 清理hook
        return recorded_neuron_values

    def _calculate_tau1_single_sample(self, sample_activation: float, histogram_bin_edges: np.ndarray, histogram_counts: np.ndarray) -> float:
        """
        计算单个样本的 τ1 分数。

        Args:
            sample_activation (float): 单个样本的激活值。
            histogram_bin_edges (np.ndarray): 直方图的边界数组。
            histogram_counts (np.ndarray): 直方图的计数数组。

        Returns:
            float: 该样本的 τ1 分数。
        """
        if np.isinf(sample_activation) or np.isnan(sample_activation):
            self.logger.debug(
                f"样本激活值 {sample_activation} 非法，τ1分数返回无穷大。")
            return float('inf')  # 或者一个很大的固定值，表示极度异常

        # 计算直方图的总样本数，用于计算频率
        total_samples = np.sum(histogram_counts)
        if total_samples == 0:
            # 如果基准直方图为空，则无法判断，可以返回1或NaN
            return 1.0  # 无法判断，视为异常

        # 计算样本在直方图中的 bin 索引
        bin_idx = np.searchsorted(
            histogram_bin_edges, sample_activation, side='right') - 1

        if bin_idx < 0 or bin_idx >= len(histogram_counts):
            # 如果样本激活值超出直方图范围，则视为异常
            return 1.0
        # 获取所在 bin 的频率
        f_j = histogram_counts[bin_idx] / total_samples
        # 方法1：通过相邻bin进行平滑处理
        # if bin_idx - 1 >= 0:
        #     f_j += histogram_counts[bin_idx - 1] / total_samples
        # if bin_idx + 1 < len(histogram_counts):
        #     f_j += histogram_counts[bin_idx + 1] / total_samples
        # tau1 = 1.0 - f_j

        # 方法2：基于最大值缩放
        max_count = np.max(histogram_counts) if total_samples > 0 else 1
        tau1 = 1.0 - histogram_counts[bin_idx] / max_count

        if tau1 < 0:
            assert tau1 >= 0, "异常分数 tau1 计算结果不应为负值。"
            tau1 = 0.0  # 如果频率大于1，则视为正常，τ1 分数为 0
        return tau1

    def _calculate_tau2(self, current_batch_activations_sampled: np.ndarray, baseline_bin_edges: np.ndarray, baseline_counts: np.ndarray) -> float:
        """
        计算 τ2 分数。

        Args:
            current_batch_activations_sampled (np.ndarray): 当前批次的激活值，形状通常为 (batch_size, num_neurons)。
            baseline_bin_edges (np.ndarray): 整体 Profiling 期间的直方图边界数组。
            baseline_counts (np.ndarray): 整体 Profiling 期间的直方图计数数组。

        Returns:
            float: τ2 分数，表示当前批次与整体 Profiling 基准的差异。
        """
        # 检查输入
        if current_batch_activations_sampled.size == 0:
            self.logger.warning("当前批次采样神经元激活值为空，τ2分数返回inf")
            return float("inf")

        # 展平并处理nan/inf
        current_sampled_flat_activations = current_batch_activations_sampled.flatten()
        # 标记超出bin范围、nan、inf的数量
        num_nan = np.sum(np.isnan(current_sampled_flat_activations))
        num_inf = np.sum(np.isinf(current_sampled_flat_activations))
        # 有nan/inf直接返回inf
        if num_nan > 0 or num_inf > 0:
            self.logger.debug(f"当前批次激活值包含 {num_nan} 个nan, {num_inf} 个inf，τ2分数返回inf")
            return float("inf")

        # 统计落在bin外的数据
        min_edge, max_edge = baseline_bin_edges[0], baseline_bin_edges[-1]
        below_min_mask = current_sampled_flat_activations < min_edge
        above_max_mask = current_sampled_flat_activations > max_edge
        below_min = np.sum(below_min_mask)
        above_max = np.sum(above_max_mask)
        in_bin_mask = (~below_min_mask) & (~above_max_mask)

        # 用相同bin_edges统计直方图
        current_counts, _ = np.histogram(current_sampled_flat_activations, bins=baseline_bin_edges)

        # baseline部分
        baseline_total = np.sum(baseline_counts)
        if not np.isfinite(baseline_total) or baseline_total <= 0:
            self.logger.error("baseline_counts 求和非正或非有限，无法计算 τ2 分数。")
            return float('inf')

        # 统计bin外数据的均值距离
        below_min_vals = current_sampled_flat_activations[below_min_mask]
        above_max_vals = current_sampled_flat_activations[above_max_mask]
        # 如果没有bin外数据，mean_dist设为0
        if below_min > 0:
            below_min_dist = np.mean(min_edge - below_min_vals)
        else:
            below_min_dist = 0.0
        if above_max > 0:
            above_max_dist = np.mean(above_max_vals - max_edge)
        else:
            above_max_dist = 0.0

        # 构造扩展的pmf和bin中心
        n_bins = len(baseline_bin_edges) - 1
        # baseline扩展
        baseline_pmf = baseline_counts / baseline_total
        baseline_pmf_ext = np.zeros(n_bins + 2)
        baseline_pmf_ext[1:-1] = baseline_pmf
        # current扩展
        current_total = np.sum(current_counts) + below_min + above_max
        if not np.isfinite(current_total) or current_total <= 0:
            self.logger.warning("当前批次所有激活值均超出baseline直方图范围，τ2分数返回inf。")
            return float('inf')
        current_pmf_ext = np.zeros(n_bins + 2)
        current_pmf_ext[1:-1] = current_counts / current_total
        if below_min > 0:
            current_pmf_ext[0] = below_min / current_total
        if above_max > 0:
            current_pmf_ext[-1] = above_max / current_total

        # bin中心扩展
        bin_centers = (baseline_bin_edges[:-1] + baseline_bin_edges[1:]) / 2.0
        bin_width = (baseline_bin_edges[1] - baseline_bin_edges[0]) if n_bins > 0 else 1.0
        # 虚拟bin中心
        left_virtual = min_edge - (below_min_dist if below_min > 0 else bin_width)
        right_virtual = max_edge + (above_max_dist if above_max > 0 else bin_width)
        bin_centers_ext = np.concatenate([[left_virtual], bin_centers, [right_virtual]])

        # 检查pmf
        if not np.all(np.isfinite(baseline_pmf_ext)) or not np.all(baseline_pmf_ext >= 0):
            self.logger.error(f"baseline_pmf_ext 非法: {baseline_pmf_ext}")
            return float('inf')
        if not np.all(np.isfinite(current_pmf_ext)) or not np.all(current_pmf_ext >= 0):
            self.logger.error(f"current_pmf_ext 非法: {current_pmf_ext}")
            return float('inf')

        if len(bin_centers_ext) != len(baseline_pmf_ext) or len(bin_centers_ext) != len(current_pmf_ext):
            self.logger.error(
                "扩展后直方图的 bin 中心和 PMF 长度不匹配，无法计算 τ2 分数。请检查 baseline_bin_edges 和 baseline_counts 的计算。")
            return float('inf')

        try:
            tau2 = wasserstein_distance(
                bin_centers_ext, bin_centers_ext, u_weights=baseline_pmf_ext, v_weights=current_pmf_ext)
        except Exception as e:
            self.logger.error(f"wasserstein_distance 计算异常: {e}")
            return float('inf')

        return tau2

    @staticmethod
    def _calculate_tau3(current_batch_activations_sampled: np.ndarray, extreme_neurons_data: dict, sampled_neurons_indices: list) -> float:
        """
        计算 τ3 分数。

        Args:
            current_batch_activations_sampled (np.ndarray): 当前批次的激活值，形状通常为 (batch_size, num_neurons)。
            extreme_neurons_data (dict): 极端神经元的 Profiling 数据，包含每个神经元的统计信息。

        Returns:
            dict: 包含每个极端神经元的 τ3 分数。
        """
        tau3 = 0.0

        if not extreme_neurons_data or not sampled_neurons_indices or current_batch_activations_sampled.size == 0:
            return 0.0

        max_neuron_idx_baseline = extreme_neurons_data.get('max_neuron_idx')
        min_neuron_idx_baseline = extreme_neurons_data.get('min_neuron_idx')

        # 计算当前批次中采样神经元的平均激活值
        # current_batch_activations_sampled 的形状是 (batch_size, num_sampled_neurons)
        # 对 batch_size 维度求平均，得到 (num_sampled_neurons,)
        current_sampled_neuron_means = np.mean(
            current_batch_activations_sampled, axis=0)

        # 找到在“采样神经元”中，激活值最大和最小的神经元在**采样数组中的相对索引**
        current_max_sampled_idx_relative = np.argmax(
            current_sampled_neuron_means)
        current_min_sampled_idx_relative = np.argmin(
            current_sampled_neuron_means)

        # 将相对索引转换为原始层中的全局索引
        current_max_neuron_idx_global = sampled_neurons_indices[current_max_sampled_idx_relative]
        current_min_neuron_idx_global = sampled_neurons_indices[current_min_sampled_idx_relative]

        # 比较当前批次中（在采样神经元范围内）的极端神经元与Profiling时记录的全局极端神经元
        if max_neuron_idx_baseline is not None and current_max_neuron_idx_global == max_neuron_idx_baseline:
            tau3 += 0
        else:
            tau3 += 0.5

        if min_neuron_idx_baseline is not None and current_min_neuron_idx_global == min_neuron_idx_baseline:
            tau3 += 0
        else:
            tau3 += 0.5

        tau3 = min(tau3, 1.0)

        return tau3

    def _calculate_abnormality_scores(self, layer_name, current_batch_activations, layer_profiling_data):
        """
        根据当前批次的激活值和 Profiling 基准数据计算异常分数。

        Args:
            layer_name (str): 当前激活值所属的层名称。
            current_batch_activations (torch.Tensor): 当前批次该层的所有神经元的激活值，形状通常为 (batch_size, num_neurons)。
            layer_profiling_data (dict): 当前层在 Profiling 阶段收集的统计基准数据（已从整个 profiling_metadata 中提取）。
                                         预期包含单个神经元和 'overall' 的统计信息。

        Returns:
            dict: 包含该批次异常分数的字典，例如 {neuron_index: score} 或 {layer_name: overall_score}。
                  具体的结构和分数类型取决于后续的异常计算逻辑。
        """

        abnormality_scores_for_batch = {}

        if not layer_profiling_data:
            self.logger.error(f"层 '{layer_name}' 的 Profiling 数据为空，无法计算异常分数。")
            return abnormality_scores_for_batch

        sampled_neurons = self.profiling_sites.get(layer_name, [])
        if not sampled_neurons:
            self.logger.error(f"层 '{layer_name}' 没有采样神经元，无法计算异常分数。")
            return abnormality_scores_for_batch

        # 采样神经元现在是 (flat_idx, token_idx, feat_idx) 元组
        valid_sampled_neurons = [
            (flat_idx, token_idx, feat_idx)
            for (flat_idx, token_idx, feat_idx) in sampled_neurons
            if flat_idx < current_batch_activations.shape[1]
        ]
        if not valid_sampled_neurons:
            self.logger.warning(f"层 '{layer_name}' 的采样神经元索引均无效，无法计算异常分数。")
            return abnormality_scores_for_batch

        # 只选择采样神经元的激活值
        # current_batch_activations 的形状是 (batch_size, num_neurons)
        # 采样点的 flat_idx 用于索引列
        flat_indices = [flat_idx for (flat_idx, _, _) in valid_sampled_neurons]
        current_batch_activations_sampled = current_batch_activations[:, flat_indices]
        # 获取该层整体的直方图数据 (用于 τ2)
        overall_data = layer_profiling_data.get('overall', {})
        overall_bin_edges = overall_data.get('histogram_bin_edges')
        overall_counts = overall_data.get('histogram_counts')

        # 获取极端神经元数据 (用于 τ3)
        extreme_neurons_data = overall_data.get('extreme_neurons')

        # 初始化本层异常分数tau1
        tau1_sum = 0.0
        tau1_count = 0

        # 计算 τ1: 单个神经元的异常值得分
        for idx, (flat_idx, token_idx, feat_idx) in enumerate(valid_sampled_neurons):
            neuron_activations_in_batch_np = current_batch_activations_sampled[:, idx]
            neuron_baseline_data = layer_profiling_data.get(flat_idx)
            if neuron_baseline_data and 'histogram_bin_edges' in neuron_baseline_data and 'histogram_counts' in neuron_baseline_data:
                histogram_bin_edges = neuron_baseline_data['histogram_bin_edges']
                histogram_counts = neuron_baseline_data['histogram_counts']
                # print(f"边界: ({histogram_bin_edges[0]}, {histogram_bin_edges[-1]})")
                # for i in range(len(histogram_bin_edges) - 1):
                #     print(f"bin {i}: {histogram_bin_edges[i]} - {histogram_bin_edges[i+1]}, count: {histogram_counts[i]}")
                tau1_per_sample = []
                for sample_activation in neuron_activations_in_batch_np:
                    score = self._calculate_tau1_single_sample(
                        sample_activation,
                        histogram_bin_edges,
                        histogram_counts
                    )
                    tau1_per_sample.append(score)
                avg_tau1_for_neuron = np.mean(tau1_per_sample)
                abnormality_scores_for_batch[f'tau1_neuron_{flat_idx}'] = avg_tau1_for_neuron
                tau1_sum += avg_tau1_for_neuron
                tau1_count += 1
            else:
                self.logger.warning(
                    f"神经元 ({layer_name}, {flat_idx}) 缺少 Profiling 直方图数据，无法计算 τ1。")
                abnormality_scores_for_batch[f'tau1_neuron_{flat_idx}'] = float('nan')

        avg_tau1_overall = tau1_sum / tau1_count if tau1_count > 0 else 0.0
        abnormality_scores_for_batch['tau1_overall'] = avg_tau1_overall

        # 计算 τ2: 每层异常指标 (EMD)
        tau2 = 0.0
        if overall_bin_edges is not None and overall_counts is not None:
            tau2 = self._calculate_tau2(
                current_batch_activations_sampled, overall_bin_edges, overall_counts)
        else:
            self.logger.warning(f"层 '{layer_name}' 缺少整体直方图数据，无法计算 τ2。")
        abnormality_scores_for_batch['tau2'] = tau2

        # 计算 τ3: 极端值异常指标
        tau3 = 0.0
        if extreme_neurons_data:
            tau3 = self._calculate_tau3(
                current_batch_activations_sampled, extreme_neurons_data, flat_indices)
        else:
            self.logger.warning(f"层 '{layer_name}' 缺少极端神经元数据，无法计算 τ3。")
        abnormality_scores_for_batch['tau3'] = tau3

        # 根据公式计算当前层的总异常分数 τ_l
        # $T_l = T_{l-1) + 𝜆_1 * 𝜏_1 + 𝜆_2 * 𝜏_2+𝜆_3 * 𝜏_3$
        # 注意：这里的 T_l 是累积的，需要在 DnaModel 实例中维护
        # 而 _calculate_abnormality_scores 只计算当前层的 tau_l
        self.lambda1 = self.config["detecting"].get("lambda1", 1.0)
        self.lambda2 = self.config["detecting"].get("lambda2", 1.0)
        self.lambda3 = self.config["detecting"].get("lambda3", 1.0)
        tau_l_current_layer = (
            self.lambda1 * avg_tau1_overall +
            self.lambda2 * tau2 +
            self.lambda3 * tau3
        )
        abnormality_scores_for_batch["tau_l_current_layer"] = tau_l_current_layer

        # 更新累计总异常分数
        self.total_abnormality_score += tau_l_current_layer
        abnormality_scores_for_batch["total_abnormality_score"] = self.total_abnormality_score

        self.logger.debug(
            f"层 '{layer_name}' 异常分数计算完成: τ1={avg_tau1_overall:.4f}, τ2={tau2:.4f}, τ3={tau3:.4f}, 本层总异常分数={tau_l_current_layer:.4f}")
        return abnormality_scores_for_batch

    def _create_detecting_hook(self, layer_name, record_act_values=False):
        """
        创建用于异常检测的hook。
        如果record_act_values为True，则同时记录激活值到self.activations。
        """
        # 检查是否有 Profiling 数据
        if not self.profiling_data:
            self.logger.error("Profiling 数据未加载，无法创建异常检测hook。")

            def empty_hook(module, input, output):
                self.logger.error("Profiling 数据未加载，无法捕获激活值。")
            return empty_hook

        # 提取当前层的profiling信息
        layer_profiling_data = self.profiling_data.get(
            'layer_info', {}).get(layer_name)
        if not layer_profiling_data:
            self.logger.error(
                f"层 '{layer_name}' 的 Profiling 数据未找到，无法创建异常检测hook。")

            def empty_hook(module, input, output):
                self.logger.error(
                    f"层 '{layer_name}' 的 Profiling 数据未找到，无法捕获激活值。")
            return empty_hook

        def hook_fn(module, input, output):
            # 将激活值从 GPU 转移到 CPU，并转换为 numpy 数组
            current_batch_activations = output.detach().cpu().numpy()
            # 计算异常值
            if current_batch_activations.ndim > 2:
                current_batch_activations = current_batch_activations.reshape(
                    current_batch_activations.shape[0], -1)
            abnormality_scores = self._calculate_abnormality_scores(
                layer_name,
                current_batch_activations,
                layer_profiling_data
            )
            # 存储异常值
            self.abnormality_scores[layer_name].append(abnormality_scores)
            # 新增：如果需要记录激活值
            if record_act_values:
                self.activations[layer_name].append(current_batch_activations)

        return hook_fn

    def _register_detecting_hooks(self, record_act_values=False):
        """
        注册用于异常检测的hook。
        支持record_act_values参数，决定hook是否记录激活值。
        """
        self._remove_activation_hooks()  # 清理之前的hook
        self.logger.info("正在注册异常检测hook...")

        # 获取目标层名称
        if self.target_layer_names is None or len(self.target_layer_names) == 0:
            self.logger.error("没有有效的目标层进行异常检测。请先运行 Profiling 或检查配置。")
            return

        if record_act_values:
            self._infer_and_record_layer_output_shapes()

        # 注册hook
        for name, layer in self.model.named_modules():
            if name in self.target_layer_names and name in self.profiling_sites and self.profiling_sites[name]:
                hook_fn = self._create_detecting_hook(name, record_act_values=record_act_values)
                handle = layer.register_forward_hook(hook_fn)
                self.activation_hook_handles.append(handle)
                self.logger.debug(f"已为层 '{name}' 注册异常检测hook。")
            elif name in self.target_layer_names:
                self.logger.warning(f"层 '{name}' 在目标层列表中，但没有采样神经元，跳过hook注册。")
        self.logger.info(f"已成功注册 {len(self.activation_hook_handles)} 个异常检测hook。")

    def _load_profiling_metadata(self):
        """
        加载 Profiling 元数据。

        此方法从配置中获取 Profiling 元数据的路径，并加载相关信息。
        这些元数据通常包含关于模型层、采样神经元等的详细信息。

        Returns:
            dict: 包含 Profiling 元数据的字典。
        """
        metadata_dir = self.config["profiling"].get(
            "output_dir", "./profiling_results")
        timestamp_id = self.config["detecting"].get(
            "profiling_timestamp_id", "latest")

        if not os.path.exists(metadata_dir):
            self.logger.error(f"指定的 Profiling 元数据目录 '{metadata_dir}' 不存在。")
            return None
        else:
            if timestamp_id == "latest":
                timestamp_id = get_latest_timestamp(metadata_dir)
                if timestamp_id is None:
                    self.logger.error(
                        f"在目录 '{metadata_dir}' 中未找到任何 Profiling 元数据。")
                    return None
            metadata_path = os.path.join(
                metadata_dir, timestamp_id, 'profiling_metadata.pkl')
            self.config["detecting"]["profiling_timestamp_id"] = timestamp_id

            if not os.path.exists(metadata_path):
                self.logger.error(
                    f"指定的 Profiling 元数据文件 '{metadata_path}' 不存在。")
                return None
            try:
                import pickle
                with open(metadata_path, 'rb') as f:
                    profiling_data = pickle.load(f)
                self.logger.info(f"成功加载 Profiling 元数据: {metadata_path}")

                if 'layer_info' not in profiling_data:
                    self.logger.error(
                        "Profiling 元数据中缺少 'layer_info' 键。请检查 Profiling 过程是否正确。")
                    return None
                else:
                    self.target_layer_names = list(
                        profiling_data['layer_info'].keys())
                    self.profiling_sites = defaultdict(list)

                    # 新增：推理一次，获得每层的输出 shape
                    self._infer_and_record_layer_output_shapes()

                    for layer_name, layer_data in profiling_data['layer_info'].items():
                        profiled_neurons_in_layer = [
                            int(k) for k in layer_data.keys() if k != 'overall' and isinstance(k, (int, str)) and str(k).isdigit()
                        ]
                        shape = self.target_layer_output_shapes.get(layer_name, None)
                        if profiled_neurons_in_layer and shape is not None:
                            # shape: (token_num, out_feature) 或 (out_feature,)
                            if len(shape) == 1:
                                token_num, out_feature = 1, shape[0]
                            else:
                                token_num, out_feature = shape[-2], shape[-1]
                            tuples = [(idx, idx // out_feature, idx % out_feature) for idx in profiled_neurons_in_layer]
                            self.profiling_sites[layer_name] = tuples
                        elif profiled_neurons_in_layer:
                            # shape未知，保留flat index但警告
                            self.profiling_sites[layer_name] = [(idx, -1, -1) for idx in profiled_neurons_in_layer]
                            self.logger.warning(f"层 '{layer_name}' 未能获取输出shape，profiling_sites仅包含flat index，后续可能出错。")
                        else:
                            self.logger.warning(
                                f"层 '{layer_name}' 没有有效的采样神经元，跳过该层。")
                self.config['profiling']['num_layers_to_sample'] = len(self.profiling_sites[0]) if self.profiling_sites else 0
                return profiling_data
            except Exception as e:
                self.logger.error(f"加载 Profiling 元数据失败: {e}")
                return None

    def run_detecting_inference(self, record_act_values=False):
        """
        运行异常检测推理

        此方法在推理过程中实时计算每个批次的异常值和推理正确率。
        如果record_act_values为True，则还会记录采样神经元的激活值。
        模型将处于评估模式。
        """
        self.logger.info("正在运行异常检测推理...")

        # 清空之前的异常值
        self.abnormality_scores = defaultdict(list)
        self.activations.clear()  # 确保激活缓存清空
        # 重新设置随机种子
        # 因为有的进程可能需要重复执行多次detecting_inference
        # 比如对比无故障和故障注入情况下的结果
        # 为了对比的公平性，需要在这里显示地保证每次推理使用相同的随机种子
        set_random_seed(self.random_seed)

        # 加载存储的metadata
        self.profiling_data = self._load_profiling_metadata()
        if self.profiling_data is None:
            self.logger.error("无法加载 Profiling 元数据，异常检测将无法进行。")
            # 终止程序
            return

        # 注册异常检测hook
        self._register_detecting_hooks(record_act_values=record_act_values)
        if not self.activation_hook_handles:
            self.logger.error("没有有效的异常检测hook可供采样。请先运行 Profiling 或检查配置。")
            return

        # 获取数据加载器
        dataloader = self._get_dataloader()

        total_correct = 0
        total_samples = 0
        batch_accuracies = []

        # 新增：记录采样神经元激活值
        recorded_neuron_values = defaultdict(list) if record_act_values else None

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloader):
                self.total_abnormality_score = 0.0  # 重置总异常分数
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)

                # 计算当前 batch 的正确率
                if outputs.dim() > 1 and outputs.size(1) > 1:
                    preds = torch.argmax(outputs, dim=1)
                else:
                    preds = (outputs > 0.5).long().view(-1)
                correct = (preds == labels).sum().item()
                batch_size = labels.size(0)
                total_correct += correct
                total_samples += batch_size
                batch_acc = correct / batch_size if batch_size > 0 else 0.0
                cum_acc = total_correct / total_samples if total_samples > 0 else 0.0
                batch_accuracies.append(batch_acc)
                self.logger.debug(
                    f"Batch {i+1}: 当前batch正确率={batch_acc:.4f}, 累计正确率={cum_acc:.4f}")

                # 记录采样神经元激活值（仿照run_profiling_inference）
                if record_act_values and self.activations:
                    for layer_name, activation_list in self.activations.items():
                        if activation_list:
                            current_batch_activations = activation_list.pop(0)
                            shape = self.target_layer_output_shapes.get(layer_name, None)
                            if shape is None:
                                self.logger.warning(
                                    f"未找到层 '{layer_name}' 的输出shape，跳过采样。")
                                continue
                            if len(shape) == 1:
                                token_num, out_feature = 1, shape[0]
                            else:
                                token_num, out_feature = shape[-2], shape[-1]
                            for flat_idx, token_idx, feat_idx in self.profiling_sites[layer_name]:
                                try:
                                    if current_batch_activations.ndim == 2:
                                        values = current_batch_activations[:, feat_idx]
                                    elif current_batch_activations.ndim >= 3:
                                        values = current_batch_activations[:, token_idx, feat_idx]
                                    else:
                                        self.logger.warning(
                                            f"激活shape异常: {current_batch_activations.shape}")
                                        continue
                                    recorded_neuron_values[(layer_name, flat_idx, token_idx, feat_idx)].append(values)
                                except Exception as e:
                                    self.logger.warning(
                                        f"采样点(flat_idx={flat_idx}, token={token_idx}, feat={feat_idx}) 采集激活异常: {e}")
                    self.activations.clear()
                else:
                    self.activations.clear()

                if (i + 1) % self.config["data_loader"].get("log_interval", 10) == 0:
                    self.logger.info(f"已处理 {i + 1}/{len(dataloader)} 批次。")
                if i + 1 >= self.config["data_loader"].get("iteration_limit", -1) > 0:
                    self.logger.info(
                        f"已达到迭代限制 {self.config['data_loader']['iteration_limit']}，停止异常检测。")
                    break

        self._remove_activation_hooks()  # 清理hook
        self.logger.info("Detecting 推理完成。最终累计正确率: {:.4f}".format(
            total_correct / total_samples if total_samples > 0 else 0.0))
        result = {
            'abnormality_scores': self.abnormality_scores,
            'batch_accuracies': batch_accuracies,
            'final_accuracy': total_correct / total_samples if total_samples > 0 else 0.0
        }
        if record_act_values:
            result['recorded_neuron_values'] = recorded_neuron_values

        return result

    def get_config(self):
        return self.config

    def clear_internal_state(self):
        """
        清理模型推理相关的所有中间状态，包括：
        - 激活缓存
        - profiling_sites
        - target_layer_names
        - abnormality_scores
        - profiling_data
        - 累计异常分数
        - 已注册的hook
        """
        self.activations.clear()
        self.profiling_sites.clear()
        self.target_layer_names = []
        if hasattr(self, 'abnormality_scores'):
            self.abnormality_scores.clear()
        if hasattr(self, 'profiling_data'):
            self.profiling_data = None
        self.total_abnormality_score = 0.0
        self._remove_activation_hooks()
        self.logger.info("已清理模型的中间状态（缓存、hook、累计分数等）")

    def _get_dataloader(self):
        """
        获取指定分割类型的数据加载器。
        """
        return self.data_module.get_dataloader()

    def _infer_and_record_layer_output_shapes(self):
        """
        用一小批数据前向推理，记录目标层的输出shape（不含batch维）。
        """
        self.target_layer_output_shapes = {}
        hooks = []

        def make_shape_hook(name):
            def hook(module, input, output):
                # output shape: (batch, ...)
                self.target_layer_output_shapes[name] = tuple(output.shape[1:])
            return hook
        for name in self.target_layer_names:
            layer = dict(self.model.named_modules()).get(name, None)
            if layer is not None:
                hooks.append(layer.register_forward_hook(
                    make_shape_hook(name)))
        dataloader = self._get_dataloader()
        self.model.eval()
        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(self.device)
                _ = self.model(inputs)
                break  # 只需一批
        for h in hooks:
            h.remove()
