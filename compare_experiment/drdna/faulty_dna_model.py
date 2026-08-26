import logging
from typing import List, Dict, Any, Optional
import torch
from utils.help_func import set_random_seed

from models.dna_model import DnaModel

import sys
sys.path.append('/mnt/llm_eval_fi/lm-evaluation-harness')
from eval_fi.fault_injector import FaultInjector

class FaultyDnaModel(DnaModel):
    def __init__(self, config):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)

        # 初始化故障注入器
        if not config.get('fault_config', {}):
            self.logger.error("配置中缺少 'fault_config' 字段")
            raise ValueError("配置中缺少 'fault_config' 字段")
        self.fault_config = config['fault_config']

        self.fault_injector = FaultInjector(self.config)
        self.fi_hooks: List[torch.utils.hooks.RemovableHandle] = []

    def _inject_faults(self):
        if self.fault_config.get("enabled", False):
            self.logger.info("正在执行故障注入")
            # 重新设置随机种子
            set_random_seed(self.random_seed)
            # 先注入权重故障
            self.fault_injector.inject_weight_faults_only(self.model)
            # 注入运行时故障
            runtime_handles = self.fault_injector.create_runtime_hooks(self.model, "")
            self.logger.info(f"已注入 {len(runtime_handles)} 个运行时故障fi hook")
            self.fi_hooks.extend(runtime_handles)
        else:
            self.logger.info("故障注入未启用，跳过注入步骤")

    def run_fi_detecting_inference(self, fi_enabled=True, record_act_values=False):
        # 重新初始化随机种子
        set_random_seed(self.random_seed)
        # 注入故障
        if fi_enabled:
            self._inject_faults()
        # 执行推理并计算异常值和正确率
        result = self.run_detecting_inference(record_act_values)
        # 清理hooks
        self._remove_fi_hooks()
        # 返回异常值和正确率等指标
        return result

    def _remove_fi_hooks(self):
        self.fi_hooks.clear()
        self.logger.info("fi hook 已清理")

    def clear_internal_state(self):
        """
        清空模型的内部状态，包括hook、缓存、累计分数等。
        这在每次运行检测推理前都需要调用，以确保状态干净。
        """
        # 调用父类方法清理模型状态
        super().clear_internal_state()
        # 清理故障注入相关的hooks
        self._remove_fi_hooks()
        self.logger.info("已清空模型内部状态")

if __name__ == "__main__":
    pass