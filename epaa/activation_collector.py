import torch
import numpy as np
from torch import nn
from collections import defaultdict

class ActivationCollector:
    def __init__(self, model):
        self.model = model
        self.handles = []
        self.activations = defaultdict(dict)

        self.current_step = 0
        self._step_hook_handle = None

    def register(self):
        layers = self._get_layers("language_model")
        # layers = self.model.model.layers
        # merger = getattr(self.model.model.visual, "merger", None)
        # h4 = merger.mlp[2].register_forward_hook(self._make_hook(0, "merger.mlp.2"))
        # self.handles.append(h4)
        # self.model.model.language_model.norm.register_forward_hook(self._make_hook(0, "norm"))

        for layer_idx in (26, 27):
            h = layers[layer_idx].mlp.down_proj.register_forward_hook(
                self._make_hook(layer_idx, "mlp.down_proj")
            )
            self.handles.append(h)
        # for layer_idx, layer in enumerate(layers):
        #     # h4 = layer.self_attn.o_proj.register_forward_hook(self._make_hook(layer_idx, "self_attn.o_proj"))
        #     #h4 = layer.input_layernorm.register_forward_hook(self._make_hook(layer_idx, "input_layernorm"))
        #     h4 = layer.mlp.down_proj.register_forward_hook(self._make_hook(layer_idx, "mlp.down_proj"))
        #     self.handles.append(h4)

        self._step_hook_handle = self.model.lm_head.register_forward_hook(self._make_step_counter_hook())


    def _make_step_counter_hook(self):
        def hook(module, input, output):
            self.current_step += 1
            return output
        return hook
    
    def _get_layers(self, name="language_model"):
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

    def _make_hook(self, layer_idx, name):
        def hook(m, inp, out):
            out = out.detach().cpu().to(torch.float32)
            if out.dim() == 3:
                vec = out[0, -1, :]  # batch=0, last token
            elif out.dim() == 2:
                vec = out[-1, :]
            else:
                vec = out
            layer_key = f"L{layer_idx}.{name}"
            self.activations[self.current_step][layer_key] = vec.unsqueeze(0)  # 保持 shape 一致
        return hook

    def reset(self):
        self.current_step = 0
        self.activations = defaultdict(dict)

    def unregister(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        self.current_step = 0
        self._step_hook_handle.remove()

