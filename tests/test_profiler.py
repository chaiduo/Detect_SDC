import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from detect_sdc.profiler import Profiler


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.o_proj = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.o_proj.weight.copy_(torch.eye(4))

    def forward(self, values):
        return self.o_proj(values)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attention()

    def forward(self, values):
        return self.self_attn(values)


class _LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Layer(), _Layer()])


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = _LanguageModel()
        self.lm_head = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(torch.eye(4))

    def forward(self, values):
        for layer in self.language_model.layers:
            values = layer(values)
        return self.lm_head(values)


class _IdentityMapping(nn.Module):
    def forward(self, values, source_layer, target_layer, step=None):
        return values


class ProfilerTest(unittest.TestCase):
    def test_active_mapping_and_telemetry_paths(self):
        model = _TinyModel()
        profiler = Profiler(model, proj_dim=2, proj_method="mean")
        profiler.register()

        values = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        model(values)  # Prefill establishes the decode-step boundary.
        model(values)
        profiler.finalize()

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "mapping.jsonl"
            rows_written = profiler.save_attn_proj_interlayer_jsonl(
                str(output),
                sample_id=7,
            )
            rows = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]

        telemetry = profiler.get_attn_proj_model_compare_result(
            _IdentityMapping(),
            device="cpu",
            include_vectors=False,
        )
        profiler.unregister()

        self.assertEqual(rows_written, 1)
        self.assertEqual(rows[0]["sample_id"], 7)
        self.assertEqual(rows[0]["x"], [1.5, 3.5])
        self.assertEqual(rows[0]["y"], [1.5, 3.5])
        self.assertEqual(telemetry["num_steps"], 1)
        self.assertEqual(telemetry["num_layer_pairs"], 1)
        self.assertEqual(telemetry["records"][0]["l2_distance"], 0.0)
        self.assertEqual(profiler.handles, [])


if __name__ == "__main__":
    unittest.main()
