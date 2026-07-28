import unittest

import numpy as np
import torch
from torch import nn

from detect_sdc.fault_injector import FaultInjector


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 2, bias=False)
        self.lm_head = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.hidden.weight.copy_(torch.eye(2))
            self.lm_head.weight.copy_(torch.eye(2))

    def forward(self, values):
        return self.lm_head(self.hidden(values))


class FaultInjectorTest(unittest.TestCase):
    def test_float32_flip_matches_raw_xor_and_deduplicates_bits(self):
        injector = FaultInjector(_TinyModel())
        value = torch.tensor([1.0], dtype=torch.float32)

        flipped = injector.flip_bits(value, 0, [0, 0, 2])

        raw = np.asarray([1.0], dtype=np.float32).view(np.uint32)
        expected = (raw ^ np.uint32((1 << 0) | (1 << 2))).view(np.float32)
        self.assertEqual(flipped.item(), float(expected[0]))
        self.assertEqual(injector.fault_info["bit_positions"], [0, 2])

    def test_activation_fault_is_injected_once_and_hooks_are_released(self):
        model = _TinyModel()
        injector = FaultInjector(model, mode="activation")
        injector.set_inject_info(
            idx=0,
            module_name="hidden",
            inject_step=0,
            bit_positions=[22],
        )
        injector.register_step_hooks()
        injector.inject()

        first = model(torch.tensor([[1.0, 2.0]]))
        second = model(torch.tensor([[1.0, 2.0]]))

        self.assertTrue(injector.select_target_has_injected)
        self.assertNotEqual(first[0, 0].item(), 1.0)
        self.assertEqual(second.tolist(), [[1.0, 2.0]])
        injector.unregister_hooks()
        self.assertIsNone(injector.fault_hook_handle)
        self.assertIsNone(injector.step_hook_handle)

    def test_weight_fault_can_be_restored_exactly(self):
        model = _TinyModel()
        injector = FaultInjector(model, mode="weight")
        injector.set_inject_info(
            idx=0,
            module_name="hidden",
            bit_positions=[22],
        )
        original = model.hidden.weight.detach().clone()

        injector.inject()
        self.assertFalse(torch.equal(model.hidden.weight, original))

        injector.restore_weight_fault()
        self.assertTrue(torch.equal(model.hidden.weight, original))
        self.assertFalse(injector.weight_fault_injected)


if __name__ == "__main__":
    unittest.main()
