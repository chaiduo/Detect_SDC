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
        self.assertEqual(
            injector.fault_info["bit_categories"],
            ["mantissa", "mantissa"],
        )
        self.assertEqual(injector.fault_info["bit_policy"], "random")

    def test_mantissa_only_policy_never_selects_exponent_or_sign_bits(self):
        layouts = {
            torch.float16: set(range(10)),
            torch.bfloat16: set(range(7)),
            torch.float32: set(range(23)),
        }
        for dtype, allowed in layouts.items():
            with self.subTest(dtype=dtype):
                injector = FaultInjector(_TinyModel())
                injector.set_bit_policy("mantissa_only")
                injector.set_num_bits(2)

                injector.random_bitflip(
                    torch.tensor([1.0], dtype=dtype),
                )

                self.assertTrue(
                    set(injector.fault_info["bit_positions"]) <= allowed
                )
                self.assertEqual(
                    injector.fault_info["bit_categories"],
                    ["mantissa", "mantissa"],
                )
                self.assertEqual(
                    injector.fault_info["bit_policy"],
                    "mantissa_only",
                )

    def test_low_mantissa_policy_uses_only_low_order_bits(self):
        layouts = {
            torch.float16: set(range(5)),
            torch.bfloat16: set(range(4)),
            torch.float32: set(range(11)),
        }
        for dtype, allowed in layouts.items():
            with self.subTest(dtype=dtype):
                injector = FaultInjector(_TinyModel())
                injector.set_bit_policy("low_mantissa")
                injector.set_num_bits(2)

                injector.random_bitflip(
                    torch.tensor([1.0], dtype=dtype),
                )

                self.assertTrue(
                    set(injector.fault_info["bit_positions"]) <= allowed
                )
                self.assertEqual(
                    injector.fault_info["bit_policy"],
                    "low_mantissa",
                )

    def test_low_exponent_policy_includes_mantissa_and_five_exponent_bits(self):
        layouts = {
            torch.float16: set(range(15)),
            torch.bfloat16: set(range(12)),
            torch.float32: set(range(28)),
        }
        for dtype, allowed in layouts.items():
            with self.subTest(dtype=dtype):
                injector = FaultInjector(_TinyModel())
                injector.set_bit_policy("low_exponent")
                injector.set_num_bits(2)

                self.assertEqual(
                    set(injector._candidate_bit_positions(dtype)),
                    allowed,
                )
                injector.random_bitflip(
                    torch.tensor([1.0], dtype=dtype),
                )

                self.assertTrue(
                    set(injector.fault_info["bit_positions"]) <= allowed
                )
                self.assertTrue(
                    set(injector.fault_info["bit_categories"])
                    <= {"mantissa", "exponent"}
                )
                self.assertEqual(
                    injector.fault_info["bit_policy"],
                    "low_exponent",
                )

    def test_targeted_policy_rejects_integer_tensors(self):
        injector = FaultInjector(_TinyModel())
        injector.set_bit_policy("mantissa_only")

        with self.assertRaisesRegex(ValueError, "requires a floating dtype"):
            injector.random_bitflip(torch.tensor([1], dtype=torch.int32))

    def test_rejects_unknown_bit_policy(self):
        injector = FaultInjector(_TinyModel())

        with self.assertRaisesRegex(ValueError, "bit_policy must be one of"):
            injector.set_bit_policy("exponent_only")

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

    def test_lm_head_fault_precedes_step_counter(self):
        model = _TinyModel()
        injector = FaultInjector(model, mode="activation")
        injector.set_inject_info(
            idx=0,
            module_name="lm_head",
            inject_step=0,
            bit_positions=[22],
        )
        injector.inject()
        injector.register_step_hooks()

        output = model(torch.tensor([[1.0, 2.0]]))

        self.assertTrue(injector.select_target_has_injected)
        self.assertNotEqual(output[0, 0].item(), 1.0)
        self.assertEqual(injector.current_step, 1)
        self.assertEqual(injector.fault_info["component"], "lm_head")
        self.assertEqual(injector.fault_info["op_type"], "lm_head")
        injector.unregister_hooks()

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
