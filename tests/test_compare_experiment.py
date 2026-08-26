import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from compare_experiment import (
    ActivationTrace,
    DrDNAConfig,
    DrDNAProfile,
    DrDNAProfiler,
    OnlineActivationMonitor,
    RangeProfile,
    RangeProfiler,
    evaluate_detection,
    threshold_at_fpr,
)
from compare_experiment.cohorts import load_comparison_cohorts
from compare_experiment.config import load_comparison_config
from compare_experiment.evaluation import apply_threshold


def _trace(values_by_step):
    return ActivationTrace(
        vectors={
            step: {
                layer: np.asarray(values, dtype=np.float32)
                for layer, values in layers.items()
            }
            for step, layers in values_by_step.items()
        },
        max_steps=2,
    )


class RangeProfileTest(unittest.TestCase):
    def test_profiles_clean_bounds_and_scores_excursions(self):
        profiler = RangeProfiler(monitored_layers=(1, 2), max_steps=2)
        profiler.add_trace(
            _trace({0: {1: [0.0, 1.0], 2: [-1.0, 2.0]}})
        )
        profiler.add_trace(
            _trace({0: {1: [-0.5, 0.5], 2: [0.0, 1.5]}})
        )
        profile = profiler.finalize()

        self.assertEqual(
            profile.score(_trace({0: {1: [0.2], 2: [1.0]}})),
            0.0,
        )
        self.assertGreater(
            profile.score(_trace({0: {1: [3.0], 2: [1.0]}})),
            0.0,
        )
        self.assertTrue(
            np.isposinf(
                profile.score(
                    _trace({0: {1: [np.nan], 2: [1.0]}})
                )
            )
        )

    def test_range_profile_round_trip(self):
        original = RangeProfile(
            monitored_layers=(1,),
            max_steps=2,
            lower={(0, 1): -2.0},
            upper={(0, 1): 3.0},
        )
        restored = RangeProfile.from_dict(original.to_dict())
        self.assertEqual(restored.monitored_layers, (1,))
        self.assertEqual(restored.lower[(0, 1)], -2.0)
        self.assertEqual(restored.upper[(0, 1)], 3.0)


class DrDNAProfileTest(unittest.TestCase):
    def test_fault_distribution_scores_above_clean_distribution(self):
        profiler = DrDNAProfiler(
            monitored_layers=(1, 2, 3),
            max_steps=1,
            config=DrDNAConfig(
                cohort_size=4,
                bins=4,
                strike_count=2,
                random_seed=7,
            ),
        )
        for offset in (0.0, 0.1, -0.1, 0.05, -0.05):
            profiler.add_trace(
                _trace(
                    {
                        0: {
                            1: [0.0 + offset, 1.0, 2.0, 3.0],
                            2: [0.5 + offset, 1.5, 2.5, 3.5],
                            3: [1.0 + offset, 2.0, 3.0, 4.0],
                        }
                    }
                )
            )
        profile = profiler.finalize()
        clean_score = profile.score(
            _trace(
                {
                    0: {
                        1: [0.02, 1.0, 2.0, 3.0],
                        2: [0.52, 1.5, 2.5, 3.5],
                        3: [1.02, 2.0, 3.0, 4.0],
                    }
                }
            )
        )
        fault_score = profile.score(
            _trace(
                {
                    0: {
                        1: [100.0, -100.0, 50.0, -50.0],
                        2: [100.0, -100.0, 50.0, -50.0],
                        3: [100.0, -100.0, 50.0, -50.0],
                    }
                }
            )
        )

        self.assertGreater(fault_score, clean_score)
        restored = DrDNAProfile.from_dict(profile.to_dict())
        self.assertAlmostEqual(restored.score(
            _trace(
                {
                    0: {
                        1: [100.0, -100.0, 50.0, -50.0],
                        2: [100.0, -100.0, 50.0, -50.0],
                        3: [100.0, -100.0, 50.0, -50.0],
                    }
                }
            )
        ), fault_score)


class EvaluationTest(unittest.TestCase):
    def test_threshold_respects_false_positive_budget(self):
        calibration = threshold_at_fpr(
            [0.0, 0.1, 0.2, 0.3, 0.4] * 20,
            target_fpr=0.1,
        )
        detected = apply_threshold(
            [0.0, 0.1, 0.2, 0.3, 0.4] * 20,
            calibration.threshold,
        )
        self.assertLessEqual(float(detected.mean()), 0.1)

    def test_metrics_separate_sdc_and_significant_sdc(self):
        metrics = evaluate_detection(
            is_sdc=[0, 1, 1, 1],
            is_significant_sdc=[0, 0, 1, 1],
            detected=[1, 1, 1, 0],
        )
        self.assertEqual(metrics.sdc_recall, 2 / 3)
        self.assertEqual(metrics.significant_sdc_recall, 0.5)
        self.assertEqual(metrics.non_sdc_fpr, 1.0)
        self.assertEqual(metrics.significant_sdc_precision, 1 / 3)

    def test_non_finite_fast_path_reports_infeasible_fpr_budget(self):
        calibration = threshold_at_fpr(
            [float("inf"), 0.0, 0.1, 0.2],
            target_fpr=0.1,
        )
        self.assertFalse(calibration.budget_feasible)
        self.assertEqual(calibration.achieved_fpr, 0.25)


class ComparisonConfigurationTest(unittest.TestCase):
    def test_loads_repository_comparison_config(self):
        root = Path(__file__).resolve().parents[1]
        config = load_comparison_config(
            root
            / "compare_experiment/configs/detection_comparison.yaml",
            repository_root=root,
        )
        self.assertEqual(config.max_steps, 2)
        self.assertEqual(config.monitored_layers, (6, 7, 22, 23, 24, 25, 26, 27))
        self.assertEqual(config.drdna.cohort_size, 64)
        self.assertEqual(config.drdna.bins, 10)

    def test_cohort_split_is_deterministic_and_leakage_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.csv"
            valid = root / "valid.csv"
            pd.DataFrame(
                {"orig_id": [f"train-{index}" for index in range(10)]}
            ).to_csv(train, index=False)
            pd.DataFrame(
                {"orig_id": [f"test-{index}" for index in range(3)]}
            ).to_csv(valid, index=False)
            first = load_comparison_cohorts(
                train,
                valid,
                calibration_ratio=0.2,
                random_seed=42,
            )
            second = load_comparison_cohorts(
                train,
                valid,
                calibration_ratio=0.2,
                random_seed=42,
            )
        self.assertEqual(first, second)
        self.assertEqual(len(first.fit_orig_ids), 8)
        self.assertEqual(len(first.calibration_orig_ids), 2)
        self.assertEqual(len(first.test_orig_ids), 3)


class _Attention(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.o_proj = nn.Linear(width, width, bias=False)
        with torch.no_grad():
            self.o_proj.weight.copy_(torch.eye(width))


class _Layer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.self_attn = _Attention(width)


class _Inner(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = nn.ModuleList([_Layer(width), _Layer(width)])


class _FakeModel(nn.Module):
    def __init__(self, width=4):
        super().__init__()
        self.model = _Inner(width)
        self.lm_head = nn.Linear(width, width, bias=False)

    def run_step(self, values):
        output = values
        for layer in self.model.layers:
            output = layer.self_attn.o_proj(output)
        return self.lm_head(output)


class OnlineActivationMonitorTest(unittest.TestCase):
    def test_collects_decode_steps_without_changing_outputs(self):
        model = _FakeModel()
        monitor = OnlineActivationMonitor(
            model,
            monitored_layers=(0, 1),
            max_steps=2,
        )
        values = torch.arange(4, dtype=torch.float32).reshape(1, 1, 4)
        monitor.register()
        try:
            monitor.start_sample()
            prefill = model.run_step(values)
            first = model.run_step(values + 1)
            second = model.run_step(values + 2)
            trace = monitor.finish_sample()
        finally:
            monitor.unregister()

        self.assertEqual(trace.steps, (0, 1))
        np.testing.assert_allclose(trace.vector(0, 0), [1, 2, 3, 4])
        self.assertTrue(torch.isfinite(prefill).all())
        self.assertTrue(torch.isfinite(first).all())
        self.assertTrue(torch.isfinite(second).all())


if __name__ == "__main__":
    unittest.main()
