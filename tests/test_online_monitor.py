from __future__ import annotations

import unittest

import numpy as np
import torch

from detect_sdc.online_monitor import OnlineSieveMonitor


PAIRS = ((6, 7), (22, 23))


def make_monitor(feature_profile: str = "full") -> OnlineSieveMonitor:
    return OnlineSieveMonitor(
        object(),
        mode="monitor",
        layer_pairs=PAIRS,
        projection_dim=64,
        projection_seed=42,
        max_steps=2,
        feature_profile=feature_profile,
    )


class OnlineSieveMonitorTests(unittest.TestCase):
    def test_full_profile_remains_default(self) -> None:
        monitor = make_monitor()
        self.assertEqual(len(monitor.expected_feature_columns()), 24)
        self.assertEqual(
            monitor.expected_feature_columns()[:3],
            (
                "cos_sim_mean_p6_7",
                "cos_sim_max_p6_7",
                "cos_sim_min_p6_7",
            ),
        )

    def test_cosine_mean_profile_has_one_feature_per_pair(self) -> None:
        monitor = make_monitor("cos_sim_mean")
        self.assertEqual(
            monitor.expected_feature_columns(),
            (
                "cos_sim_mean_p6_7",
                "cos_sim_mean_p22_23",
            ),
        )

    def test_cosine_mean_profile_uses_finite_only_step_mean(self) -> None:
        monitor = make_monitor("cos_sim_mean")
        features = monitor._aggregate_features(
            (
                torch.tensor([0.8, float("nan")]),
                torch.tensor([1.0, 0.6]),
            )
        )
        torch.testing.assert_close(features, torch.tensor([0.9, 0.6]))

    def test_sieve_uses_calibrated_threshold(self) -> None:
        class Detector:
            def predict_proba(self, _values):
                return np.asarray([[0.3, 0.7]])

        monitor = OnlineSieveMonitor(
            object(),
            mode="sieve",
            layer_pairs=((6, 7),),
            projection_dim=64,
            projection_seed=42,
            max_steps=1,
            predictor=object(),
            detector=Detector(),
            detector_threshold=0.8,
        )
        monitor._step_metrics = [torch.tensor([[0.9, 0.1, 0.2, 0.3]])]
        monitor._complete_prediction()

        self.assertEqual(monitor.detector_probability, 0.7)
        self.assertEqual(monitor.detector_prediction, 0)

    def test_rejects_unknown_feature_profile(self) -> None:
        with self.assertRaisesRegex(ValueError, "feature profile"):
            make_monitor("unknown")


if __name__ == "__main__":
    unittest.main()
