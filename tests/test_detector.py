import unittest

import numpy as np
import pandas as pd

from detect_sdc.detector import (
    XGBoostConfig,
    add_significant_sdc_target,
    binary_metrics,
    calibrate_threshold_at_fpr,
    calibrate_threshold_max_f1,
    get_feature_columns,
    prepare_features,
    significant_sdc_negative_mask,
)


class XGBoostDetectorTest(unittest.TestCase):
    def test_ternary_and_legacy_binary_labels_share_one_target(self):
        ternary = pd.DataFrame(
            {
                "label": [0, 1, 2],
                "significance": [0, 1, 2],
            }
        )
        binary = pd.DataFrame(
            {
                "label": [0, 1, 1],
                "significance": [0, 1, 2],
            }
        )

        self.assertEqual(
            add_significant_sdc_target(ternary)["significant_sdc_target"].tolist(),
            [0, 0, 1],
        )
        self.assertEqual(
            add_significant_sdc_target(binary)["significant_sdc_target"].tolist(),
            [0, 0, 1],
        )

    def test_inconsistent_explicit_target_is_rejected(self):
        frame = pd.DataFrame(
            {
                "label": [0, 2],
                "significance": [0, 2],
                "significant_sdc_target": [0, 0],
            }
        )

        with self.assertRaisesRegex(ValueError, "inconsistent"):
            add_significant_sdc_target(frame)

    def test_explicit_target_disambiguates_ternary_subset_without_class_two(self):
        frame = pd.DataFrame(
            {
                "label": [0, 1],
                "significance": [0, 2],
                "significant_sdc_target": [0, 0],
            }
        )

        result = add_significant_sdc_target(frame)

        self.assertEqual(result["significant_sdc_target"].tolist(), [0, 0])

    def test_negative_mask_includes_every_non_significant_execution(self):
        frame = pd.DataFrame(
            {
                "significant_sdc_target": [0, 0, 0, 1],
                "injected": [0, 1, 1, 1],
                "is_sdc": [0, 0, 1, 1],
            }
        )

        self.assertEqual(
            significant_sdc_negative_mask(frame).tolist(),
            [True, True, True, False],
        )

    def test_metadata_and_targets_are_not_features(self):
        frame = pd.DataFrame(
            {
                "orig_id": ["a"],
                "sample_uid": ["a-1"],
                "total_steps": [2],
                "significance": [2],
                "label": [2],
                "significant_sdc_target": [1],
                "cos_sim_mean_p1_2": [float("inf")],
            }
        )

        columns = get_feature_columns(frame)
        features = prepare_features(frame, columns)

        self.assertEqual(columns, ["cos_sim_mean_p1_2"])
        self.assertTrue(features.isna().iloc[0, 0])

    def test_binary_metrics_use_significant_class_as_target(self):
        metrics = binary_metrics(
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            np.asarray(
                [
                    [0.9, 0.1],
                    [0.4, 0.6],
                    [0.2, 0.8],
                    [0.1, 0.9],
                ]
            ),
        )

        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [0, 2]])
        self.assertEqual(metrics["target_significant_sdc"]["tp"], 2)
        self.assertEqual(metrics["target_significant_sdc"]["fp"], 1)

    def test_unknown_xgboost_parameter_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown"):
            XGBoostConfig.from_mapping({"unknown": 1})

    def test_threshold_calibration_respects_strict_fpr_budget(self):
        calibration = calibrate_threshold_at_fpr(
            [0.1, 0.2, 0.3, 0.4, 0.5] * 20,
            target_fpr=0.01,
        )

        self.assertLessEqual(calibration["achieved_fpr"], 0.01)
        self.assertEqual(
            calibration["comparison"],
            "positive_probability > threshold",
        )

    def test_threshold_calibration_maximizes_f1(self):
        calibration = calibrate_threshold_max_f1(
            [0.1, 0.2, 0.8, 0.9],
            [0, 0, 1, 1],
        )

        self.assertEqual(calibration["strategy"], "maximize_f1")
        self.assertEqual(calibration["threshold"], 0.2)
        self.assertEqual(calibration["calibration_f1"], 1.0)
        self.assertEqual(calibration["calibration_fpr"], 0.0)

    def test_max_f1_tie_uses_highest_threshold(self):
        calibration = calibrate_threshold_max_f1(
            [0.6, 0.7, 0.8, 0.9],
            [1, 0, 0, 1],
        )

        self.assertEqual(calibration["threshold"], 0.8)
        self.assertEqual(calibration["tie_breaker"], "highest_threshold")


if __name__ == "__main__":
    unittest.main()
