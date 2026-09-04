import unittest

from scripts.evaluate_deviation_buckets import binary_metrics, deviation_bucket


class DeviationBucketEvaluationTest(unittest.TestCase):
    def test_deviation_bucket_boundaries(self):
        cases = (
            (1.0, 1.0, "zero"),
            (0.0, 0.5, "(0,1]"),
            (0.0, 1.0, "(0,1]"),
            (0.0, 1.0001, "(1,1e6]"),
            (0.0, 1e6, "(1,1e6]"),
            (0.0, 1e12, "(1e6,1e12]"),
            (0.0, 1e18, "(1e12,1e18]"),
            (0.0, 1e24, "(1e18,1e24]"),
            (0.0, 1e30, "(1e24,1e30]"),
            (0.0, 1e36, "(1e30,1e36]"),
            (0.0, 1e37, ">1e36"),
            (0.0, float("inf"), "non_finite"),
        )
        for before, after, expected in cases:
            with self.subTest(before=before, after=after):
                self.assertEqual(deviation_bucket(before, after), expected)

    def test_binary_metrics_use_all_non_significant_rows_as_negatives(self):
        metrics = binary_metrics(
            significant=[0, 0, 1, 1],
            detected=[0, 1, 1, 0],
        )

        self.assertEqual(metrics["true_positive"], 1)
        self.assertEqual(metrics["false_positive"], 1)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["f1"], 0.5)
        self.assertEqual(metrics["fpr"], 0.5)


if __name__ == "__main__":
    unittest.main()
