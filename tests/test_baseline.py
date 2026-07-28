import csv
import tempfile
import unittest
from pathlib import Path

from detect_sdc.baseline import hash_and_count_nonempty_lines, inspect_feature_csv


class BaselineInspectionTest(unittest.TestCase):
    def test_jsonl_hash_and_row_count_are_computed_together(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.jsonl"
            path.write_bytes(b'{"id": 1}\n\n{"id": 2}\n')

            digest, rows = hash_and_count_nonempty_lines(path)

            self.assertEqual(rows, 2)
            self.assertEqual(len(digest), 64)

    def test_binary_and_ternary_legacy_labels_map_to_same_target(self):
        with tempfile.TemporaryDirectory() as directory:
            binary_path = Path(directory) / "binary.csv"
            with binary_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=["orig_id", "sample_uid", "label", "significance"],
                )
                writer.writeheader()
                writer.writerow({"orig_id": "1", "sample_uid": "1_a", "label": 1, "significance": 2})
                writer.writerow({"orig_id": "2", "sample_uid": "2_a", "label": 1, "significance": 0})

            ternary_path = Path(directory) / "ternary.csv"
            with ternary_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=["orig_id", "sample_uid", "label", "significance"],
                )
                writer.writeheader()
                writer.writerow({"orig_id": "1", "sample_uid": "1_a", "label": 2, "significance": 2})
                writer.writerow({"orig_id": "2", "sample_uid": "2_a", "label": 1, "significance": 0})

            binary_summary = inspect_feature_csv(binary_path)
            ternary_summary = inspect_feature_csv(ternary_path)

            expected = {"0": 1, "1": 1}
            self.assertEqual(binary_summary["significant_sdc_target_counts"], expected)
            self.assertEqual(ternary_summary["significant_sdc_target_counts"], expected)

    def test_ternary_target_is_inferred_from_the_complete_label_set(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ternary.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=["orig_id", "sample_uid", "label", "significance"],
                )
                writer.writeheader()
                writer.writerow({"orig_id": "1", "sample_uid": "1_a", "label": 1, "significance": 2})
                writer.writerow({"orig_id": "2", "sample_uid": "2_a", "label": 2, "significance": 2})

            summary = inspect_feature_csv(path)

            self.assertEqual(
                summary["significant_sdc_target_counts"],
                {"0": 1, "1": 1},
            )


if __name__ == "__main__":
    unittest.main()
