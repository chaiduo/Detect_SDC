import unittest
from types import SimpleNamespace

import pandas as pd

from detect_sdc.dataset_splits import create_split_manifest
from detect_sdc.splitting import split_by_group, validate_identity_columns


class GroupSplittingTest(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(
            {
                "orig_id": ["scene-a", "scene-a", "scene-b", "scene-c", "scene-c"],
                "sample_uid": ["a-1", "a-2", "b-1", "c-1", "c-2"],
                "feature": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )

    def test_nonnumeric_groups_are_kept_disjoint(self):
        split = split_by_group(
            self.frame,
            group_column="orig_id",
            holdout_ratio=1 / 3,
            random_state=42,
        )

        train_groups = set(split.train["orig_id"])
        valid_groups = set(split.holdout["orig_id"])
        self.assertFalse(train_groups & valid_groups)
        self.assertEqual(len(split.train) + len(split.holdout), len(self.frame))
        self.assertEqual(split.summary.group_overlap, 0)

    def test_split_is_deterministic(self):
        first = split_by_group(self.frame, holdout_ratio=1 / 3, random_state=42)
        second = split_by_group(self.frame, holdout_ratio=1 / 3, random_state=42)

        self.assertEqual(
            first.holdout["sample_uid"].tolist(),
            second.holdout["sample_uid"].tolist(),
        )

    def test_duplicate_sample_uids_are_rejected(self):
        duplicate = self.frame.copy()
        duplicate.loc[1, "sample_uid"] = "a-1"

        with self.assertRaisesRegex(ValueError, "duplicate rows"):
            validate_identity_columns(duplicate)

    def test_dataset_manifest_keeps_semantic_entities_together(self):
        samples = [
            SimpleNamespace(
                orig_id=f"image-{image}:question-{question}",
                semantic_group_id=f"image-{image}",
            )
            for image in range(10)
            for question in range(2)
        ]

        manifest = create_split_manifest(
            "test",
            samples,
            seed=42,
            fit_ratio=0.7,
            calibration_ratio=0.15,
            test_ratio=0.15,
        )

        group_splits = {}
        for assignment in manifest.assignments:
            group_splits.setdefault(
                assignment.semantic_group_id,
                set(),
            ).add(assignment.split)
        self.assertTrue(
            all(len(splits) == 1 for splits in group_splits.values())
        )
        self.assertEqual(len(manifest.assignments), len(samples))


if __name__ == "__main__":
    unittest.main()
