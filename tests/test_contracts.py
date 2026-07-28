import unittest

from detect_sdc.records import LabeledRecord, LabelStatus, SampleRef


class LabeledRecordTest(unittest.TestCase):
    def setUp(self):
        self.ref = SampleRef(dataset="earthvqa", orig_id="10", sample_uid="10_fault_1")

    def test_identical_answer_is_non_significant(self):
        record = LabeledRecord(
            ref=self.ref,
            question="What is shown?",
            clean_answer="Residential",
            pred_answer="Residential",
            quality_score=2,
            significance=0,
            status=LabelStatus.IDENTICAL_ANSWER,
        )

        self.assertFalse(record.target_significant_sdc)

    def test_severe_changed_answer_is_target(self):
        record = LabeledRecord(
            ref=self.ref,
            question="What is shown?",
            clean_answer="Residential",
            pred_answer="Forest",
            quality_score=0,
            significance=2,
        )

        self.assertTrue(record.target_significant_sdc)

    def test_legacy_parse_failure_does_not_use_minus_one(self):
        record = LabeledRecord.from_legacy(
            {
                "id": 10,
                "question": "What is shown?",
                "clean_answer": "Residential",
                "pred_answer": "Forest",
                "quality_score": -1,
                "significance": -1,
            },
            dataset="earthvqa",
        )

        self.assertEqual(record.status, LabelStatus.PARSE_ERROR)
        self.assertIsNone(record.quality_score)
        self.assertIsNone(record.significance)

    def test_inconsistent_scales_are_rejected(self):
        with self.assertRaises(ValueError):
            LabeledRecord(
                ref=self.ref,
                question="What is shown?",
                clean_answer="Residential",
                pred_answer="Forest",
                quality_score=2,
                significance=2,
            )


if __name__ == "__main__":
    unittest.main()
