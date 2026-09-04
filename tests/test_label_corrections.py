import unittest

from scripts.correct_empty_response_labels import (
    RULE_ID,
    correct_record,
    needs_empty_response_correction,
)


class EmptyResponseCorrectionTest(unittest.TestCase):
    def test_only_changed_empty_fault_responses_are_corrected(self):
        base = {
            "injected": True,
            "clean_answer": "Yes",
            "pred_answer": "",
            "quality_score": 2,
            "significance": 0,
            "label_status": "valid",
        }

        self.assertTrue(needs_empty_response_correction(base))
        self.assertFalse(
            needs_empty_response_correction(
                {**base, "pred_answer": "Yes"}
            )
        )
        self.assertFalse(
            needs_empty_response_correction(
                {**base, "injected": False}
            )
        )
        self.assertFalse(
            needs_empty_response_correction(
                {**base, "significance": 2}
            )
        )

    def test_correction_is_auditable(self):
        original = {
            "injected": True,
            "clean_answer": "Yes",
            "pred_answer": "",
            "quality_score": 1,
            "significance": 1,
            "label_status": "valid",
        }

        corrected = correct_record(original)

        self.assertEqual(corrected["quality_score"], 0)
        self.assertEqual(corrected["significance"], 2)
        self.assertEqual(corrected["label_status"], "valid")
        self.assertEqual(
            corrected["manual_label_correction"]["rule_id"],
            RULE_ID,
        )
        self.assertEqual(original["quality_score"], 1)


if __name__ == "__main__":
    unittest.main()
