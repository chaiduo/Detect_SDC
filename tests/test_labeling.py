import json
import tempfile
import unittest
from pathlib import Path

from detect_sdc.labeling import (
    _encode_chat_batch,
    extract_score,
    label_jsonl,
    label_records,
    quality_score_to_significance,
)


class _FakeJudge:
    def __init__(self, scores):
        self.scores = iter(scores)
        self.calls = []

    def evaluate_batch(self, items):
        self.calls.append([item["pred_answer"] for item in items])
        scores = [next(self.scores) for _ in items]
        return scores, [f"Feedback [RESULT] {score}" for score in scores]


class _FakeTokenizer:
    def __init__(self):
        self.conversations = []
        self.rendered = []

    def apply_chat_template(self, conversation, *, tokenize, add_generation_prompt):
        self.conversations.append(conversation)
        if not isinstance(conversation[0], dict):
            raise TypeError("Expected one conversation, not a batch")
        return f"rendered-{len(self.conversations)}"

    def __call__(self, rendered, **kwargs):
        self.rendered = rendered
        return {"input_ids": rendered, "attention_mask": rendered}


class PrometheusLabelingTest(unittest.TestCase):
    def test_chat_template_is_rendered_per_conversation_before_batch_tokenization(self):
        tokenizer = _FakeTokenizer()
        messages = [
            [{"role": "user", "content": "first"}],
            [{"role": "user", "content": "second"}],
        ]

        encoded = _encode_chat_batch(tokenizer, messages, max_length=128)

        self.assertEqual(tokenizer.conversations, messages)
        self.assertEqual(tokenizer.rendered, ["rendered-1", "rendered-2"])
        self.assertEqual(encoded["input_ids"], tokenizer.rendered)

    def test_strict_result_parsing(self):
        self.assertEqual(extract_score("Feedback [RESULT] 2"), 2)
        self.assertEqual(extract_score("Feedback 2"), -1)
        self.assertEqual(extract_score("[RESULT] 1 trailing"), -1)
        self.assertEqual(quality_score_to_significance(0), 2)
        self.assertIsNone(quality_score_to_significance(-1))

    def test_identical_answers_skip_judge_and_order_is_preserved(self):
        records = [
            _record("same", "same"),
            _record("wrong", "clean"),
            _record("minor", "clean"),
        ]
        judge = _FakeJudge([0, 1])

        labeled, failures = label_records(records, judge, batch_size=2)

        self.assertEqual([item["pred_answer"] for item in labeled], ["same", "wrong", "minor"])
        self.assertEqual(labeled[0]["label_status"], "identical_answer")
        self.assertEqual(labeled[0]["significance"], 0)
        self.assertEqual(labeled[1]["significance"], 2)
        self.assertEqual(labeled[2]["significance"], 1)
        self.assertEqual(judge.calls, [["wrong", "minor"]])
        self.assertEqual(failures, [])

    def test_empty_response_is_significant_without_calling_judge(self):
        judge = _FakeJudge([])

        labeled, failures = label_records(
            [_record("", "clean")],
            judge,
            batch_size=1,
        )

        self.assertEqual(labeled[0]["label_status"], "empty_response")
        self.assertEqual(labeled[0]["quality_score"], 0)
        self.assertEqual(labeled[0]["significance"], 2)
        self.assertEqual(judge.calls, [])
        self.assertEqual(failures, [])

    def test_parse_failure_uses_explicit_status_and_null_labels(self):
        judge = _FakeJudge([-1])

        labeled, failures = label_records(
            [_record("changed", "clean")],
            judge,
            batch_size=1,
        )

        self.assertEqual(labeled[0]["label_status"], "parse_error")
        self.assertIsNone(labeled[0]["quality_score"])
        self.assertIsNone(labeled[0]["significance"])
        self.assertEqual(len(failures), 1)

    def test_jsonl_labeling_is_atomic_and_bounded_by_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            output = root / "output.jsonl"
            with source.open("w", encoding="utf-8") as stream:
                for record in (
                    _record("same", "same"),
                    _record("wrong", "clean"),
                    _record("same-2", "same-2"),
                ):
                    stream.write(json.dumps(record) + "\n")

            summary = label_jsonl(
                source,
                output,
                _FakeJudge([0]),
                batch_size=1,
                chunk_size=2,
            )
            rows = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(summary["rows"], 3)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[1]["significance"], 2)


def _record(pred_answer, clean_answer):
    return {
        "question": "question",
        "pred_answer": pred_answer,
        "clean_answer": clean_answer,
    }


if __name__ == "__main__":
    unittest.main()
