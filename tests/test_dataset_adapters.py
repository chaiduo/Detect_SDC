import unittest
from pathlib import Path
from unittest.mock import patch

from detect_sdc.adapters import load_dataset_adapter
from detect_sdc.adapters.datasets.lingoqa import LingoQAAdapter


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class DatasetAdapterTest(unittest.TestCase):
    def test_earthvqa_uses_image_and_qa_index_as_stable_id(self):
        adapter = load_dataset_adapter(
            REPOSITORY_ROOT / "configs/datasets/earthvqa.yaml"
        )

        samples = list(adapter.iter_samples(max_samples=3))

        self.assertEqual(adapter.name, "earthvqa")
        self.assertEqual(len(samples), 3)
        self.assertTrue(samples[0].orig_id.startswith("275.png:"))
        self.assertTrue(Path(samples[0].image).is_file())
        self.assertEqual(len({sample.orig_id for sample in samples}), 3)

    def test_lingoqa_expands_each_question_into_image_samples(self):
        adapter = load_dataset_adapter(
            REPOSITORY_ROOT / "configs/datasets/lingoqa.yaml"
        )

        samples = list(adapter.iter_samples(max_samples=6))

        self.assertEqual(adapter.name, "lingoqa")
        self.assertEqual(len(samples), 6)
        self.assertTrue(samples[0].orig_id.endswith(":0"))
        self.assertTrue(samples[4].orig_id.endswith(":4"))
        self.assertTrue(Path(samples[0].image).is_file())
        self.assertEqual(len({sample.orig_id for sample in samples}), 6)

    def test_lingoqa_preserves_duplicate_annotations_in_source_order(self):
        import pandas as pd

        row = {
            "question_id": "question-1",
            "segment_id": "segment-1",
            "images": ["first.jpg", "second.jpg"],
            "question": "What happened?",
            "answer": "A vehicle stopped.",
        }
        frame = pd.DataFrame([row, row])
        adapter = LingoQAAdapter(
            annotations=Path("unused.parquet"),
            images=Path("/images"),
            images_per_question=2,
        )

        with patch("pandas.read_parquet", return_value=frame):
            samples = list(adapter.iter_samples())

        self.assertEqual(len(samples), 4)
        self.assertEqual(len({sample.orig_id for sample in samples}), 4)
        self.assertEqual(
            [sample.metadata["annotation_occurrence"] for sample in samples],
            [0, 0, 1, 1],
        )

    def test_vqav2_uses_question_id_and_preserves_embedded_image(self):
        adapter = load_dataset_adapter(
            REPOSITORY_ROOT / "configs/datasets/vqav2.yaml"
        )

        samples = list(adapter.iter_samples(max_samples=2))

        self.assertEqual(adapter.name, "vqav2")
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0].orig_id, "262148000")
        self.assertIsInstance(samples[0].image, dict)
        self.assertEqual(len({sample.orig_id for sample in samples}), 2)


if __name__ == "__main__":
    unittest.main()
