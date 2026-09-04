import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from detect_sdc.dataset_splits import (
    create_split_manifest,
    write_split_manifest,
)
from detect_sdc.features import (
    FeatureSpec,
    SampleSkipped,
    build_supervision,
    extract_feature_row,
    iter_json_samples,
    stable_sample_uid,
)
from detect_sdc.features.jobs import FeatureJob, execute_feature_job


class FeatureExtractionTest(unittest.TestCase):
    def setUp(self):
        self.spec = FeatureSpec(
            selected_layer_pairs=((1, 2),),
            distance_pairs=((1, 2),),
            last_k_steps=2,
            finite_only=True,
        )

    def test_extracts_last_steps_and_ignores_non_finite_values(self):
        sample = _sample()
        sample["mean_std_cos"]["records"] = [
            _record(1, cos_sim=100.0),
            _record(2, cos_sim=float("inf")),
            _record(3, cos_sim=3.0),
        ]

        row = extract_feature_row(
            sample,
            spec=self.spec,
            uid_namespace="model_dataset",
        )

        self.assertEqual(row["total_steps"], 3)
        self.assertEqual(row["num_steps_used"], 2)
        self.assertEqual(row["cos_sim_mean_p1_2"], 3.0)
        self.assertEqual(row["cos_sim_max_p1_2"], 3.0)
        self.assertEqual(row["label"], 2)
        self.assertEqual(row["significant_sdc_target"], 1)
        self.assertFalse(math.isnan(row["l2_distance_mean_p1_2"]))

    def test_extracts_prefix_steps_for_online_detection(self):
        sample = _sample()
        sample["mean_std_cos"]["records"] = [
            _record(1, cos_sim=1.0),
            _record(2, cos_sim=2.0),
            _record(3, cos_sim=100.0),
        ]
        spec = FeatureSpec(
            selected_layer_pairs=((1, 2),),
            distance_pairs=((1, 2),),
            last_k_steps=2,
            finite_only=True,
            step_window="prefix",
        )

        row = extract_feature_row(
            sample,
            spec=spec,
            uid_namespace="model_dataset",
        )

        self.assertEqual(row["num_steps_used"], 2)
        self.assertEqual(row["cos_sim_mean_p1_2"], 1.5)
        self.assertEqual(row["cos_sim_max_p1_2"], 2.0)
        self.assertEqual(row["cos_sim_min_p1_2"], 1.0)

    def test_rejects_unknown_step_window(self):
        with self.assertRaisesRegex(ValueError, "step_window"):
            FeatureSpec(
                selected_layer_pairs=((1, 2),),
                distance_pairs=((1, 2),),
                step_window="middle",
            )

    def test_sample_uid_depends_on_stable_fields_not_mapping_order(self):
        first = _sample()
        first["fault"] = {"module": "layer.1", "idx": 5, "bit_positions": [2, 7]}
        second = _sample()
        second["fault"] = {"bit_positions": [2, 7], "idx": 5, "module": "layer.1"}

        self.assertEqual(
            stable_sample_uid(first, namespace="model_dataset"),
            stable_sample_uid(second, namespace="model_dataset"),
        )

    def test_identical_answers_override_invalid_legacy_significance(self):
        sample = _sample()
        sample["pred_answer"] = sample["clean_answer"]
        sample["significance"] = -1

        self.assertEqual(build_supervision(sample), (0, 0, 0))

    def test_changed_answer_with_invalid_significance_is_skipped(self):
        sample = _sample()
        sample["significance"] = -1

        with self.assertRaisesRegex(SampleSkipped, "invalid_significance"):
            build_supervision(sample)

    def test_jsonl_reader_streams_and_limits_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.jsonl"
            with path.open("w", encoding="utf-8") as stream:
                for sample_id in range(3):
                    stream.write(json.dumps({"id": sample_id}) + "\n")

            samples = list(iter_json_samples(path, max_samples=2))

        self.assertEqual([sample["id"] for sample in samples], [0, 1])

    def test_exact_duplicate_sample_is_deduplicated_by_stable_uid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "samples.jsonl"
            samples = []
            for sample_id in ("scene-a", "scene-a", "scene-b", "scene-c"):
                sample = _sample()
                sample["id"] = sample_id
                samples.append(sample)
            with input_path.open("w", encoding="utf-8") as stream:
                for sample in samples:
                    stream.write(json.dumps(sample) + "\n")
            manifest_path = root / "split.json"
            manifest = create_split_manifest(
                "test_dataset",
                (
                    SimpleNamespace(
                        orig_id=sample_id,
                        semantic_group_id=sample_id,
                    )
                    for sample_id in ("scene-a", "scene-b", "scene-c")
                ),
                seed=42,
                fit_ratio=0.34,
                calibration_ratio=0.33,
                test_ratio=0.33,
            )
            write_split_manifest(manifest, manifest_path)

            job = FeatureJob(
                name="test_job",
                model="test_model",
                dataset="test_dataset",
                uid_namespace="test_model_test_dataset",
                input_path=input_path,
                fit_output=root / "fit.csv",
                calibration_output=root / "calibration.csv",
                test_output=root / "test.csv",
                split_manifest=manifest_path,
                group_column="semantic_group_id",
                spec=self.spec,
            )
            summary = execute_feature_job(job)

            rows = []
            for path in (
                job.fit_output,
                job.calibration_output,
                job.test_output,
            ):
                with path.open(encoding="utf-8-sig") as stream:
                    rows.extend(csv.DictReader(stream))

        self.assertEqual(summary["extracted_rows"], 3)
        self.assertEqual(summary["skipped"], {"duplicate_sample": 1})
        self.assertEqual(len(rows), 3)
        self.assertEqual(summary["group_overlap"], 0)


def _sample():
    return {
        "id": "sample-a",
        "fault": None,
        "clean_answer": "clean",
        "pred_answer": "changed",
        "significance": 2,
        "mean_std_cos": {"records": [_record(1)]},
    }


def _record(step, *, cos_sim=1.0):
    return {
        "step": step,
        "src_layer": 1,
        "tgt_layer": 2,
        "cos_sim": cos_sim,
        "mean_diff": 2.0,
        "std_diff": 3.0,
        "l2_distance": 4.0,
    }


if __name__ == "__main__":
    unittest.main()
