import copy
import tempfile
import unittest
from pathlib import Path

from detect_sdc.config import atomic_write_yaml, load_yaml
from detect_sdc.experiment import (
    load_experiment,
    validate_experiment_configuration,
)
from detect_sdc.features.jobs import load_feature_job
from detect_sdc.pipeline import PipelineStage


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class ExperimentConfigTest(unittest.TestCase):
    def test_current_matrix_is_valid(self):
        experiment = load_experiment(REPOSITORY_ROOT / "configs/experiments/current.yaml")
        experiment.validate_references(REPOSITORY_ROOT)

        self.assertEqual(experiment.name, "current_significant_sdc_matrix")
        self.assertEqual(len(experiment.matrix), 9)
        self.assertEqual(experiment.stages[0], PipelineStage.PROFILE)
        self.assertEqual(experiment.stages[-1], PipelineStage.REPORT)
        summary = validate_experiment_configuration(
            REPOSITORY_ROOT / "configs/experiments/current.yaml",
            repository_root=REPOSITORY_ROOT,
        )
        self.assertEqual(summary["jobs"], 9)

    def test_current_matrix_defines_existing_feature_inputs(self):
        config = REPOSITORY_ROOT / "configs/experiments/current.yaml"
        expected = {
            "qwen25_vl_earthvqa",
            "qwen25_vl_lingoqa",
            "qwen25_vl_vqav2",
            "llava15_earthvqa",
            "llava15_lingoqa",
            "llava15_vqav2",
        }

        jobs = {
            name: load_feature_job(config, name, repository_root=REPOSITORY_ROOT)
            for name in expected
        }

        self.assertEqual(set(jobs), expected)
        self.assertTrue(all(job.input_path.is_file() for job in jobs.values()))
        self.assertTrue(all(len(job.spec.feature_columns) == 72 for job in jobs.values()))

    def test_validation_rejects_missing_execution_pair(self):
        config = load_yaml(
            REPOSITORY_ROOT / "configs/experiments/current.yaml"
        )
        config["execution"]["jobs"].pop("qwen25_vl_earthvqa")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            atomic_write_yaml(path, config)
            with self.assertRaisesRegex(
                ValueError,
                "exactly one execution job",
            ):
                validate_experiment_configuration(
                    path,
                    repository_root=REPOSITORY_ROOT,
                )

    def test_validation_rejects_mapping_dimension_mismatch(self):
        config = load_yaml(
            REPOSITORY_ROOT / "configs/experiments/current.yaml"
        )
        config = copy.deepcopy(config)
        config["execution"]["jobs"]["qwen25_vl_earthvqa"]["injection"] = {
            "mapping_kwargs": {
                "x_dim": 32,
                "num_layers": 28,
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            atomic_write_yaml(path, config)
            with self.assertRaisesRegex(
                ValueError,
                "must equal projection_dim",
            ):
                validate_experiment_configuration(
                    path,
                    repository_root=REPOSITORY_ROOT,
                )

    def test_validation_rejects_unknown_trainer_argument(self):
        config = load_yaml(
            REPOSITORY_ROOT / "configs/experiments/current.yaml"
        )
        config = copy.deepcopy(config)
        job = config["execution"]["jobs"]["qwen25_vl_earthvqa"]
        job["mapping_training"] = {"kwargs": {"unknown_option": True}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            atomic_write_yaml(path, config)
            with self.assertRaisesRegex(
                TypeError,
                "does not match callable signature",
            ):
                validate_experiment_configuration(
                    path,
                    repository_root=REPOSITORY_ROOT,
                )


if __name__ == "__main__":
    unittest.main()
