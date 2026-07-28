import unittest
from pathlib import Path

from detect_sdc.experiment import load_experiment
from detect_sdc.features.jobs import load_feature_job
from detect_sdc.pipeline import PipelineStage


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class ExperimentConfigTest(unittest.TestCase):
    def test_current_matrix_is_valid(self):
        experiment = load_experiment(REPOSITORY_ROOT / "configs/experiments/current.yaml")
        experiment.validate_references(REPOSITORY_ROOT)

        self.assertEqual(experiment.name, "current_significant_sdc_matrix")
        self.assertEqual(len(experiment.matrix), 6)
        self.assertEqual(experiment.stages[0], PipelineStage.PROFILE)
        self.assertEqual(experiment.stages[-1], PipelineStage.REPORT)

    def test_current_matrix_defines_all_six_feature_jobs(self):
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


if __name__ == "__main__":
    unittest.main()
