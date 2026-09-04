import json
import tempfile
import unittest
from pathlib import Path

from detect_sdc.adapters.registry import import_symbol
from detect_sdc.mapping import (
    LayerAwareResidualMLP,
    split_mapping_dataset,
    train_model,
)
from detect_sdc.pipeline.jobs import load_pipeline_job


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
JOB_NAMES = (
    "qwen25_vl_earthvqa",
    "qwen25_vl_lingoqa",
    "qwen25_vl_vqav2",
    "llava15_earthvqa",
    "llava15_lingoqa",
    "llava15_vqav2",
    "internvl3_earthvqa",
    "internvl3_lingoqa",
    "internvl3_vqav2",
)


class _SizedDataset:
    group_ids = tuple(
        f"group-{index // 5}" for index in range(100)
    )

    def __len__(self):
        return 100


class MappingTrainingTest(unittest.TestCase):
    def test_mapping_model_defaults_match_canonical_architecture(self):
        model = LayerAwareResidualMLP()

        self.assertEqual(model.input_proj.out_features, 64)
        self.assertEqual(len(model.blocks), 8)

    def test_mapping_split_is_group_disjoint_and_deterministic(self):
        first = split_mapping_dataset(
            _SizedDataset(),
            valid_ratio=0.15,
            test_ratio=0.15,
            seed=42,
        )
        second = split_mapping_dataset(
            _SizedDataset(),
            valid_ratio=0.15,
            test_ratio=0.15,
            seed=42,
        )
        self.assertEqual(first.train.indices, second.train.indices)
        groups = _SizedDataset.group_ids
        train_groups = {groups[index] for index in first.train.indices}
        valid_groups = {
            groups[index] for index in first.selection.indices
        }
        test_groups = {groups[index] for index in first.final.indices}
        self.assertFalse(train_groups & valid_groups)
        self.assertFalse(train_groups & test_groups)
        self.assertFalse(valid_groups & test_groups)
        self.assertEqual(
            len(first.train) + len(first.selection) + len(first.final),
            100,
        )

    def test_all_jobs_use_one_trainer_and_model_class(self):
        config = REPOSITORY_ROOT / "configs/experiments/current.yaml"
        jobs = [
            load_pipeline_job(
                config,
                name,
                repository_root=REPOSITORY_ROOT,
            )
            for name in JOB_NAMES
        ]
        trainers = {
            import_symbol(job.mapping_training_config["trainer"])
            for job in jobs
        }
        model_classes = {
            import_symbol(job.injection_config["mapping_class"])
            for job in jobs
        }
        self.assertEqual(trainers, {train_model})
        self.assertEqual(model_classes, {LayerAwareResidualMLP})
        architectures = {
            (
                job.injection_config["mapping_kwargs"]["hidden_dim"],
                job.injection_config["mapping_kwargs"]["num_blocks"],
            )
            for job in jobs
        }
        self.assertEqual(architectures, {(64, 8)})
        training_profiles = {
            tuple(sorted(job.mapping_training_config["kwargs"].items()))
            for job in jobs
        }
        self.assertEqual(len(training_profiles), 1)

    def test_shared_trainer_runs_one_cpu_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "mapping.jsonl"
            checkpoint = root / "mapping.pt"
            with source.open("w", encoding="utf-8") as stream:
                for index in range(30):
                    inputs = [
                        index / 100.0,
                        index / 100.0 + 0.1,
                        index / 100.0 + 0.2,
                        index / 100.0 + 0.3,
                    ]
                    stream.write(
                        json.dumps(
                            {
                                "x": inputs,
                                "y": [value + 0.01 for value in inputs],
                                "src_layer": index % 3,
                                "tgt_layer": index % 3 + 1,
                                "step": index,
                                "semantic_group_id": f"group-{index // 3}",
                            }
                        )
                        + "\n"
                    )

            model, metrics = train_model(
                str(source),
                str(checkpoint),
                model_kwargs={
                    "x_dim": 4,
                    "num_layers": 4,
                    "layer_emb_dim": 2,
                    "hidden_dim": 8,
                    "num_blocks": 1,
                    "dropout": 0.0,
                },
                batch_size=8,
                epochs=1,
                num_workers=0,
                valid_ratio=0.2,
                test_ratio=0.2,
                scheduler_enabled=False,
                pin_memory=False,
                persistent_workers=False,
                device="cpu",
            )

            self.assertTrue(checkpoint.is_file())
            self.assertEqual(model.out_proj.out_features, 4)
            self.assertIn("cosine_similarity", metrics)


if __name__ == "__main__":
    unittest.main()
