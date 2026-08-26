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
)


class _SizedDataset:
    def __len__(self):
        return 100


class MappingTrainingTest(unittest.TestCase):
    def test_historical_split_contracts_are_preserved(self):
        sequential = split_mapping_dataset(
            _SizedDataset(),
            strategy="sequential",
            valid_ratio=0.15,
            test_ratio=0.1,
            test_ratio_in_train=0.15,
            seed=42,
        )
        self.assertEqual(sequential.train.indices, list(range(73)))
        self.assertEqual(sequential.selection.indices, list(range(73, 85)))
        self.assertEqual(sequential.final.indices, list(range(85, 100)))

        random_split = split_mapping_dataset(
            _SizedDataset(),
            strategy="random",
            valid_ratio=0.1,
            test_ratio=0.1,
            test_ratio_in_train=0.15,
            seed=42,
        )
        self.assertEqual(len(random_split.train), 80)
        self.assertEqual(len(random_split.selection), 10)
        self.assertEqual(len(random_split.final), 10)
        self.assertEqual(
            set(random_split.train.indices)
            & set(random_split.selection.indices),
            set(),
        )
        self.assertEqual(
            set(random_split.train.indices) & set(random_split.final.indices),
            set(),
        )

        partition = split_mapping_dataset(
            _SizedDataset(),
            strategy="partition",
            valid_ratio=0.15,
            test_ratio=0.15,
            test_ratio_in_train=0.15,
            seed=42,
        )
        self.assertEqual(partition.train.indices, list(range(70)))
        self.assertEqual(partition.selection.indices, list(range(70, 85)))
        self.assertEqual(partition.final.indices, list(range(85, 100)))

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
        self.assertEqual(
            jobs[2].mapping_training_config["kwargs"]["split_strategy"],
            "random",
        )
        self.assertEqual(
            jobs[4].mapping_training_config["kwargs"]["cosine_weight"],
            2.0,
        )
        self.assertEqual(
            jobs[4].injection_config["mapping_kwargs"]["hidden_dim"],
            1024,
        )

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
                split_strategy="sequential",
                valid_ratio=0.2,
                test_ratio=0.1,
                test_ratio_in_train=0.2,
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
