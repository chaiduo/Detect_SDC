import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from detect_sdc.adapters.datasets.base import DatasetSample
from detect_sdc.adapters.registry import import_symbol
from detect_sdc.pipeline import PipelineStage
from detect_sdc.pipeline.injection import (
    CleanAnswerIndex,
    load_clean_answers,
    run_injection_samples,
)
from detect_sdc.pipeline.jobs import load_pipeline_job
from detect_sdc.pipeline.mapping import (
    collect_mapping_samples,
    train_mapping_checkpoint,
)
from detect_sdc.pipeline.profile import profile_samples
from detect_sdc.pipeline.runner import run_stage


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class _FakeDataset:
    name = "fake"

    def iter_samples(self, max_samples=None):
        samples = [
            DatasetSample(
                orig_id="stable-a",
                semantic_group_id="group-a",
                question="Question A",
                ground_truth="Answer A",
                image="image-a",
            ),
            DatasetSample(
                orig_id="stable-b",
                semantic_group_id="group-b",
                question="Question B",
                ground_truth="Answer B",
                image="image-b",
            ),
        ]
        yield from samples if max_samples is None else samples[:max_samples]


class _FakeModel:
    def __init__(self):
        self.loaded_device = None
        self.closed = False

    @property
    def model(self):
        return self

    def load(self, device):
        self.loaded_device = device

    def generate(self, question, image, *, max_new_tokens):
        return f"{question}:{image}:{max_new_tokens}"

    def close(self):
        self.closed = True


class _FakeProfiler:
    last_instance = None

    def __init__(self, model, *, proj_dim, proj_method, seed):
        self.model = model
        self.proj_dim = proj_dim
        self.proj_method = proj_method
        self.seed = seed
        self.registered = False
        self.unregistered = False
        self.reset_count = 0
        _FakeProfiler.last_instance = self

    def register(self):
        self.registered = True

    def finalize(self):
        pass

    def save_attn_proj_interlayer_jsonl(
        self,
        path,
        sample_id=None,
        orig_id=None,
        semantic_group_id=None,
        split=None,
    ):
        with Path(path).open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "orig_id": orig_id,
                        "semantic_group_id": semantic_group_id,
                        "split": split,
                    }
                )
                + "\n"
            )
        return 1

    def get_attn_proj_model_compare_result(
        self,
        *,
        predictor_model,
        device,
        include_vectors,
        max_steps=None,
    ):
        return {
            "device": device,
            "include_vectors": include_vectors,
            "mapping": type(predictor_model).__name__,
        }

    def reset(self, clear_stats=False):
        self.reset_count += int(clear_stats)

    def unregister(self):
        self.unregistered = True


class _FakeMappingModel:
    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self


class _FakeInjector:
    inject_count = 0

    def __init__(self, model, mode):
        self.model = model
        self.mode = mode
        self.fault_info = None

    def reset(self):
        self.model.fault_active = False
        self.fault_info = None

    def register_step_hooks(self):
        pass

    def set_num_bits(self, num_bits):
        self.num_bits = num_bits

    def set_bit_policy(self, bit_policy):
        self.bit_policy = bit_policy

    def inject(self):
        type(self).inject_count += 1
        self.model.fault_active = type(self).inject_count == 1
        self.fault_info = {
            "bit_positions": list(range(self.num_bits)),
            "bit_policy": self.bit_policy,
        }

    def unregister_hooks(self):
        self.model.fault_active = False


class _AlwaysFaultInjector(_FakeInjector):
    def inject(self):
        self.model.fault_active = True
        self.fault_info = {
            "bit_positions": list(range(self.num_bits)),
            "bit_policy": self.bit_policy,
        }


class _NoOpInjector(_FakeInjector):
    def inject(self):
        self.model.fault_active = False
        self.fault_info = None


class _AuxiliaryMonitor:
    def __init__(self, model):
        self.model = model

    def register(self):
        pass

    def unregister(self):
        pass

    def start_sample(self):
        pass

    def finish_sample(self):
        return {"trace": 1}


class PipelineTest(unittest.TestCase):
    def test_profile_stage_preserves_stable_orig_id(self):
        model = _FakeModel()

        rows = profile_samples(
            model,
            _FakeDataset(),
            device="cpu",
            max_samples=1,
            max_new_tokens=12,
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], 0)
        self.assertEqual(rows[0]["orig_id"], "stable-a")
        self.assertEqual(rows[0]["sample_uid"], "stable-a")
        self.assertEqual(model.loaded_device, "cpu")
        self.assertTrue(model.closed)

    def test_all_matrix_pairs_have_pipeline_paths(self):
        names = (
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

        jobs = [
            load_pipeline_job(
                REPOSITORY_ROOT / "configs/experiments/current.yaml",
                name,
                repository_root=REPOSITORY_ROOT,
            )
            for name in names
        ]

        self.assertTrue(all(job.paths.profile_output.name == "profile.json" for job in jobs))
        self.assertTrue(all(job.paths.mapping_data.name == "mapping.jsonl" for job in jobs))
        self.assertTrue(all(job.paths.injected_output.name == "injection.jsonl" for job in jobs))
        self.assertTrue(all(job.paths.labeled_output.name == "labels.jsonl" for job in jobs))
        self.assertEqual(jobs[1].projection_method, "project")
        self.assertEqual(jobs[4].projection_method, "project")
        self.assertEqual(jobs[4].profiler_seed, 42)
        self.assertEqual(jobs[0].injection_config["fault_runs"], 10)
        self.assertEqual(jobs[0].injection_config["bit_policy"], "random")
        self.assertEqual(
            jobs[4].injection_config["mapping_kwargs"]["hidden_dim"],
            64,
        )
        self.assertEqual(
            jobs[4].injection_config["mapping_kwargs"]["num_blocks"],
            8,
        )

    def test_mapping_collection_is_atomic_and_uses_shared_lifecycle(self):
        model = _FakeModel()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "mapping.jsonl"
            summary = collect_mapping_samples(
                model,
                _FakeDataset(),
                output,
                device="cpu",
                max_samples=1,
                max_new_tokens=12,
                projection_dim=64,
                projection_method="project",
                seed=42,
                profiler_factory=_FakeProfiler,
            )

            rows = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]

        profiler = _FakeProfiler.last_instance
        self.assertEqual(
            rows,
            [
                {
                    "sample_id": 0,
                    "orig_id": "stable-a",
                    "semantic_group_id": "group-a",
                    "split": "fit",
                }
            ],
        )
        self.assertEqual(summary["samples"], 1)
        self.assertEqual(summary["mapping_rows"], 1)
        self.assertEqual(profiler.reset_count, 1)
        self.assertTrue(profiler.registered)
        self.assertTrue(profiler.unregistered)
        self.assertTrue(model.closed)

    def test_injection_stage_retains_every_fault_run_atomically(self):
        model = _FakeModel()
        model.fault_active = False
        original_generate = model.generate

        def generate(question, image, *, max_new_tokens):
            clean = original_generate(
                question,
                image,
                max_new_tokens=max_new_tokens,
            )
            return f"{clean}:fault" if model.fault_active else clean

        model.generate = generate
        clean = CleanAnswerIndex(
            by_sequence={0: "Question A:image-a:12"},
            by_orig_id={},
        )
        _FakeInjector.inject_count = 0

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "injected.jsonl"
            summary = run_injection_samples(
                model,
                _FakeDataset(),
                _FakeMappingModel(),
                clean,
                output,
                device="cpu",
                max_samples=1,
                max_new_tokens=12,
                projection_dim=64,
                projection_method="project",
                profiler_seed=42,
                fault_runs=2,
                num_bits=2,
                fault_seed=42,
                profiler_factory=_FakeProfiler,
                injector_factory=_FakeInjector,
                auxiliary_monitor_factory=_AuxiliaryMonitor,
                auxiliary_scorer=lambda trace: {
                    "ranger_score": float(trace["trace"])
                },
            )
            rows = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(summary["generated"], 3)
        self.assertEqual(summary["rows_written"], 3)
        self.assertEqual(summary["sdc_rows_observed"], 1)
        self.assertEqual(
            [row["injected"] for row in rows],
            [False, True, True],
        )
        self.assertEqual(
            [row["sample_uid"] for row in rows],
            [
                "stable-a:clean",
                "stable-a:fault:0",
                "stable-a:fault:1",
            ],
        )
        self.assertEqual(rows[1]["is_sdc"], 1)
        self.assertEqual(rows[2]["is_sdc"], 0)
        self.assertTrue(all(row["ranger_score"] == 1.0 for row in rows))
        self.assertEqual(summary["retention_policy"], "all_runs")
        self.assertEqual(summary["bit_policy"], "random")
        self.assertEqual(rows[1]["fault"]["bit_policy"], "random")
        self.assertTrue(model.closed)

    def test_injection_rejects_a_fault_hook_that_never_fires(self):
        model = _FakeModel()
        model.fault_active = False
        clean = CleanAnswerIndex(
            by_sequence={0: "Question A:image-a:12"},
            by_orig_id={},
        )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "injected.jsonl"
            with self.assertRaisesRegex(
                RuntimeError,
                "fault hook did not inject",
            ):
                run_injection_samples(
                    model,
                    _FakeDataset(),
                    _FakeMappingModel(),
                    clean,
                    output,
                    device="cpu",
                    max_samples=1,
                    max_new_tokens=12,
                    projection_dim=64,
                    projection_method="project",
                    profiler_seed=42,
                    fault_runs=1,
                    num_bits=2,
                    fault_seed=42,
                    profiler_factory=_FakeProfiler,
                    injector_factory=_NoOpInjector,
                )

            self.assertFalse(output.exists())
            self.assertFalse(output.with_name("injected.jsonl.tmp").exists())

    def test_injection_resume_restarts_interrupted_run(self):
        model = _FakeModel()
        model.fault_active = False
        original_generate = model.generate

        def generate(question, image, *, max_new_tokens):
            clean = original_generate(
                question,
                image,
                max_new_tokens=max_new_tokens,
            )
            return f"{clean}:fault" if model.fault_active else clean

        model.generate = generate
        clean = CleanAnswerIndex(
            by_sequence={0: "Question A:image-a:12"},
            by_orig_id={},
        )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "injected.jsonl"
            temporary = output.with_name("injected.jsonl.tmp")
            temporary.write_text(
                "\n".join(
                    json.dumps(record)
                    for record in (
                        {
                            "run_index": None,
                            "is_sdc": 0,
                            "marker": "clean",
                        },
                        {
                            "run_index": 0,
                            "is_sdc": 1,
                            "marker": "completed",
                        },
                        {
                            "run_index": 1,
                            "is_sdc": 1,
                            "marker": "partial",
                        },
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_injection_samples(
                model,
                _FakeDataset(),
                _FakeMappingModel(),
                clean,
                output,
                device="cpu",
                max_samples=1,
                max_new_tokens=12,
                projection_dim=64,
                projection_method="project",
                profiler_seed=42,
                fault_runs=3,
                num_bits=2,
                fault_seed=42,
                resume_from_run=1,
                profiler_factory=_FakeProfiler,
                injector_factory=_AlwaysFaultInjector,
            )
            rows = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(
            [row["run_index"] for row in rows],
            [None, 0, 1, 2],
        )
        self.assertEqual(
            [row.get("marker") for row in rows],
            ["clean", "completed", None, None],
        )
        self.assertEqual(summary["generated"], 4)
        self.assertEqual(summary["rows_written"], 4)
        self.assertEqual(summary["sdc_rows_observed"], 3)
        self.assertEqual(summary["resumed_from_run"], 1)

    def test_all_mapping_trainers_resolve_from_configuration(self):
        config = REPOSITORY_ROOT / "configs/experiments/current.yaml"
        names = (
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
        for name in names:
            job = load_pipeline_job(
                config,
                name,
                repository_root=REPOSITORY_ROOT,
            )
            trainer = import_symbol(job.mapping_training_config["trainer"])
            self.assertTrue(callable(trainer))

    def test_mapping_training_publishes_checkpoint_atomically(self):
        def trainer(*, jsonl_path, save_best_path, device, epochs):
            self.assertEqual(device, "cpu")
            self.assertEqual(epochs, 1)
            self.assertTrue(Path(jsonl_path).is_file())
            Path(save_best_path).write_bytes(b"new checkpoint")
            return object(), {"loss": 0.25}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "mapping.jsonl"
            output = root / "mapping.pt"
            source.write_text("{}\n", encoding="utf-8")
            summary = train_mapping_checkpoint(
                source,
                output,
                trainer=trainer,
                trainer_kwargs={"epochs": 1},
                device="cpu",
            )

            self.assertEqual(output.read_bytes(), b"new checkpoint")
            self.assertFalse((root / "mapping.pt.tmp").exists())
            self.assertEqual(summary["metrics"], {"loss": 0.25})

    def test_mapping_training_failure_preserves_existing_checkpoint(self):
        def trainer(**kwargs):
            Path(kwargs["save_best_path"]).write_bytes(b"partial")
            raise RuntimeError("training failed")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "mapping.jsonl"
            output = root / "mapping.pt"
            source.write_text("{}\n", encoding="utf-8")
            output.write_bytes(b"existing checkpoint")
            with self.assertRaisesRegex(RuntimeError, "training failed"):
                train_mapping_checkpoint(
                    source,
                    output,
                    trainer=trainer,
                    trainer_kwargs={},
                    device="cpu",
                    overwrite=True,
                )

            self.assertEqual(output.read_bytes(), b"existing checkpoint")
            self.assertFalse((root / "mapping.pt.tmp").exists())

    def test_clean_answer_index_prefers_orig_id_and_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "golden.json"
            path.write_text(
                json.dumps(
                    [
                        {"id": 0, "orig_id": "stable-a", "pre_answer": "A"},
                        {"id": 1, "orig_id": "stable-b", "pre_answer": "B"},
                    ]
                ),
                encoding="utf-8",
            )
            index = load_clean_answers(path)
            self.assertEqual(index.get(0, "stable-b"), "B")

            path.write_text(
                json.dumps(
                    [
                        {"id": 0, "orig_id": "same", "pre_answer": "A"},
                        {"id": 1, "orig_id": "same", "pre_answer": "B"},
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Duplicate golden orig_id"):
                load_clean_answers(path)

    def test_unified_runner_supports_gpu_free_dry_run(self):
        summary = run_stage(
            REPOSITORY_ROOT / "configs/experiments/current.yaml",
            "llava15_lingoqa",
            PipelineStage.COLLECT_MAPPING,
            repository_root=REPOSITORY_ROOT,
            device="cuda:0",
            dry_run=True,
        )

        self.assertTrue(summary["dry_run"])
        self.assertEqual(summary["stage"], "collect_mapping")

    def test_unified_runner_forwards_injection_resume_run(self):
        with mock.patch(
            "detect_sdc.pipeline.runner.run_injection_job",
            return_value={"ok": True},
        ) as run_injection:
            summary = run_stage(
                REPOSITORY_ROOT / "configs/experiments/current.yaml",
                "internvl3_vqav2",
                PipelineStage.INJECT,
                repository_root=REPOSITORY_ROOT,
                device="cuda:0",
                resume_injection_from_run=4,
            )

        self.assertEqual(summary, {"ok": True})
        self.assertEqual(run_injection.call_args.kwargs["resume_from_run"], 4)


if __name__ == "__main__":
    unittest.main()
