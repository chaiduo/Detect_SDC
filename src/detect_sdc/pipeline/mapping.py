"""Model-agnostic inter-layer mapping collection stage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from ..adapters import load_dataset_adapter, load_model_adapter
from ..adapters.datasets.base import DatasetAdapter
from ..adapters.models.base import ModelAdapter
from ..adapters.registry import import_symbol
from ..dataset_splits import DatasetSplitManifest
from ..profiler import Profiler
from .jobs import load_pipeline_job
from .split import load_configured_split_manifest


ProfilerFactory = Callable[..., Any]
MappingTrainer = Callable[..., Any]


def collect_mapping_samples(
    model_adapter: ModelAdapter,
    dataset_adapter: DatasetAdapter,
    output_path: str | Path,
    *,
    device: str,
    max_samples: int | None,
    max_new_tokens: int,
    projection_dim: int,
    projection_method: str,
    seed: int,
    split_manifest: DatasetSplitManifest | None = None,
    selected_split: str = "fit",
    overwrite: bool = False,
    profiler_factory: ProfilerFactory = Profiler,
) -> dict[str, Any]:
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive")
    if projection_dim <= 0:
        raise ValueError("projection_dim must be positive")

    destination = Path(output_path).resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Mapping output already exists; pass overwrite=True: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp")
    temporary.unlink(missing_ok=True)
    temporary.touch()

    profiler = None
    samples = 0
    scanned_samples = 0
    mapping_rows = 0
    try:
        model_adapter.load(device)
        profiler = profiler_factory(
            model_adapter.model,
            proj_dim=projection_dim,
            proj_method=projection_method,
            seed=seed,
        )
        profiler.register()
        for sequence_id, sample in enumerate(
            dataset_adapter.iter_samples(max_samples=max_samples)
        ):
            scanned_samples += 1
            assignment = (
                None
                if split_manifest is None
                else split_manifest.assignment_for_orig_id(sample.orig_id)
            )
            if assignment is not None:
                if assignment.sequence_id != sequence_id:
                    raise ValueError(
                        "Dataset iteration order differs from split manifest "
                        f"for {sample.orig_id}"
                    )
                if (
                    assignment.semantic_group_id
                    != sample.semantic_group_id
                ):
                    raise ValueError(
                        "Dataset semantic_group_id differs from split "
                        f"manifest for {sample.orig_id}"
                    )
                if assignment.split != selected_split:
                    continue
            model_adapter.generate(
                sample.question,
                sample.image,
                max_new_tokens=max_new_tokens,
            )
            profiler.finalize()
            mapping_rows += int(
                profiler.save_attn_proj_interlayer_jsonl(
                    str(temporary),
                    sample_id=sequence_id,
                    orig_id=sample.orig_id,
                    semantic_group_id=sample.semantic_group_id,
                    split=(
                        selected_split
                        if assignment is None
                        else assignment.split
                    ),
                )
            )
            profiler.reset(clear_stats=True)
            samples += 1
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        if profiler is not None:
            profiler.unregister()
        model_adapter.close()

    temporary.replace(destination)
    return {
        "samples": samples,
        "scanned_samples": scanned_samples,
        "selected_split": selected_split,
        "mapping_rows": mapping_rows,
        "projection_dim": projection_dim,
        "projection_method": projection_method,
        "seed": seed,
        "output": str(destination),
    }


def run_mapping_job(
    config_path: str | Path,
    job_name: str,
    *,
    repository_root: str | Path,
    device: str,
    max_samples: int | None = None,
    max_new_tokens: int | None = None,
    output_path: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    job = load_pipeline_job(
        config_path,
        job_name,
        repository_root=repository_root,
    )
    split_manifest = load_configured_split_manifest(
        job.dataset_config,
        repository_root=repository_root,
    )
    summary = collect_mapping_samples(
        load_model_adapter(job.model_config_path),
        load_dataset_adapter(job.dataset_config_path),
        output_path or job.paths.mapping_data,
        device=device,
        max_samples=job.max_samples if max_samples is None else max_samples,
        max_new_tokens=(
            job.max_new_tokens if max_new_tokens is None else max_new_tokens
        ),
        projection_dim=job.projection_dim,
        projection_method=job.projection_method,
        seed=job.profiler_seed,
        split_manifest=split_manifest,
        selected_split="fit",
        overwrite=overwrite,
    )
    summary.update(
        {
            "job": job.name,
            "model": job.model_name,
            "dataset": job.dataset_name,
            "device": device,
        }
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def train_mapping_checkpoint(
    mapping_data: str | Path,
    output_path: str | Path,
    *,
    trainer: MappingTrainer,
    trainer_kwargs: Mapping[str, Any],
    device: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    source = Path(mapping_data).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Missing mapping training data: {source}")
    destination = Path(output_path).resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Mapping model already exists; pass overwrite=True: {destination}"
        )
    reserved = {"jsonl_path", "save_best_path", "device"}
    conflicts = reserved.intersection(trainer_kwargs)
    if conflicts:
        raise ValueError(
            f"mapping_training.kwargs cannot override {sorted(conflicts)}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        result = trainer(
            jsonl_path=str(source),
            save_best_path=str(temporary),
            device=device,
            **dict(trainer_kwargs),
        )
        if not temporary.is_file():
            raise RuntimeError(
                "Mapping trainer completed without producing a checkpoint: "
                f"{temporary}"
            )
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    metrics = {}
    if isinstance(result, tuple) and len(result) >= 2:
        metrics = _json_safe(result[1])
    return {
        "stage": "train_mapping",
        "input": str(source),
        "output": str(destination),
        "device": device,
        "metrics": metrics,
    }


def run_mapping_training_job(
    config_path: str | Path,
    job_name: str,
    *,
    repository_root: str | Path,
    device: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    job = load_pipeline_job(
        config_path,
        job_name,
        repository_root=repository_root,
    )
    config = job.mapping_training_config
    trainer = import_symbol(str(config["trainer"]))
    configured_kwargs = config.get("kwargs", {})
    if not isinstance(configured_kwargs, Mapping):
        raise ValueError("mapping_training.kwargs must be a mapping")
    trainer_kwargs = dict(configured_kwargs)
    mapping_kwargs = job.injection_config.get("mapping_kwargs")
    if not isinstance(mapping_kwargs, Mapping):
        raise ValueError("injection.mapping_kwargs must be a mapping")
    trainer_kwargs.setdefault("model_kwargs", dict(mapping_kwargs))

    summary = train_mapping_checkpoint(
        job.paths.mapping_data,
        job.paths.mapping_model,
        trainer=trainer,
        trainer_kwargs=trainer_kwargs,
        device=device,
        overwrite=overwrite,
    )
    summary.update(
        {
            "job": job.name,
            "model": job.model_name,
            "dataset": job.dataset_name,
            "trainer": str(config["trainer"]),
        }
    )
    split_manifest = load_configured_split_manifest(
        job.dataset_config,
        repository_root=repository_root,
    )
    metadata_path = job.paths.mapping_model.with_suffix(".metadata.json")
    metadata = {
        **summary,
        "mapping_architecture": dict(mapping_kwargs),
        "training_parameters": dict(configured_kwargs),
        "mapping_data_sha256": _sha256_file(job.paths.mapping_data),
        "checkpoint_sha256": _sha256_file(job.paths.mapping_model),
        "split_assignment_sha256": split_manifest.assignment_sha256,
    }
    _atomic_write_json(metadata_path, metadata)
    summary["metadata"] = str(metadata_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(_json_safe(value), stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temporary.replace(path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    try:
        import numpy as np
        import torch

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
    except ImportError:
        pass
    return value
