"""Compatibility CLIs for legacy profile, mapping, and injection scripts."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..adapters import create_dataset_adapter, create_model_adapter
from .injection import (
    load_clean_answers,
    load_mapping_model,
    run_injection_samples,
)
from .jobs import load_pipeline_job
from .mapping import collect_mapping_samples
from .profile import profile_samples


def add_adapter_override_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model_path")
    parser.add_argument("--model_base")
    parser.add_argument("--data_dir")
    parser.add_argument("--dataset_json")
    parser.add_argument("--val_file")
    parser.add_argument("--parquet_path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--max_new_tokens", type=int)
    parser.add_argument("--overwrite", action="store_true")


def apply_adapter_overrides(
    model_config: dict[str, Any],
    dataset_config: dict[str, Any],
    dataset_name: str,
    args: argparse.Namespace,
) -> None:
    if args.model_path:
        model_config["model_path"] = args.model_path
    if args.model_base is not None:
        model_config["model_base"] = args.model_base or None

    paths = dataset_config.get("paths")
    if not isinstance(paths, Mapping):
        raise ValueError("dataset paths must be a mapping")
    updated_paths = dict(paths)
    if dataset_name == "earthvqa":
        if args.data_dir:
            updated_paths["images"] = args.data_dir
        if args.dataset_json:
            updated_paths["annotations"] = args.dataset_json
    elif dataset_name == "lingoqa":
        if args.data_dir:
            updated_paths["images"] = args.data_dir
        if args.val_file:
            updated_paths["annotations"] = args.val_file
    elif dataset_name == "vqav2" and args.parquet_path:
        updated_paths["parquet"] = args.parquet_path
    dataset_config["paths"] = updated_paths


def profile_main(
    job_name: str,
    *,
    repository_root: str | Path,
    argv: Sequence[str] | None = None,
) -> int:
    parser = argparse.ArgumentParser()
    add_adapter_override_arguments(parser)
    parser.add_argument("--golden_json")
    args = parser.parse_args(argv)
    _, job, model_config, dataset_config = _load_compat_job(
        repository_root,
        job_name,
        args,
    )

    rows = profile_samples(
        create_model_adapter(model_config),
        create_dataset_adapter(dataset_config),
        device=args.device,
        max_samples=_effective(args.max_samples, job.max_samples),
        max_new_tokens=_effective(
            args.max_new_tokens,
            job.max_new_tokens,
        ),
    )
    output = (
        Path(args.golden_json).resolve()
        if args.golden_json
        else job.paths.profile_output
    )
    _atomic_write_json(output, rows)
    print(json.dumps({"job": job.name, "rows": len(rows), "output": str(output)}))
    return 0


def mapping_main(
    job_name: str,
    *,
    repository_root: str | Path,
    argv: Sequence[str] | None = None,
) -> int:
    parser = argparse.ArgumentParser()
    add_adapter_override_arguments(parser)
    parser.add_argument("--output_jsonl")
    parser.add_argument("--save_dir")
    args = parser.parse_args(argv)
    _, job, model_config, dataset_config = _load_compat_job(
        repository_root,
        job_name,
        args,
    )

    output = _mapping_output_path(args, job.paths.mapping_data)
    summary = collect_mapping_samples(
        create_model_adapter(model_config),
        create_dataset_adapter(dataset_config),
        output,
        device=args.device,
        max_samples=_effective(args.max_samples, job.max_samples),
        max_new_tokens=_effective(
            args.max_new_tokens,
            job.max_new_tokens,
        ),
        projection_dim=job.projection_dim,
        projection_method=job.projection_method,
        seed=job.profiler_seed,
        overwrite=args.overwrite,
    )
    summary["job"] = job.name
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def injection_main(
    job_name: str,
    *,
    repository_root: str | Path,
    argv: Sequence[str] | None = None,
) -> int:
    parser = argparse.ArgumentParser()
    add_adapter_override_arguments(parser)
    parser.add_argument("--output_jsonl")
    parser.add_argument("--golden_json")
    parser.add_argument("--mapping_model")
    args = parser.parse_args(argv)
    _, job, model_config, dataset_config = _load_compat_job(
        repository_root,
        job_name,
        args,
    )
    injection = job.injection_config

    summary = run_injection_samples(
        create_model_adapter(model_config),
        create_dataset_adapter(dataset_config),
        load_mapping_model(
            injection,
            args.mapping_model or job.paths.mapping_model,
            device=args.device,
        ),
        load_clean_answers(args.golden_json or job.paths.profile_output),
        args.output_jsonl or job.paths.injected_output,
        device=args.device,
        max_samples=_effective(args.max_samples, job.max_samples),
        max_new_tokens=_effective(
            args.max_new_tokens,
            job.max_new_tokens,
        ),
        projection_dim=job.projection_dim,
        projection_method=job.projection_method,
        profiler_seed=job.profiler_seed,
        fault_runs=int(injection.get("fault_runs", 8)),
        retain_all_fault_runs=int(
            injection.get("retain_all_fault_runs", 1)
        ),
        num_bits=int(injection.get("num_bits", 2)),
        fault_seed=int(injection.get("seed", 42)),
        overwrite=args.overwrite,
    )
    summary["job"] = job.name
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _load_compat_job(
    repository_root: str | Path,
    job_name: str,
    args: argparse.Namespace,
) -> tuple[Path, Any, dict[str, Any], dict[str, Any]]:
    root = Path(repository_root).resolve()
    job = load_pipeline_job(
        root / "configs/experiments/current.yaml",
        job_name,
        repository_root=root,
    )
    model_config = copy.deepcopy(dict(job.model_config))
    dataset_config = copy.deepcopy(dict(job.dataset_config))
    apply_adapter_overrides(
        model_config,
        dataset_config,
        job.dataset_name,
        args,
    )
    return root, job, model_config, dataset_config


def _effective(override: int | None, configured: int | None) -> int | None:
    return configured if override is None else override


def _mapping_output_path(
    args: argparse.Namespace,
    configured: Path,
) -> Path:
    if args.output_jsonl:
        return Path(args.output_jsonl)
    if args.save_dir:
        return Path(args.save_dir) / "attn_proj_interlayer.jsonl"
    return configured


def _atomic_write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
    temporary.replace(path)
