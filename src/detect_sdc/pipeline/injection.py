"""Shared fault-injection execution for all model and dataset jobs."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from ..adapters import load_dataset_adapter, load_model_adapter
from ..adapters.datasets.base import DatasetAdapter, DatasetSample
from ..adapters.models.base import ModelAdapter
from ..adapters.registry import import_symbol
from ..config import load_yaml
from ..dataset_splits import DatasetSplitManifest
from ..fault_injector import FaultInjector
from ..profiler import Profiler
from .jobs import load_pipeline_job
from .split import load_configured_split_manifest


@dataclass(frozen=True)
class CleanAnswerIndex:
    by_sequence: Mapping[int, str]
    by_orig_id: Mapping[str, str]

    def get(self, sequence_id: int, orig_id: str) -> str:
        if orig_id in self.by_orig_id:
            return self.by_orig_id[orig_id]
        if sequence_id in self.by_sequence:
            return self.by_sequence[sequence_id]
        raise KeyError(
            f"Golden answer is missing for sequence={sequence_id}, "
            f"orig_id={orig_id}"
        )


def load_clean_answers(path: str | Path) -> CleanAnswerIndex:
    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        records = json.load(stream)
    if not isinstance(records, list):
        raise ValueError("Golden answers must be a JSON list")

    by_sequence: dict[int, str] = {}
    by_orig_id: dict[str, str] = {}
    for position, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"Golden record {position} must be a mapping")
        sequence_id = int(record.get("id", position))
        answer = str(record.get("pre_answer", ""))
        if sequence_id in by_sequence:
            raise ValueError(f"Duplicate golden sequence id: {sequence_id}")
        by_sequence[sequence_id] = answer
        orig_id = record.get("orig_id")
        if orig_id is not None:
            stable_id = str(orig_id)
            if stable_id in by_orig_id:
                raise ValueError(f"Duplicate golden orig_id: {stable_id}")
            by_orig_id[stable_id] = answer
    return CleanAnswerIndex(by_sequence, by_orig_id)


@dataclass(frozen=True)
class _InjectionResumeState:
    start_run_index: int
    rows_written: int
    sdc_rows: int


def run_injection_samples(
    model_adapter: ModelAdapter,
    dataset_adapter: DatasetAdapter,
    mapping_model: Any,
    clean_answers: CleanAnswerIndex,
    output_path: str | Path,
    *,
    device: str,
    max_samples: int | None,
    max_new_tokens: int,
    projection_dim: int,
    projection_method: str,
    profiler_seed: int,
    fault_runs: int,
    num_bits: int,
    fault_seed: int,
    bit_policy: str = "random",
    telemetry_max_steps: int | None = None,
    split_manifest: DatasetSplitManifest | None = None,
    overwrite: bool = False,
    resume_from_run: int | None = None,
    profiler_factory: Callable[..., Any] = Profiler,
    injector_factory: Callable[..., Any] = FaultInjector,
    auxiliary_monitor_factory: Callable[[Any], Any] | None = None,
    auxiliary_scorer: Callable[[Any], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    bit_policy = str(bit_policy).strip().lower()
    if (auxiliary_monitor_factory is None) != (auxiliary_scorer is None):
        raise ValueError(
            "auxiliary_monitor_factory and auxiliary_scorer must be "
            "provided together"
        )
    _validate_injection_parameters(
        max_samples=max_samples,
        fault_runs=fault_runs,
        num_bits=num_bits,
        bit_policy=bit_policy,
        resume_from_run=resume_from_run,
    )
    destination = Path(output_path).resolve()
    if resume_from_run is not None and overwrite:
        raise ValueError("resume_from_run and overwrite cannot be used together")
    if destination.exists() and (resume_from_run is not None or not overwrite):
        raise FileExistsError(
            f"Injection output already exists; pass overwrite=True: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp")
    resume_state = None
    if resume_from_run is None:
        temporary.unlink(missing_ok=True)
    else:
        resume_state = _prepare_injection_resume(
            temporary,
            resume_from_run=resume_from_run,
            fault_runs=fault_runs,
        )
        print(
            "[inject] resume_from_run="
            f"{resume_from_run} retained_rows={resume_state.rows_written}"
        )

    profiler = None
    injector = None
    auxiliary_monitor = None
    if resume_state is None:
        total_generated = 0
        rows_written = 0
        sdc_rows = 0
        run_indices = [None, *range(fault_runs)]
        stream_mode = "w"
    else:
        sample_count = len(clean_answers.by_sequence)
        if max_samples is not None:
            sample_count = min(sample_count, max_samples)
        total_generated = sample_count * (resume_from_run + 1)
        rows_written = resume_state.rows_written
        sdc_rows = resume_state.sdc_rows
        run_indices = list(range(resume_from_run, fault_runs))
        stream_mode = "a"
    try:
        model_adapter.load(device)
        mapping_model = mapping_model.to(device).eval()
        profiler = profiler_factory(
            model_adapter.model,
            proj_dim=projection_dim,
            proj_method=projection_method,
            seed=profiler_seed,
        )
        injector = injector_factory(model_adapter.model, mode="activation")
        profiler.register()
        if auxiliary_monitor_factory is not None:
            auxiliary_monitor = auxiliary_monitor_factory(model_adapter.model)
            auxiliary_monitor.register()

        with temporary.open(stream_mode, encoding="utf-8") as stream:
            for run_index in run_indices:
                injected = run_index is not None
                random.seed(
                    fault_seed if run_index is None else fault_seed + run_index
                )
                for sequence_id, sample in enumerate(
                    dataset_adapter.iter_samples(max_samples=max_samples)
                ):
                    assignment = (
                        None
                        if split_manifest is None
                        else split_manifest.assignment_for_orig_id(
                            sample.orig_id
                        )
                    )
                    if assignment is not None and assignment.sequence_id != sequence_id:
                        raise ValueError(
                            "Dataset iteration order differs from split manifest "
                            f"for {sample.orig_id}"
                        )
                    if (
                        assignment is not None
                        and assignment.semantic_group_id
                        != sample.semantic_group_id
                    ):
                        raise ValueError(
                            "Dataset semantic_group_id differs from split "
                            f"manifest for {sample.orig_id}"
                        )
                    profiler.reset(clear_stats=True)
                    injector.reset()
                    if injected:
                        injector.set_num_bits(num_bits)
                        injector.set_bit_policy(bit_policy)
                        injector.inject()
                    injector.register_step_hooks()
                    if auxiliary_monitor is not None:
                        auxiliary_monitor.start_sample()
                    try:
                        prediction = model_adapter.generate(
                            sample.question,
                            sample.image,
                            max_new_tokens=max_new_tokens,
                        )
                        profiler.finalize()
                        telemetry = (
                            profiler.get_attn_proj_model_compare_result(
                                predictor_model=mapping_model,
                                device=device,
                                include_vectors=False,
                                max_steps=telemetry_max_steps,
                            )
                        )
                        auxiliary_values = (
                            {}
                            if auxiliary_monitor is None
                            else dict(
                                auxiliary_scorer(
                                    auxiliary_monitor.finish_sample()
                                )
                            )
                        )
                    finally:
                        injector.unregister_hooks()

                    if injected:
                        injection_succeeded = bool(
                            getattr(
                                injector,
                                "select_target_has_injected",
                                injector.fault_info is not None,
                            )
                        )
                        if not injection_succeeded or injector.fault_info is None:
                            raise RuntimeError(
                                "Activation fault hook did not inject for "
                                f"orig_id={sample.orig_id}, run={run_index}"
                            )

                    clean_answer = clean_answers.get(
                        sequence_id,
                        sample.orig_id,
                    )
                    is_sdc = clean_answer.strip() != prediction.strip()
                    total_generated += 1
                    sdc_rows += int(is_sdc)
                    record = _build_result_record(
                        sequence_id,
                        sample,
                        clean_answer,
                        prediction,
                        telemetry,
                        injector.fault_info,
                        run_index=run_index,
                        injected=injected,
                        split=(
                            None
                            if assignment is None
                            else assignment.split
                        ),
                        auxiliary_values=auxiliary_values,
                    )
                    stream.write(
                        json.dumps(
                            _json_safe(record),
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    rows_written += 1
    except BaseException:
        if resume_state is None:
            temporary.unlink(missing_ok=True)
        raise
    finally:
        if injector is not None:
            injector.unregister_hooks()
        if profiler is not None:
            profiler.unregister()
        if auxiliary_monitor is not None:
            auxiliary_monitor.unregister()
        model_adapter.close()

    temporary.replace(destination)
    return {
        "generated": total_generated,
        "rows_written": rows_written,
        "sdc_rows_observed": sdc_rows,
        "clean_runs": 1,
        "fault_runs": fault_runs,
        "retention_policy": "all_runs",
        "num_bits": num_bits,
        "bit_policy": bit_policy,
        "telemetry_max_steps": telemetry_max_steps,
        "resumed_from_run": resume_from_run,
        "output": str(destination),
    }


def run_injection_job(
    config_path: str | Path,
    job_name: str,
    *,
    repository_root: str | Path,
    device: str,
    max_samples: int | None = None,
    output_path: str | Path | None = None,
    golden_path: str | Path | None = None,
    mapping_model_path: str | Path | None = None,
    overwrite: bool = False,
    resume_from_run: int | None = None,
    telemetry_max_steps: int | None = None,
    auxiliary_monitor_factory: Callable[[Any], Any] | None = None,
    auxiliary_scorer: Callable[[Any], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    job = load_pipeline_job(
        config_path,
        job_name,
        repository_root=repository_root,
    )
    injection = job.injection_config
    experiment = load_yaml(config_path)
    features = experiment.get("features", {})
    if not isinstance(features, Mapping):
        raise ValueError("Experiment features must be a mapping")
    configured_max_steps = int(features.get("last_k_steps", 2))
    if telemetry_max_steps is None:
        telemetry_max_steps = configured_max_steps
    if telemetry_max_steps <= 0:
        raise ValueError("telemetry_max_steps must be positive")
    if features.get("step_window") != "prefix":
        raise ValueError("Injection telemetry requires prefix step_window")
    split_manifest = load_configured_split_manifest(
        job.dataset_config,
        repository_root=repository_root,
    )
    checkpoint = Path(mapping_model_path or job.paths.mapping_model).resolve()
    mapping_model = load_mapping_model(injection, checkpoint, device=device)

    summary = run_injection_samples(
        load_model_adapter(job.model_config_path),
        load_dataset_adapter(job.dataset_config_path),
        mapping_model,
        load_clean_answers(golden_path or job.paths.profile_output),
        output_path or job.paths.injected_output,
        device=device,
        max_samples=job.max_samples if max_samples is None else max_samples,
        max_new_tokens=job.max_new_tokens,
        projection_dim=job.projection_dim,
        projection_method=job.projection_method,
        profiler_seed=job.profiler_seed,
        fault_runs=int(injection.get("fault_runs", 10)),
        num_bits=int(injection.get("num_bits", 2)),
        fault_seed=int(injection.get("seed", 42)),
        bit_policy=str(injection.get("bit_policy", "random")),
        telemetry_max_steps=telemetry_max_steps,
        split_manifest=split_manifest,
        overwrite=overwrite,
        resume_from_run=resume_from_run,
        auxiliary_monitor_factory=auxiliary_monitor_factory,
        auxiliary_scorer=auxiliary_scorer,
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


def load_mapping_model(
    injection_config: Mapping[str, Any],
    checkpoint_path: str | Path,
    *,
    device: str,
) -> Any:
    import torch

    mapping_class = import_symbol(str(injection_config["mapping_class"]))
    mapping_kwargs = injection_config.get("mapping_kwargs", {})
    if not isinstance(mapping_kwargs, Mapping):
        raise ValueError("injection.mapping_kwargs must be a mapping")
    checkpoint = Path(checkpoint_path).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Mapping model does not exist: {checkpoint}")
    mapping_model = mapping_class(**mapping_kwargs)
    mapping_model.load_state_dict(torch.load(checkpoint, map_location=device))
    return mapping_model


def _build_result_record(
    sequence_id: int,
    sample: DatasetSample,
    clean_answer: str,
    prediction: str,
    telemetry: Any,
    fault: Any,
    *,
    run_index: int | None,
    injected: bool,
    split: str | None,
    auxiliary_values: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = dict(sample.metadata)
    run_identity = "clean" if not injected else f"fault:{run_index}"
    record = {
        "id": sequence_id,
        "orig_id": sample.orig_id,
        "semantic_group_id": sample.semantic_group_id,
        "split": split,
        "sample_uid": f"{sample.orig_id}:{run_identity}",
        "before_score": 0.0,
        "after_score": 0.0,
        "dtel_score": 0.0,
        "is_sdc": int(clean_answer.strip() != prediction.strip()),
        "fault": fault,
        "question": sample.question,
        "gt_answer": sample.ground_truth,
        "clean_answer": clean_answer,
        "pred_answer": prediction,
        "mean_std_cos": telemetry,
        "run_index": run_index,
        "injected": injected,
        "metadata": metadata,
    }
    for key in ("image_path", "source_file", "row_idx"):
        if key in metadata:
            record[key] = metadata[key]
    if "image_filename" in metadata and "image_path" not in record:
        record["image_path"] = metadata["image_filename"]
    for key, value in dict(auxiliary_values or {}).items():
        if key in record:
            raise ValueError(
                f"Auxiliary score collides with canonical field: {key}"
            )
        record[str(key)] = value
    return record


def _validate_injection_parameters(
    *,
    max_samples: int | None,
    fault_runs: int,
    num_bits: int,
    bit_policy: str,
    resume_from_run: int | None,
) -> None:
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive")
    if fault_runs < 0:
        raise ValueError("fault_runs must be non-negative")
    if num_bits <= 0:
        raise ValueError("num_bits must be positive")
    if bit_policy not in FaultInjector.BIT_POLICIES:
        raise ValueError(
            f"bit_policy must be one of {FaultInjector.BIT_POLICIES}, "
            f"got {bit_policy!r}"
        )
    if resume_from_run is not None and not 0 <= resume_from_run < fault_runs:
        raise ValueError(
            "resume_from_run must identify an existing fault run"
        )


def _prepare_injection_resume(
    temporary: Path,
    *,
    resume_from_run: int,
    fault_runs: int,
) -> _InjectionResumeState:
    if not temporary.is_file():
        raise FileNotFoundError(
            f"Injection resume file does not exist: {temporary}"
        )

    recovery = temporary.with_name(f"{temporary.name}.resume")
    recovery.unlink(missing_ok=True)
    rows_written = 0
    sdc_rows = 0
    last_fault_run: int | None = None
    try:
        with (
            temporary.open("r", encoding="utf-8") as source,
            recovery.open("w", encoding="utf-8") as destination,
        ):
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    raise ValueError(
                        f"Empty injection record at line {line_number}"
                    )
                record = json.loads(line)
                run_index = record.get("run_index")
                if run_index is not None:
                    if (
                        not isinstance(run_index, int)
                        or isinstance(run_index, bool)
                        or not 0 <= run_index < fault_runs
                    ):
                        raise ValueError(
                            f"Invalid run_index at line {line_number}: "
                            f"{run_index!r}"
                        )
                    if (
                        last_fault_run is not None
                        and run_index < last_fault_run
                    ):
                        raise ValueError(
                            "Injection resume records are not ordered by run"
                        )
                    last_fault_run = run_index
                elif last_fault_run is not None:
                    raise ValueError(
                        "Clean injection record appears after a fault run"
                    )

                if run_index is None or run_index < resume_from_run:
                    destination.write(line)
                    rows_written += 1
                    sdc_rows += int(bool(record.get("is_sdc", 0)))

        if last_fault_run != resume_from_run:
            raise ValueError(
                f"Resume file ends in run {last_fault_run}; "
                f"requested run {resume_from_run}"
            )
        recovery.replace(temporary)
    except BaseException:
        recovery.unlink(missing_ok=True)
        raise

    return _InjectionResumeState(
        start_run_index=resume_from_run,
        rows_written=rows_written,
        sdc_rows=sdc_rows,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
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
