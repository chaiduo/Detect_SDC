"""Model-agnostic clean profiling stage."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from ..adapters import load_dataset_adapter, load_model_adapter
from ..adapters.datasets.base import DatasetAdapter
from ..adapters.models.base import ModelAdapter
from .jobs import load_pipeline_job


def profile_samples(
    model_adapter: ModelAdapter,
    dataset_adapter: DatasetAdapter,
    *,
    device: str,
    max_samples: int | None,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    results = []
    model_adapter.load(device)
    try:
        for sequence_id, sample in enumerate(
            dataset_adapter.iter_samples(max_samples=max_samples)
        ):
            prediction = model_adapter.generate(
                sample.question,
                sample.image,
                max_new_tokens=max_new_tokens,
            )
            results.append(
                {
                    "id": sequence_id,
                    "orig_id": sample.orig_id,
                    "sample_uid": sample.orig_id,
                    "question": sample.question,
                    "gt_answer": sample.ground_truth,
                    "pre_answer": prediction,
                    "score": 0,
                    "metadata": dict(sample.metadata),
                }
            )
    finally:
        model_adapter.close()
    return results


def run_profile_job(
    config_path: str | Path,
    job_name: str,
    *,
    repository_root: str | Path,
    device: str,
    output_path: str | Path | None = None,
    max_samples: int | None = None,
) -> dict[str, Any]:
    job = load_pipeline_job(
        config_path,
        job_name,
        repository_root=repository_root,
    )
    model = load_model_adapter(job.model_config_path)
    dataset = load_dataset_adapter(job.dataset_config_path)
    _set_seed(42)
    results = profile_samples(
        model,
        dataset,
        device=device,
        max_samples=job.max_samples if max_samples is None else max_samples,
        max_new_tokens=job.max_new_tokens,
    )
    destination = (
        Path(output_path).resolve() if output_path else job.paths.profile_output
    )
    _atomic_write_json(destination, results)
    summary = {
        "job": job.name,
        "model": job.model_name,
        "dataset": job.dataset_name,
        "device": device,
        "rows": len(results),
        "output": str(destination),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
    temporary.replace(path)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
