"""Shared deterministic-inference controls for model adapters."""

from __future__ import annotations

import os
from typing import Any


DEFAULT_DETERMINISTIC_SEED = 42


def prepare_deterministic_environment(enabled: bool) -> None:
    if enabled:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def configure_deterministic_execution(
    torch: Any,
    *,
    enabled: bool,
    seed: int,
) -> None:
    if not enabled:
        return
    seed_torch(torch, seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False


def seed_torch(torch: Any, seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
