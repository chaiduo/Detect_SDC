"""Dynamic adapter loading from YAML configuration."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Any, Mapping

from ..config import load_yaml
from .datasets.base import DatasetAdapter
from .models.base import ModelAdapter


def import_symbol(dotted_path: str) -> Any:
    module_name, separator, symbol_name = dotted_path.rpartition(".")
    if not separator or not module_name or not symbol_name:
        raise ValueError(f"Invalid dotted object path: {dotted_path}")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, symbol_name)
    except AttributeError as error:
        raise ImportError(
            f"Module {module_name} does not define {symbol_name}"
        ) from error


def load_dataset_adapter(config_path: str | Path) -> DatasetAdapter:
    return create_dataset_adapter(load_yaml(config_path))


def create_dataset_adapter(config: Mapping[str, Any]) -> DatasetAdapter:
    adapter = _instantiate(config, "dataset")
    _require_interface(
        adapter,
        "DatasetAdapter",
        methods=("iter_samples",),
        attributes=("name",),
    )
    return adapter


def load_model_adapter(config_path: str | Path) -> ModelAdapter:
    return create_model_adapter(load_yaml(config_path))


def create_model_adapter(config: Mapping[str, Any]) -> ModelAdapter:
    adapter = _instantiate(config, "model")
    _require_interface(
        adapter,
        "ModelAdapter",
        methods=("load", "generate", "close"),
        attributes=("model",),
    )
    return adapter


def _instantiate(config: Mapping[str, Any], kind: str) -> Any:
    dotted_path = str(config.get("adapter", "")).strip()
    if not dotted_path:
        raise ValueError(f"{kind} configuration requires an adapter path")
    adapter_class = import_symbol(dotted_path)
    factory = getattr(adapter_class, "from_config", None)
    if factory is None:
        raise TypeError(f"{dotted_path} must define from_config(config)")
    return factory(config)


def _require_interface(
    instance: Any,
    interface_name: str,
    *,
    methods: tuple[str, ...],
    attributes: tuple[str, ...],
) -> None:
    missing = []
    for name in methods:
        member = inspect.getattr_static(instance, name, None)
        if not callable(member):
            missing.append(name)
    for name in attributes:
        if inspect.getattr_static(instance, name, None) is None:
            missing.append(name)
    if missing:
        raise TypeError(
            f"Configured object does not implement {interface_name}; "
            f"missing: {sorted(missing)}"
        )
