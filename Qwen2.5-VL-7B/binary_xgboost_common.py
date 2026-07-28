"""Compatibility module for the experimental layer-pair detector sweep."""

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from detect_sdc.detector import layer_pair_sweep as _shared  # noqa: E402


run_binary_xgboost = _shared.run_binary_xgboost
run_binary_xgboost_compare_nan_modes = (
    _shared.run_binary_xgboost_compare_nan_modes
)


def __getattr__(name: str):
    return getattr(_shared, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_shared)))
