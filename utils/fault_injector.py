"""Compatibility import for the canonical fault injector."""

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from detect_sdc.fault_injector import FaultInjector  # noqa: E402

__all__ = ["FaultInjector"]
