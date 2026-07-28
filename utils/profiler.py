"""Compatibility import for the canonical profiler."""

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from detect_sdc.profiler import Profiler  # noqa: E402

__all__ = ["Profiler"]
