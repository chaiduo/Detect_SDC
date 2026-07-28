"""Compatibility entry point for shared profiling."""

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from detect_sdc.pipeline.compat import profile_main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(
        profile_main("llava15_lingoqa", repository_root=REPOSITORY_ROOT)
    )
