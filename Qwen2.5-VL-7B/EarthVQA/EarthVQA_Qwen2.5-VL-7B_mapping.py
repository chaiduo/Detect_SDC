"""Compatibility entry point for shared mapping collection."""

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from detect_sdc.pipeline.compat import mapping_main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(
        mapping_main("qwen25_vl_earthvqa", repository_root=REPOSITORY_ROOT)
    )
