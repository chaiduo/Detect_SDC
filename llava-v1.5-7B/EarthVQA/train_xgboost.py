"""Compatibility entry point for the shared detector trainer."""

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))


def main() -> None:
    from detect_sdc.detector import run_detector_job

    run_detector_job(
        config_path=REPOSITORY_ROOT / "configs/experiments/current.yaml",
        job_name="llava15_earthvqa",
        repository_root=REPOSITORY_ROOT,
    )


if __name__ == "__main__":
    main()
