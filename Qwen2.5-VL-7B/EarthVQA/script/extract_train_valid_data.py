"""Compatibility entry point for the shared feature extraction pipeline."""

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))


def main() -> None:
    from detect_sdc.features.jobs import run_feature_job

    run_feature_job(
        config_path=REPOSITORY_ROOT / "configs/experiments/current.yaml",
        job_name="qwen25_vl_earthvqa",
        repository_root=REPOSITORY_ROOT,
    )


if __name__ == "__main__":
    main()
