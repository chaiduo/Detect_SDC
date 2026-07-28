"""Compatibility entry point for the shared Qwen VQAv2 mapping trainer."""

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from detect_sdc.mapping.qwen_vqav2 import (  # noqa: E402
    LayerAwareResidualMLP as LayerAwareResidualMLP,
    main as main,
    train_model as train_model,
)


if __name__ == "__main__":
    main()
