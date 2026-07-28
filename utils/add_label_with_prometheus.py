"""Compatibility CLI for the shared Prometheus labeling stage."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from detect_sdc.labeling import (  # noqa: E402
    ABSOLUTE_PROMPT as ABSOLUTE_PROMPT,
    ABS_SYSTEM_PROMPT as ABS_SYSTEM_PROMPT,
    RUBRIC as RUBRIC,
    PrometheusJudge,
    build_prompt as build_prompt,
    extract_score as extract_score,
    label_jsonl,
    quality_score_to_significance as quality_score_to_significance,
)


DEFAULT_INPUT_FILE = (
    REPOSITORY_ROOT
    / "Qwen2.5-VL-7B/EarthVQA/json/"
    "detect_EarthVQA_Qwen_with_sem_project.jsonl"
)
MODEL_PATH = "/data01/cd_workspace/llm/prometheus-7b-v2.0"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        "--input-file",
        default=str(DEFAULT_INPUT_FILE),
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", "--batch-size", type=int, default=64)
    parser.add_argument("--model-path", default=MODEL_PATH)
    return parser.parse_args()


def resolve_device(args):
    return f"cuda:{args.gpu}" if args.gpu is not None else args.device


def build_output_file(input_file):
    path = Path(input_file)
    return str(path.with_name(f"{path.stem}_labeled{path.suffix}"))


def build_debug_file(input_file):
    path = Path(input_file)
    return str(
        path.with_name(
            f"{path.stem}_prometheus_parse_failed{path.suffix}"
        )
    )


def main():
    args = parse_args()
    output = args.output or build_output_file(args.input_file)
    judge = PrometheusJudge(
        args.model_path,
        device=resolve_device(args),
        max_new_tokens=200,
    )
    try:
        summary = label_jsonl(
            args.input_file,
            output,
            judge,
            batch_size=args.batch_size,
            debug_path=build_debug_file(args.input_file),
            overwrite=True,
        )
    finally:
        judge.close()
    print(f"Done: {summary}")


if __name__ == "__main__":
    main()
