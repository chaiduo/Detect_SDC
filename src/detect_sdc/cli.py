"""Unified command-line entry point for incremental project migration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .baseline import freeze_baseline
from .config import load_yaml
from .experiment import load_experiment
from .pipeline import PipelineStage


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="detect-sdc")
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline_parser = subparsers.add_parser("baseline", help="Manage reproducibility baselines")
    baseline_subparsers = baseline_parser.add_subparsers(dest="baseline_command", required=True)
    freeze_parser = baseline_subparsers.add_parser("freeze", help="Freeze current data and metrics")
    freeze_parser.add_argument("--spec", required=True, help="Baseline specification YAML")
    freeze_parser.add_argument("--output", required=True, help="Output baseline YAML")
    freeze_parser.add_argument(
        "--repository-root",
        default=str(REPOSITORY_ROOT),
        help="Repository root used to resolve relative paths",
    )

    config_parser = subparsers.add_parser("config", help="Inspect project configuration")
    config_subparsers = config_parser.add_subparsers(dest="config_command", required=True)
    show_parser = config_subparsers.add_parser("show", help="Load and print a YAML configuration")
    show_parser.add_argument("path", help="YAML file to inspect")
    validate_parser = config_subparsers.add_parser("validate", help="Validate an experiment matrix")
    validate_parser.add_argument("path", help="Experiment YAML to validate")
    validate_parser.add_argument(
        "--repository-root",
        default=str(REPOSITORY_ROOT),
        help="Repository root used to resolve referenced configuration files",
    )

    featurize_parser = subparsers.add_parser(
        "featurize",
        help="Extract telemetry features and split them by orig_id",
    )
    featurize_parser.add_argument("--job", required=True, help="Feature job name")
    featurize_parser.add_argument(
        "--config",
        default=str(REPOSITORY_ROOT / "configs/experiments/current.yaml"),
        help="Experiment YAML containing the feature job",
    )
    featurize_parser.add_argument(
        "--repository-root",
        default=str(REPOSITORY_ROOT),
        help="Repository root used to resolve input and output paths",
    )
    featurize_parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Process at most this many input samples",
    )
    featurize_parser.add_argument(
        "--train-output",
        default=None,
        help="Override the configured training CSV path",
    )
    featurize_parser.add_argument(
        "--valid-output",
        default=None,
        help="Override the configured validation CSV path",
    )

    train_parser = subparsers.add_parser(
        "train",
        help="Train and evaluate the significant-SDC detector",
    )
    train_parser.add_argument("--job", required=True, help="Experiment job name")
    train_parser.add_argument(
        "--config",
        default=str(REPOSITORY_ROOT / "configs/experiments/current.yaml"),
        help="Experiment YAML containing detector parameters",
    )
    train_parser.add_argument(
        "--repository-root",
        default=str(REPOSITORY_ROOT),
        help="Repository root used to resolve feature CSV paths",
    )
    train_parser.add_argument(
        "--output-dir",
        default=None,
        help="Override the configured detector output directory",
    )

    label_parser = subparsers.add_parser(
        "label",
        help="Assign Prometheus quality and significance labels",
    )
    label_parser.add_argument("--job", required=True, help="Experiment job name")
    label_parser.add_argument(
        "--config",
        default=str(REPOSITORY_ROOT / "configs/experiments/current.yaml"),
        help="Experiment YAML containing labeling parameters",
    )
    label_parser.add_argument(
        "--repository-root",
        default=str(REPOSITORY_ROOT),
        help="Repository root used to resolve JSONL paths",
    )
    label_parser.add_argument("--device", default="cuda:0", help="Judge device")
    label_parser.add_argument("--batch-size", type=int, default=64)
    label_parser.add_argument("--input", default=None, help="Override input JSONL")
    label_parser.add_argument("--output", default=None, help="Override output JSONL")
    label_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing label outputs",
    )

    run_parser = subparsers.add_parser(
        "run",
        help="Run one configured pipeline stage",
    )
    run_parser.add_argument("--job", required=True, help="Experiment job name")
    run_parser.add_argument(
        "--stage",
        required=True,
        action="append",
        choices=[stage.value for stage in PipelineStage],
    )
    run_parser.add_argument(
        "--config",
        default=str(REPOSITORY_ROOT / "configs/experiments/current.yaml"),
    )
    run_parser.add_argument(
        "--repository-root",
        default=str(REPOSITORY_ROOT),
    )
    run_parser.add_argument("--device", default="cuda:0")
    run_parser.add_argument("--max-samples", type=int, default=None)
    run_parser.add_argument("--batch-size", type=int, default=64)
    run_parser.add_argument("--overwrite", action="store_true")
    run_parser.add_argument("--dry-run", action="store_true")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "baseline" and args.baseline_command == "freeze":
        manifest = freeze_baseline(
            spec_path=args.spec,
            output_path=args.output,
            repository_root=args.repository_root,
        )
        print(f"Baseline frozen: {args.output}")
        print(f"Experiments: {len(manifest['experiments'])}")
        return 0

    if args.command == "config" and args.config_command == "show":
        print(json.dumps(load_yaml(args.path), ensure_ascii=False, indent=2))
        return 0

    if args.command == "config" and args.config_command == "validate":
        experiment = load_experiment(args.path)
        experiment.validate_references(args.repository_root)
        print(f"Configuration valid: {experiment.name}")
        print(f"Experiment pairs: {len(experiment.matrix)}")
        print(f"Pipeline stages: {len(experiment.stages)}")
        return 0

    if args.command == "featurize":
        from .features.jobs import run_feature_job

        run_feature_job(
            config_path=args.config,
            job_name=args.job,
            repository_root=args.repository_root,
            max_samples=args.max_samples,
            train_output=args.train_output,
            valid_output=args.valid_output,
        )
        return 0

    if args.command == "train":
        from .detector import run_detector_job

        run_detector_job(
            config_path=args.config,
            job_name=args.job,
            repository_root=args.repository_root,
            output_dir=args.output_dir,
        )
        return 0

    if args.command == "label":
        from .labeling import run_label_job

        run_label_job(
            config_path=args.config,
            job_name=args.job,
            repository_root=args.repository_root,
            device=args.device,
            batch_size=args.batch_size,
            input_path=args.input,
            output_path=args.output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "run":
        from .pipeline.runner import run_stage

        for stage in args.stage:
            run_stage(
                config_path=args.config,
                job_name=args.job,
                stage=stage,
                repository_root=args.repository_root,
                device=args.device,
                max_samples=args.max_samples,
                batch_size=args.batch_size,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )
        return 0

    parser.error("Unsupported command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
