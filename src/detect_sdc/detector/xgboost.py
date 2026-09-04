"""Canonical XGBoost pipeline for binary significant-SDC detection."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier

from ..config import load_yaml
from ..features.jobs import load_feature_job
from ..splitting import split_by_group


CLASS_LABELS = (0, 1)
CLASS_NAMES = {
    0: "non_significant_sdc",
    1: "significant_sdc",
}
EXCLUDED_COLUMNS = {
    "orig_id",
    "semantic_group_id",
    "split",
    "sample_uid",
    "injected",
    "run_index",
    "is_sdc",
    "fault_component",
    "fault_layer_index",
    "fault_op_type",
    "fault_bit_categories",
    "label",
    "significance",
    "ternary_target",
    "binary_target",
    "sdc_target",
    "significant_sdc_target",
    "total_steps",
    "last_k_steps",
    "num_steps_used",
}


@dataclass(frozen=True)
class XGBoostConfig:
    learning_rate: float = 0.01
    n_estimators: int = 10_000
    max_depth: int = 6
    min_child_weight: float = 1.0
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    early_stopping_rounds: int = 500
    test_ratio: float = 0.15
    random_state: int = 42
    n_jobs: int = 8
    device: str = "cpu"
    verbose: int = 200

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
        if self.early_stopping_rounds <= 0:
            raise ValueError("early_stopping_rounds must be positive")
        if not 0 < self.test_ratio < 1:
            raise ValueError("test_ratio must be between 0 and 1")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "XGBoostConfig":
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"Unknown XGBoost configuration fields: {unknown}")
        return cls(**value)


def add_significant_sdc_target(frame: pd.DataFrame) -> pd.DataFrame:
    if "label" not in frame.columns or "significance" not in frame.columns:
        raise ValueError("CSV must contain label and significance columns")

    label = pd.to_numeric(frame["label"], errors="coerce")
    significance = pd.to_numeric(frame["significance"], errors="coerce")
    if label.isna().any():
        raise ValueError("label contains non-numeric values")
    if significance.isna().any():
        raise ValueError("significance contains non-numeric values")

    label = label.astype(int)
    significance = significance.astype(int)
    label_values = set(label.unique().tolist())
    if not label_values.issubset({0, 1, 2}):
        raise ValueError(f"label only supports 0/1/2, got: {sorted(label_values)}")
    if not set(significance.unique().tolist()).issubset({0, 1, 2}):
        raise ValueError("significance only supports 0/1/2")

    if "significant_sdc_target" in frame.columns:
        existing = pd.to_numeric(
            frame["significant_sdc_target"],
            errors="coerce",
        )
        if existing.isna().any() or not set(existing.astype(int).unique()).issubset({0, 1}):
            raise ValueError("significant_sdc_target only supports 0/1")
        target = existing.astype(int)
        if 2 in label_values and not np.array_equal(
            target.to_numpy(),
            (label == 2).astype(int).to_numpy(),
        ):
            raise ValueError("significant_sdc_target is inconsistent with label/significance")
    elif label_values.issubset({0, 1}):
        target = ((label == 1) & (significance == 2)).astype(int)
    else:
        target = (label == 2).astype(int)

    result = frame.copy()
    result["significant_sdc_target"] = target
    return result


def significant_sdc_negative_mask(frame: pd.DataFrame) -> pd.Series:
    """Return the negative class for Significant-SDC calibration/evaluation."""
    if "significant_sdc_target" not in frame.columns:
        raise ValueError("Frame must contain significant_sdc_target")
    target = pd.to_numeric(
        frame["significant_sdc_target"],
        errors="coerce",
    )
    if target.isna().any() or not set(target.astype(int).unique()).issubset({0, 1}):
        raise ValueError("significant_sdc_target only supports 0/1")
    return target.astype(int).eq(0)


def get_feature_columns(frame: pd.DataFrame) -> list[str]:
    columns = [column for column in frame.columns if column not in EXCLUDED_COLUMNS]
    if not columns:
        raise ValueError("No detector feature columns found")
    return columns


def prepare_features(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    features = frame[feature_columns].copy()
    for column in feature_columns:
        features[column] = pd.to_numeric(features[column], errors="coerce")

    float32_max = np.finfo(np.float32).max
    features = features.replace([np.inf, -np.inf], np.nan)
    return features.mask(features.abs() > float32_max, np.nan)


def all_feature_nan_mask(
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> pd.Series:
    return prepare_features(frame, feature_columns).isna().all(axis=1)


def target_counts(values: Any) -> dict[str, Any]:
    target = np.asarray(values, dtype=int)
    total = int(len(target))
    counts = {str(label): int((target == label).sum()) for label in CLASS_LABELS}
    return {
        "total": total,
        "counts": counts,
        "rates": {
            key: float(count / total) if total else None
            for key, count in counts.items()
        },
    }


def label_significance_counts(frame: pd.DataFrame) -> dict[str, int]:
    if "label" not in frame.columns or "significance" not in frame.columns:
        return {}
    counts = (
        frame[["label", "significance"]]
        .value_counts(dropna=False)
        .sort_index()
        .to_dict()
    )
    return {
        f"label={key[0]}, significance={key[1]}": int(value)
        for key, value in counts.items()
    }


def class_weight_vector(values: Any) -> dict[int, float]:
    counts = target_counts(values)["counts"]
    total = sum(counts.values())
    return {
        label: float(total / (len(CLASS_LABELS) * counts[str(label)]))
        if counts[str(label)]
        else 0.0
        for label in CLASS_LABELS
    }


def sample_weights(values: Any) -> np.ndarray:
    weights = class_weight_vector(values)
    target = np.asarray(values, dtype=int)
    return np.asarray([weights[int(value)] for value in target], dtype=float)


def build_classifier(config: XGBoostConfig) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        max_depth=config.max_depth,
        min_child_weight=config.min_child_weight,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        objective="binary:logistic",
        eval_metric=["logloss", "error"],
        early_stopping_rounds=config.early_stopping_rounds,
        tree_method="hist",
        device=config.device,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
    )


def train_binary_model(
    x_train: pd.DataFrame,
    y_train: Any,
    x_test: pd.DataFrame,
    y_test: Any,
    *,
    config: XGBoostConfig,
) -> tuple[XGBClassifier, dict[str, Any]]:
    train_distribution = target_counts(y_train)
    test_distribution = target_counts(y_test)
    missing = [
        label
        for label in CLASS_LABELS
        if train_distribution["counts"][str(label)] == 0
    ]
    if missing:
        raise ValueError(f"training set missing classes: {missing}")

    weights = sample_weights(y_train)
    model = build_classifier(config)
    model.fit(
        x_train,
        y_train,
        sample_weight=weights,
        eval_set=[(x_train, y_train), (x_test, y_test)],
        verbose=config.verbose,
    )
    return model, {
        "train_distribution": train_distribution,
        "test_distribution": test_distribution,
        "class_weights": class_weight_vector(y_train),
        "best_iteration": _optional_model_value(model, "best_iteration", int),
        "best_score": _optional_model_value(model, "best_score", float),
        "parameters": asdict(config),
    }


def binary_metrics(
    y_true: Any,
    y_pred: Any,
    y_probability: np.ndarray,
) -> dict[str, Any]:
    truth = np.asarray(y_true, dtype=int)
    prediction = np.asarray(y_pred, dtype=int)
    if not len(truth):
        empty_class = {
            "support": 0,
            "pred_total": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "precision": 0.0,
            "recall": 0.0,
            "false_positive_rate": 0.0,
            "f1": 0.0,
            "prob_mean": None,
        }
        per_class = {
            str(label): {"name": CLASS_NAMES[label], **empty_class}
            for label in CLASS_LABELS
        }
        return {
            "confusion_matrix": [[0, 0], [0, 0]],
            "accuracy": 0.0,
            "per_class": per_class,
            "classification_report": {},
            "target_class_1": per_class["1"],
            "target_significant_sdc": per_class["1"],
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "weighted_precision": 0.0,
            "weighted_recall": 0.0,
            "weighted_f1": 0.0,
        }
    matrix = confusion_matrix(truth, prediction, labels=CLASS_LABELS)
    report = classification_report(
        truth,
        prediction,
        labels=CLASS_LABELS,
        target_names=[CLASS_NAMES[label] for label in CLASS_LABELS],
        output_dict=True,
        zero_division=0,
    )

    per_class = {}
    for label in CLASS_LABELS:
        name = CLASS_NAMES[label]
        true_mask = truth == label
        predicted_mask = prediction == label
        true_positive = int((true_mask & predicted_mask).sum())
        predicted_total = int(predicted_mask.sum())
        support = int(true_mask.sum())
        per_class[str(label)] = {
            "name": name,
            "support": support,
            "pred_total": predicted_total,
            "tp": true_positive,
            "fp": int((~true_mask & predicted_mask).sum()),
            "fn": int((true_mask & ~predicted_mask).sum()),
            "tn": int((~true_mask & ~predicted_mask).sum()),
            "precision": float(true_positive / predicted_total)
            if predicted_total
            else 0.0,
            "recall": float(true_positive / support) if support else 0.0,
            "false_positive_rate": float(
                (~true_mask & predicted_mask).sum() / (~true_mask).sum()
            )
            if (~true_mask).sum()
            else 0.0,
            "f1": float(report[name]["f1-score"]),
            "prob_mean": float(np.mean(y_probability[:, label]))
            if len(y_probability)
            else None,
        }

    metrics = {
        "confusion_matrix": matrix.tolist(),
        "accuracy": float(accuracy_score(truth, prediction)) if len(truth) else 0.0,
        "per_class": per_class,
        "classification_report": report,
        "target_class_1": per_class["1"],
        "target_significant_sdc": per_class["1"],
    }
    for average in ("macro", "weighted"):
        metrics[f"{average}_precision"] = float(
            precision_score(
                truth,
                prediction,
                labels=CLASS_LABELS,
                average=average,
                zero_division=0,
            )
        )
        metrics[f"{average}_recall"] = float(
            recall_score(
                truth,
                prediction,
                labels=CLASS_LABELS,
                average=average,
                zero_division=0,
            )
        )
        metrics[f"{average}_f1"] = float(
            f1_score(
                truth,
                prediction,
                labels=CLASS_LABELS,
                average=average,
                zero_division=0,
            )
        )
    return metrics


def calibrate_threshold_at_fpr(
    non_sdc_scores: Any,
    *,
    target_fpr: float,
) -> dict[str, Any]:
    """Choose the lowest strict-greater-than threshold within an FPR budget."""

    if not 0.0 <= target_fpr < 1.0:
        raise ValueError("target_fpr must be in [0, 1)")
    scores = np.asarray(non_sdc_scores, dtype=float)
    if scores.ndim != 1 or not len(scores):
        raise ValueError("Calibration requires injected Non-SDC scores")
    if not np.isfinite(scores).all():
        raise ValueError("Calibration probabilities must be finite")

    allowed = int(np.floor(target_fpr * len(scores)))
    if allowed <= 0:
        threshold = float(scores.max())
    elif allowed >= len(scores):
        threshold = float("-inf")
    else:
        descending = np.sort(scores)[::-1]
        threshold = float(descending[allowed])
    predicted = scores > threshold
    return {
        "threshold": threshold,
        "target_fpr": float(target_fpr),
        "achieved_fpr": float(predicted.mean()),
        "negative_samples": int(len(scores)),
        "allowed_false_positives": allowed,
        "observed_false_positives": int(predicted.sum()),
        "comparison": "positive_probability > threshold",
    }


def calibrate_threshold_max_f1(scores: Any, targets: Any) -> dict[str, Any]:
    """Select the strict-greater-than threshold maximizing calibration F1."""

    values = np.asarray(scores, dtype=float)
    truth = np.asarray(targets, dtype=int)
    if values.ndim != 1 or not len(values):
        raise ValueError("Calibration scores must be a non-empty vector")
    if truth.ndim != 1 or len(truth) != len(values):
        raise ValueError("Calibration targets must match calibration scores")
    if np.isnan(values).any():
        raise ValueError("Calibration scores contain NaN")
    if not set(np.unique(truth)).issubset({0, 1}):
        raise ValueError("Calibration targets only support 0/1")

    positive_samples = int(truth.sum())
    negative_samples = int(len(truth) - positive_samples)
    if positive_samples == 0 or negative_samples == 0:
        raise ValueError("Calibration requires both target classes")

    forced = np.isposinf(values)
    true_positive = int(np.sum(forced & (truth == 1)))
    false_positive = int(np.sum(forced & (truth == 0)))
    best: dict[str, Any] | None = None

    def consider(threshold: float) -> None:
        nonlocal best
        predicted_positive = true_positive + false_positive
        precision = (
            float(true_positive / predicted_positive)
            if predicted_positive
            else 0.0
        )
        recall = float(true_positive / positive_samples)
        f1 = (
            0.0
            if precision + recall == 0.0
            else float(2.0 * precision * recall / (precision + recall))
        )
        candidate = {
            "strategy": "maximize_f1",
            "threshold": float(threshold),
            "calibration_precision": precision,
            "calibration_recall": recall,
            "calibration_f1": f1,
            "calibration_fpr": float(false_positive / negative_samples),
            "samples": int(len(values)),
            "positive_samples": positive_samples,
            "negative_samples": negative_samples,
            "predicted_positive": predicted_positive,
            "tie_breaker": "highest_threshold",
            "comparison": "score > threshold; positive infinity always detected",
        }
        if best is None or f1 > best["calibration_f1"]:
            best = candidate

    consider(float("inf"))
    finite_mask = np.isfinite(values)
    finite_values = values[finite_mask]
    finite_targets = truth[finite_mask]
    order = np.argsort(finite_values)[::-1]
    finite_values = finite_values[order]
    finite_targets = finite_targets[order]
    index = 0
    while index < len(finite_values):
        value = finite_values[index]
        consider(float(value))
        group_end = index + 1
        while (
            group_end < len(finite_values)
            and finite_values[group_end] == value
        ):
            group_end += 1
        group_targets = finite_targets[index:group_end]
        true_positive += int(np.sum(group_targets == 1))
        false_positive += int(np.sum(group_targets == 0))
        index = group_end
    consider(float("-inf"))
    assert best is not None
    return best


def run_calibrated_xgboost(
    fit_csv: str | Path,
    calibration_csv: str | Path,
    test_csv: str | Path,
    output_dir: str | Path,
    *,
    group_column: str = "semantic_group_id",
    config: XGBoostConfig | None = None,
) -> dict[str, Any]:
    """Train on Fit, calibrate one threshold, and evaluate untouched Test."""

    detector_config = config or XGBoostConfig()
    paths = {
        "fit": Path(fit_csv).resolve(),
        "calibration": Path(calibration_csv).resolve(),
        "test": Path(test_csv).resolve(),
    }
    destination = Path(output_dir).resolve()
    _clean_output_directory(destination)
    frames = {
        name: add_significant_sdc_target(pd.read_csv(path))
        for name, path in paths.items()
    }
    _validate_external_splits(frames, group_column=group_column)

    fit = frames["fit"]
    calibration = frames["calibration"]
    test = frames["test"]
    feature_columns = get_feature_columns(fit)
    for name, frame in frames.items():
        missing = sorted(set(feature_columns) - set(frame.columns))
        if missing:
            raise ValueError(f"{name} CSV is missing features: {missing}")

    grouped = split_by_group(
        fit,
        group_column=group_column,
        holdout_ratio=detector_config.test_ratio,
        random_state=detector_config.random_state,
    )
    model, training = train_binary_model(
        prepare_features(grouped.train, feature_columns),
        grouped.train["significant_sdc_target"].astype(int),
        prepare_features(grouped.holdout, feature_columns),
        grouped.holdout["significant_sdc_target"].astype(int),
        config=detector_config,
    )

    calibration_probability = model.predict_proba(
        prepare_features(calibration, feature_columns)
    )[:, 1]
    threshold = calibrate_threshold_max_f1(
        calibration_probability,
        calibration["significant_sdc_target"].astype(int),
    )
    selected_threshold = float(threshold["threshold"])

    test_nan_mask = all_feature_nan_mask(test, feature_columns)
    cohorts = {
        "test_full": test,
        "test_finite": test.loc[~test_nan_mask].copy(),
        "test_clean": test.loc[test["injected"].astype(int).eq(0)].copy(),
        "test_injected_non_sdc": test.loc[
            test["injected"].astype(int).eq(1)
            & test["is_sdc"].astype(int).eq(0)
        ].copy(),
        "test_slight_sdc": test.loc[
            test["injected"].astype(int).eq(1)
            & test["is_sdc"].astype(int).eq(1)
            & test["significant_sdc_target"].astype(int).eq(0)
        ].copy(),
        "test_significant_sdc": test.loc[
            test["significant_sdc_target"].astype(int).eq(1)
        ].copy(),
    }
    metrics = {
        name: _evaluate(
            model,
            frame,
            prepare_features(frame, feature_columns),
            frame["significant_sdc_target"].astype(int),
            destination,
            name,
            threshold=selected_threshold,
        )
        for name, frame in cohorts.items()
    }

    model_path = destination / "significant_sdc_detector.ubj"
    booster = model.get_booster()
    best_iteration = training.get("best_iteration")
    if best_iteration is not None:
        booster = booster[: int(best_iteration) + 1]
    booster.save_model(model_path)
    summary = {
        "task_type": "binary_significant_sdc",
        "protocol": "fit_calibration_final_test",
        "input_csvs": {name: str(path) for name, path in paths.items()},
        "group_column": group_column,
        "external_split_group_overlap": 0,
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "fit_internal_split": grouped.summary.to_dict(),
        "training": training,
        "threshold_calibration": threshold,
        "calibration_objective": "maximize significant_sdc_target F1",
        "calibration_negative_definition": "significant_sdc_target == 0",
        "target_definition": {
            "0": "non-significant execution",
            "1": "pred_answer != clean_answer and significance == 2",
        },
        "cohort_rows": {
            name: int(len(frame)) for name, frame in cohorts.items()
        },
        "test_all_feature_nan": int(test_nan_mask.sum()),
        "metrics": metrics,
        "model_path": str(model_path),
        "inference_tree_count": booster.num_boosted_rounds(),
        "feature_importance_top20": _feature_importance(
            booster,
            feature_columns,
            destination / "significant_sdc_feature_importance.csv",
        ),
    }
    _atomic_write_json(destination / "metrics_summary.json", summary)
    return summary


def run_detector_job(
    config_path: str | Path,
    job_name: str,
    *,
    repository_root: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    experiment = load_yaml(config_path)
    feature_job = load_feature_job(
        config_path,
        job_name,
        repository_root=root,
    )
    detector = _require_mapping(experiment.get("detector"), "detector")
    xgboost_config = _require_mapping(detector.get("xgboost"), "detector.xgboost")
    common = dict(_require_mapping(xgboost_config.get("common"), "xgboost.common"))
    by_model = _require_mapping(xgboost_config.get("by_model"), "xgboost.by_model")
    common.update(_require_mapping(by_model.get(feature_job.model), feature_job.model))

    calibration_config = _require_mapping(
        detector.get("calibration"),
        "detector.calibration",
    )
    if calibration_config.get("strategy") != "maximize_f1":
        raise ValueError("detector.calibration.strategy must be maximize_f1")
    destination = (
        Path(output_dir).resolve()
        if output_dir
        else feature_job.fit_output.parent.parent / "output"
    )
    summary = run_calibrated_xgboost(
        feature_job.fit_output,
        feature_job.calibration_output,
        feature_job.test_output,
        destination,
        group_column=feature_job.group_column,
        config=XGBoostConfig.from_mapping(common),
    )
    print(f"Saved metrics summary to: {destination / 'metrics_summary.json'}")
    return summary


def _evaluate(
    model: XGBClassifier,
    frame: pd.DataFrame,
    features: pd.DataFrame,
    target: Any,
    output_dir: Path,
    split_name: str,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    if len(frame):
        probability = model.predict_proba(features)
        scores = probability[:, 1]
        prediction = (
            np.isposinf(scores) | (scores > threshold)
        ).astype(int)
    else:
        probability = np.empty((0, len(CLASS_LABELS)), dtype=float)
        prediction = np.asarray([], dtype=int)
    prefix = output_dir / f"significant_sdc_binary_{split_name}"
    files = _save_predictions(frame, target, prediction, probability, prefix)
    metrics = binary_metrics(target, prediction, probability)
    metrics.update(
        {
            "prediction_files": files,
            "label_significance_counts": label_significance_counts(frame),
            "threshold": threshold,
        }
    )
    return metrics


def _validate_external_splits(
    frames: Mapping[str, pd.DataFrame],
    *,
    group_column: str,
) -> None:
    expected = {"fit", "calibration", "test"}
    if set(frames) != expected:
        raise ValueError(f"Expected external splits {sorted(expected)}")
    group_sets: dict[str, set[str]] = {}
    uid_sets: dict[str, set[str]] = {}
    for name, frame in frames.items():
        for column in (group_column, "sample_uid", "injected", "is_sdc"):
            if column not in frame.columns:
                raise ValueError(f"{name} CSV is missing column: {column}")
        if frame.empty:
            raise ValueError(f"{name} CSV must not be empty")
        group_sets[name] = set(frame[group_column].astype(str))
        uid_sets[name] = set(frame["sample_uid"].astype(str))
    for left, right in (
        ("fit", "calibration"),
        ("fit", "test"),
        ("calibration", "test"),
    ):
        if group_sets[left] & group_sets[right]:
            raise ValueError(
                f"{group_column} overlap between {left} and {right}"
            )
        if uid_sets[left] & uid_sets[right]:
            raise ValueError(
                f"sample_uid overlap between {left} and {right}"
            )


def _save_predictions(
    frame: pd.DataFrame,
    truth: Any,
    prediction: Any,
    probability: np.ndarray,
    prefix: Path,
) -> dict[str, Any]:
    result = frame.copy().reset_index(drop=False)
    result.rename(columns={"index": "raw_row_index"}, inplace=True)
    result["y_true"] = np.asarray(truth, dtype=int)
    result["y_pred"] = np.asarray(prediction, dtype=int)
    result["is_correct"] = (result["y_true"] == result["y_pred"]).astype(int)
    for label in CLASS_LABELS:
        result[f"prob_class_{label}"] = probability[:, label]
    result["y_prob_max"] = (
        np.max(probability, axis=1) if len(probability) else np.asarray([], dtype=float)
    )
    wrong = result.loc[result["is_correct"] == 0].sort_values(
        "y_prob_max",
        ascending=False,
    )
    predictions_path = prefix.with_name(f"{prefix.name}_predictions.csv")
    wrong_path = prefix.with_name(f"{prefix.name}_wrong_predictions.csv")
    _atomic_write_csv(result, predictions_path)
    _atomic_write_csv(wrong, wrong_path)
    return {
        "predictions_csv": str(predictions_path),
        "wrong_predictions_csv": str(wrong_path),
        "wrong_count": int(len(wrong)),
    }


def _feature_importance(
    model: Any,
    feature_columns: list[str],
    output_path: Path,
) -> list[dict[str, Any]] | None:
    if hasattr(model, "get_score"):
        scores = model.get_score(importance_type="gain")
        importances = [float(scores.get(column, 0.0)) for column in feature_columns]
    else:
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            return None
    frame = pd.DataFrame(
        {"feature": feature_columns, "importance": importances}
    ).sort_values("importance", ascending=False)
    _atomic_write_csv(frame, output_path)
    return frame.head(20).to_dict(orient="records")


def _clean_output_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for candidate in path.rglob("*"):
        if candidate.is_file() and candidate.suffix in {
            ".csv",
            ".json",
            ".png",
            ".ubj",
        }:
            candidate.unlink()


def _atomic_write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(_json_safe(data), stream, ensure_ascii=False, indent=2)
    temporary.replace(path)


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    return value


def _optional_model_value(model: Any, name: str, converter: Any) -> Any:
    try:
        return converter(getattr(model, name))
    except (AttributeError, TypeError, ValueError):
        return None


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a non-empty mapping")
    return value
