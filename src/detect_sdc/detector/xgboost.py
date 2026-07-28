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
    "sample_uid",
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


def run_xgboost(
    train_csv: str | Path,
    valid_csv: str | Path,
    output_dir: str | Path,
    *,
    group_column: str = "orig_id",
    config: XGBoostConfig | None = None,
    clean_output: bool = True,
) -> dict[str, Any]:
    detector_config = config or XGBoostConfig()
    train_path = Path(train_csv).resolve()
    valid_path = Path(valid_csv).resolve()
    destination = Path(output_dir).resolve()
    if clean_output:
        _clean_output_directory(destination)
    else:
        destination.mkdir(parents=True, exist_ok=True)

    train_full = add_significant_sdc_target(pd.read_csv(train_path))
    valid_full = add_significant_sdc_target(pd.read_csv(valid_path))
    feature_columns = get_feature_columns(train_full)
    missing_valid_features = sorted(set(feature_columns) - set(valid_full.columns))
    if missing_valid_features:
        raise ValueError(f"Validation CSV is missing features: {missing_valid_features}")

    train_nan_mask = all_feature_nan_mask(train_full, feature_columns)
    valid_nan_mask = all_feature_nan_mask(valid_full, feature_columns)
    grouped = split_by_group(
        train_full,
        group_column=group_column,
        holdout_ratio=detector_config.test_ratio,
        random_state=detector_config.random_state,
    )
    train = grouped.train
    test = grouped.holdout
    valid_non_all_nan = valid_full.loc[~valid_nan_mask].copy()

    x_train = prepare_features(train, feature_columns)
    x_test = prepare_features(test, feature_columns)
    y_train = train["significant_sdc_target"].astype(int)
    y_test = test["significant_sdc_target"].astype(int)
    model, training = train_binary_model(
        x_train,
        y_train,
        x_test,
        y_test,
        config=detector_config,
    )

    summary = {
        "train_csv": str(train_path),
        "valid_csv": str(valid_path),
        "group_col": group_column,
        "task_type": "binary_significant_sdc",
        "class_names": CLASS_NAMES,
        "target_definition": {
            "0": "pred_answer == clean_answer or significance in {0, 1}",
            "1": "pred_answer != clean_answer and significance == 2",
        },
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "all_feature_nan_filter_stats": {
            "training_policy": "keep_all_feature_nan",
            "drop_all_feature_nan_for_training": False,
            "train_all_feature_nan": int(train_nan_mask.sum()),
            "valid_all_feature_nan": int(valid_nan_mask.sum()),
            "train_all_feature_nan_label_significance_counts": (
                label_significance_counts(train_full.loc[train_nan_mask])
            ),
            "valid_all_feature_nan_label_significance_counts": (
                label_significance_counts(valid_full.loc[valid_nan_mask])
            ),
        },
        "group_split": grouped.summary.to_dict(),
        "train_full_label_significance_counts": label_significance_counts(train_full),
        "valid_full_label_significance_counts": label_significance_counts(valid_full),
        "train_rows_used": int(len(train_full)),
        "test_rows": int(len(test)),
        "valid_full_rows": int(len(valid_full)),
        "valid_non_all_nan_rows": int(len(valid_non_all_nan)),
        "train_info": training,
        "test_metrics": _evaluate(
            model,
            test,
            x_test,
            y_test,
            destination,
            "test",
        ),
        "valid_full_metrics": _evaluate(
            model,
            valid_full,
            prepare_features(valid_full, feature_columns),
            valid_full["significant_sdc_target"].astype(int),
            destination,
            "valid_full",
        ),
        "valid_non_all_nan_metrics": _evaluate(
            model,
            valid_non_all_nan,
            prepare_features(valid_non_all_nan, feature_columns),
            valid_non_all_nan["significant_sdc_target"].astype(int),
            destination,
            "valid_non_all_nan",
        ),
        "feature_importance_top20": _feature_importance(
            model,
            feature_columns,
            destination / "significant_sdc_binary_feature_importance.csv",
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

    destination = (
        Path(output_dir).resolve()
        if output_dir
        else feature_job.train_output.parent.parent / "output"
    )
    _clean_output_directory(destination)
    model_output = destination / "train_with_nan"
    summary = run_xgboost(
        feature_job.train_output,
        feature_job.valid_output,
        model_output,
        group_column=feature_job.group_column,
        config=XGBoostConfig.from_mapping(common),
        clean_output=False,
    )
    combined = {
        "train_csv": str(feature_job.train_output),
        "valid_csv": str(feature_job.valid_output),
        "group_col": feature_job.group_column,
        "training_nan_policy": "keep_all_feature_nan",
        "evaluation_splits": [
            "valid_full_metrics",
            "valid_non_all_nan_metrics",
        ],
        "model_summary": summary,
    }
    _atomic_write_json(destination / "metrics_summary.json", combined)
    print(f"Saved metrics summary to: {destination / 'metrics_summary.json'}")
    return combined


def run_legacy_xgboost(
    train_csv: str | Path,
    valid_csv: str | Path,
    *,
    group_col: str = "orig_id",
    learning_rate: float = 0.01,
    output_subdir: str | None = None,
    clean_output: bool = True,
    **_: Any,
) -> dict[str, Any]:
    train_path = Path(train_csv).resolve()
    base_output = train_path.parent.parent / "output"
    destination = base_output / output_subdir if output_subdir else base_output
    return run_xgboost(
        train_path,
        valid_csv,
        destination,
        group_column=group_col,
        config=XGBoostConfig(learning_rate=learning_rate),
        clean_output=clean_output,
    )


def run_legacy_compare(
    train_csv: str | Path,
    valid_csv: str | Path,
    *,
    group_col: str = "orig_id",
    learning_rate: float = 0.01,
) -> dict[str, Any]:
    train_path = Path(train_csv).resolve()
    output = train_path.parent.parent / "output"
    _clean_output_directory(output)
    summary = run_legacy_xgboost(
        train_path,
        valid_csv,
        group_col=group_col,
        learning_rate=learning_rate,
        output_subdir="train_with_nan",
        clean_output=False,
    )
    combined = {
        "train_csv": str(train_path),
        "valid_csv": str(Path(valid_csv).resolve()),
        "group_col": group_col,
        "training_nan_policy": "keep_all_feature_nan",
        "evaluation_splits": [
            "valid_full_metrics",
            "valid_non_all_nan_metrics",
        ],
        "model_summary": summary,
    }
    _atomic_write_json(output / "metrics_summary.json", combined)
    return combined


def _evaluate(
    model: XGBClassifier,
    frame: pd.DataFrame,
    features: pd.DataFrame,
    target: Any,
    output_dir: Path,
    split_name: str,
) -> dict[str, Any]:
    if len(frame):
        probability = model.predict_proba(features)
        prediction = np.argmax(probability, axis=1).astype(int)
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
        }
    )
    return metrics


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
    model: XGBClassifier,
    feature_columns: list[str],
    output_path: Path,
) -> list[dict[str, Any]] | None:
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
        if candidate.is_file() and candidate.suffix in {".csv", ".json", ".png"}:
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
