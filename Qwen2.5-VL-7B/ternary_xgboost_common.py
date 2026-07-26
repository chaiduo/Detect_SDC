import json
import os

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
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier


RANDOM_STATE = 42
NUM_CLASSES = 2
CLASS_LABELS = [0, 1]
CLASS_NAMES = {
    0: "non_significant_sdc",
    1: "significant_sdc",
}
LEAKAGE_PRONE_META_COLS = ["total_steps", "last_k_steps", "num_steps_used"]
DROP_ALL_FEATURE_NAN = False


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def make_json_safe(value):
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [make_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [make_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return make_json_safe(value.tolist())
    return value


def atomic_write_json(path, data):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(data), f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def atomic_write_csv(df, path):
    tmp_path = f"{path}.tmp"
    df.to_csv(tmp_path, index=False, encoding="utf-8-sig")
    os.replace(tmp_path, path)


def clean_output_dir(output_dir):
    ensure_dir(output_dir)
    removable_suffixes = {".csv", ".json", ".png"}
    for root, _, files in os.walk(output_dir):
        for name in files:
            path = os.path.join(root, name)
            if os.path.splitext(name)[1] in removable_suffixes:
                os.remove(path)


def target_counts(y):
    y_arr = np.asarray(y, dtype=int)
    total = int(len(y_arr))
    counts = {str(cls): int((y_arr == cls).sum()) for cls in CLASS_LABELS}
    rates = {
        str(cls): float(counts[str(cls)] / total) if total else None
        for cls in CLASS_LABELS
    }
    return {
        "total": total,
        "counts": counts,
        "rates": rates,
    }


def label_significance_counts(df):
    if "label" not in df.columns or "significance" not in df.columns:
        return {}
    counts = (
        df[["label", "significance"]]
        .value_counts(dropna=False)
        .sort_index()
        .to_dict()
    )
    return {f"label={k[0]}, significance={k[1]}": int(v) for k, v in counts.items()}


def add_significant_sdc_target(df):
    if "label" not in df.columns:
        raise ValueError("CSV missing label column")
    if "significance" not in df.columns:
        raise ValueError("CSV missing significance column")

    label = pd.to_numeric(df["label"], errors="coerce")
    if label.isna().any():
        raise ValueError("label contains non-numeric values")

    label = label.astype(int)
    values = set(label.unique().tolist())
    if not values.issubset({0, 1, 2}):
        raise ValueError(f"label only supports 0/1/2, got: {sorted(values)}")

    binary = (label == 2).astype(int)
    bad_values = sorted(set(binary.unique().tolist()) - set(CLASS_LABELS))
    if bad_values:
        raise ValueError(f"binary label only supports 0/1, got: {bad_values}")

    df = df.copy()
    df["significant_sdc_target"] = binary
    return df


def get_feature_columns(df):
    excluded = {
        "orig_id",
        "sample_uid",
        "label",
        "significance",
        "ternary_target",
        "binary_target",
        "sdc_target",
        "significant_sdc_target",
        *LEAKAGE_PRONE_META_COLS,
    }
    return [c for c in df.columns if c not in excluded]


def prepare_features(df, feature_cols):
    x = df[feature_cols].copy()
    for col in feature_cols:
        x[col] = pd.to_numeric(x[col], errors="coerce")

    float32_max = np.finfo(np.float32).max
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.mask(x.abs() > float32_max, np.nan)
    return x


def all_feature_nan_mask(df, feature_cols):
    return prepare_features(df, feature_cols).isna().all(axis=1)


def split_train_test(df, group_col):
    if group_col not in df.columns:
        raise ValueError(f"CSV missing group column: {group_col}")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(df, groups=df[group_col]))
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    overlap = set(train_df[group_col].unique()).intersection(set(test_df[group_col].unique()))
    if overlap:
        raise AssertionError(f"train/test {group_col} overlap: {len(overlap)}")

    return train_df, test_df


def class_weight_vector(y):
    counts = target_counts(y)["counts"]
    total = sum(counts.values())
    weights = {}
    for cls in CLASS_LABELS:
        count = counts[str(cls)]
        weights[cls] = float(total / (NUM_CLASSES * count)) if count else 0.0
    return weights


def sample_weights(y):
    weights = class_weight_vector(y)
    y_arr = np.asarray(y, dtype=int)
    return np.asarray([weights[int(v)] for v in y_arr], dtype=float)


def build_xgb_binary_classifier():
    return XGBClassifier(
        n_estimators=10000,
        learning_rate=0.02,
        max_depth=6,
        min_child_weight=1,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric=["logloss", "error"],
        early_stopping_rounds=500,
        tree_method="hist",
        device="cpu",
        random_state=RANDOM_STATE,
        n_jobs=8,
    )


def train_binary_model(X_train, y_train, X_test, y_test):
    train_counts = target_counts(y_train)
    test_counts = target_counts(y_test)
    missing = [cls for cls in CLASS_LABELS if train_counts["counts"][str(cls)] == 0]
    if missing:
        raise ValueError(f"training set missing classes: {missing}")

    weights = sample_weights(y_train)
    model = build_xgb_binary_classifier()

    print("=" * 60)
    print("Start training significant_sdc_binary")
    print(json.dumps({
        "class_names": CLASS_NAMES,
        "train_distribution": train_counts,
        "test_distribution": test_counts,
        "class_weights": class_weight_vector(y_train),
    }, ensure_ascii=False, indent=2))
    print("=" * 60)

    model.fit(
        X_train,
        y_train,
        sample_weight=weights,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=200,
    )

    return model, {
        "train_distribution": train_counts,
        "test_distribution": test_counts,
        "class_weights": class_weight_vector(y_train),
        "best_iteration": int(model.best_iteration) if hasattr(model, "best_iteration") else None,
        "best_score": float(model.best_score) if hasattr(model, "best_score") else None,
    }


def binary_metrics(y_true, y_pred, y_prob):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)
    report = classification_report(
        y_true,
        y_pred,
        labels=CLASS_LABELS,
        target_names=[CLASS_NAMES[i] for i in CLASS_LABELS],
        output_dict=True,
        zero_division=0,
    )

    per_class = {}
    for cls in CLASS_LABELS:
        cls_name = CLASS_NAMES[cls]
        true_mask = y_true == cls
        pred_mask = y_pred == cls
        tp = int((true_mask & pred_mask).sum())
        fp = int((~true_mask & pred_mask).sum())
        fn = int((true_mask & ~pred_mask).sum())
        tn = int((~true_mask & ~pred_mask).sum())
        support = int(true_mask.sum())
        pred_total = int(pred_mask.sum())
        per_class[str(cls)] = {
            "name": cls_name,
            "support": support,
            "pred_total": pred_total,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": float(tp / pred_total) if pred_total else 0.0,
            "recall": float(tp / support) if support else 0.0,
            "f1": float(report[cls_name]["f1-score"]),
            "prob_mean": float(np.mean(y_prob[:, cls])) if len(y_prob) else None,
        }

    return {
        "confusion_matrix": cm.tolist(),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=CLASS_LABELS, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=CLASS_LABELS, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=CLASS_LABELS, average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(y_true, y_pred, labels=CLASS_LABELS, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, labels=CLASS_LABELS, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=CLASS_LABELS, average="weighted", zero_division=0)),
        "per_class": per_class,
        "classification_report": report,
        "target_class_1": per_class["1"],
        "target_significant_sdc": per_class["1"],
    }


def save_prediction_files(df_eval, y_true, y_pred, y_prob, output_prefix):
    result = df_eval.copy().reset_index(drop=False)
    result.rename(columns={"index": "raw_row_index"}, inplace=True)
    result["y_true"] = np.asarray(y_true, dtype=int)
    result["y_pred"] = np.asarray(y_pred, dtype=int)
    result["is_correct"] = (result["y_true"] == result["y_pred"]).astype(int)
    for cls in CLASS_LABELS:
        result[f"prob_class_{cls}"] = y_prob[:, cls]
    result["y_prob_max"] = np.max(y_prob, axis=1)

    wrong = result[result["is_correct"] == 0].copy()
    wrong = wrong.sort_values("y_prob_max", ascending=False)

    all_path = f"{output_prefix}_predictions.csv"
    wrong_path = f"{output_prefix}_wrong_predictions.csv"
    atomic_write_csv(result, all_path)
    atomic_write_csv(wrong, wrong_path)
    return {
        "predictions_csv": all_path,
        "wrong_predictions_csv": wrong_path,
        "wrong_count": int(len(wrong)),
    }


def evaluate_model(model, df_eval, X_eval, y_eval, output_dir, split_name):
    y_prob = model.predict_proba(X_eval)
    y_pred = np.argmax(y_prob, axis=1).astype(int)
    prefix = os.path.join(output_dir, f"significant_sdc_binary_{split_name}")
    prediction_files = save_prediction_files(df_eval, y_eval, y_pred, y_prob, prefix)

    metrics = binary_metrics(y_eval, y_pred, y_prob)
    metrics.update({
        "prediction_files": prediction_files,
        "label_significance_counts": label_significance_counts(df_eval),
    })
    return metrics


def feature_importance(model, feature_cols, output_path):
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return None

    df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances,
    }).sort_values("importance", ascending=False)
    atomic_write_csv(df, output_path)
    return df.head(20).to_dict(orient="records")


def run_ternary_xgboost(
    train_csv,
    valid_csv,
    group_col="orig_id",
    drop_all_feature_nan=None,
    output_subdir=None,
    clean_output=True,
):
    script_dir = os.path.dirname(os.path.abspath(train_csv))
    dataset_dir = os.path.dirname(script_dir)
    base_output_dir = os.path.join(dataset_dir, "output")
    output_dir = os.path.join(base_output_dir, output_subdir) if output_subdir else base_output_dir
    if clean_output:
        clean_output_dir(output_dir)
    else:
        ensure_dir(output_dir)

    if drop_all_feature_nan:
        print("[Warning] drop_all_feature_nan is deprecated for training and will be ignored. "
              "All-NaN samples stay in training; use valid_non_all_nan_metrics for the non-NaN eval slice.")

    train_df_full = add_significant_sdc_target(pd.read_csv(train_csv))
    valid_df_full = add_significant_sdc_target(pd.read_csv(valid_csv))
    feature_cols = get_feature_columns(train_df_full)

    train_nan_mask = all_feature_nan_mask(train_df_full, feature_cols)
    valid_nan_mask = all_feature_nan_mask(valid_df_full, feature_cols)
    filter_stats = {
        "training_policy": "keep_all_feature_nan",
        "drop_all_feature_nan_for_training": False,
        "train_all_feature_nan": int(train_nan_mask.sum()),
        "valid_all_feature_nan": int(valid_nan_mask.sum()),
        "train_all_feature_nan_label_significance_counts": label_significance_counts(train_df_full.loc[train_nan_mask]),
        "valid_all_feature_nan_label_significance_counts": label_significance_counts(valid_df_full.loc[valid_nan_mask]),
    }

    train_df_for_split = train_df_full.copy()
    train_df, test_df = split_train_test(train_df_for_split, group_col)
    valid_df_non_all_nan = valid_df_full.loc[~valid_nan_mask].copy()

    X_train = prepare_features(train_df, feature_cols)
    X_test = prepare_features(test_df, feature_cols)
    X_valid_full = prepare_features(valid_df_full, feature_cols)
    X_valid_non_all_nan = prepare_features(valid_df_non_all_nan, feature_cols)

    y_train = train_df["significant_sdc_target"].astype(int)
    y_test = test_df["significant_sdc_target"].astype(int)
    y_valid_full = valid_df_full["significant_sdc_target"].astype(int)
    y_valid_non_all_nan = valid_df_non_all_nan["significant_sdc_target"].astype(int)

    model, train_info = train_binary_model(X_train, y_train, X_test, y_test)

    summary = {
        "train_csv": train_csv,
        "valid_csv": valid_csv,
        "group_col": group_col,
        "output_subdir": output_subdir,
        "task_type": "binary_significant_sdc",
        "class_names": CLASS_NAMES,
        "target_definition": {
            "0": "non-significant samples: original label in {0, 1}",
            "1": "significant_sdc: original label == 2, i.e. pred_answer != clean_answer and significance == 2",
        },
        "feature_count": len(feature_cols),
        "feature_columns": feature_cols,
        "all_feature_nan_filter_stats": filter_stats,
        "train_full_label_significance_counts": label_significance_counts(train_df_full),
        "valid_full_label_significance_counts": label_significance_counts(valid_df_full),
        "train_rows_used": int(len(train_df_for_split)),
        "test_rows": int(len(test_df)),
        "valid_full_rows": int(len(valid_df_full)),
        "valid_non_all_nan_rows": int(len(valid_df_non_all_nan)),
        "train_info": train_info,
        "test_metrics": evaluate_model(model, test_df, X_test, y_test, output_dir, "test"),
        "valid_full_metrics": evaluate_model(model, valid_df_full, X_valid_full, y_valid_full, output_dir, "valid_full"),
        "valid_non_all_nan_metrics": evaluate_model(
            model,
            valid_df_non_all_nan,
            X_valid_non_all_nan,
            y_valid_non_all_nan,
            output_dir,
            "valid_non_all_nan",
        ),
        "feature_importance_top20": feature_importance(
            model,
            feature_cols,
            os.path.join(output_dir, "significant_sdc_binary_feature_importance.csv"),
        ),
    }

    summary_path = os.path.join(output_dir, "metrics_summary.json")
    atomic_write_json(summary_path, summary)
    print(f"Saved metrics summary to: {summary_path}")
    return summary


def run_ternary_xgboost_compare_nan_modes(
    train_csv,
    valid_csv,
    group_col="orig_id",
):
    script_dir = os.path.dirname(os.path.abspath(train_csv))
    dataset_dir = os.path.dirname(script_dir)
    output_dir = os.path.join(dataset_dir, "output")
    clean_output_dir(output_dir)

    print("=" * 60)
    print("Run NaN policy: train_with_all_nan")
    print("Evaluation slices: valid_full_metrics, valid_non_all_nan_metrics")
    print("=" * 60)
    summary = run_ternary_xgboost(
        train_csv=train_csv,
        valid_csv=valid_csv,
        group_col=group_col,
        drop_all_feature_nan=False,
        output_subdir="train_with_nan",
        clean_output=False,
    )

    combined_summary = {
        "train_csv": train_csv,
        "valid_csv": valid_csv,
        "group_col": group_col,
        "training_nan_policy": "keep_all_feature_nan",
        "evaluation_splits": [
            "valid_full_metrics",
            "valid_non_all_nan_metrics",
        ],
        "model_summary": summary,
    }
    summary_path = os.path.join(output_dir, "metrics_summary.json")
    atomic_write_json(summary_path, combined_summary)
    print(f"Saved combined metrics summary to: {summary_path}")
    return combined_summary
