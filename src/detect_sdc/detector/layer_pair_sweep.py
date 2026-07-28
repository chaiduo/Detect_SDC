import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBClassifier

from ..splitting import split_by_group
from .xgboost import add_significant_sdc_target

RANDOM_STATE = 42
TARGET_SIGNIFICANCE = 2
TARGET_NON_SDC_FPR = 0.02
DROP_ALL_FEATURE_NAN = False
LEAKAGE_PRONE_META_COLS = ["total_steps", "last_k_steps", "num_steps_used"]
THRESHOLDS = np.round(np.arange(0.0, 0.951, 0.05), 6)


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


def add_targets(df):
    return add_significant_sdc_target(df)


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


def get_feature_columns(df):
    excluded = {
        "orig_id",
        "sample_uid",
        "label",
        "significance",
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
    split = split_by_group(
        df,
        group_column=group_col,
        holdout_ratio=0.15,
        random_state=RANDOM_STATE,
    )
    return split.train, split.holdout


def build_xgb_binary_classifier(scale_pos_weight):
    return XGBClassifier(
        n_estimators=10000,
        learning_rate=0.01,
        max_depth=6,
        min_child_weight=1,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric=["logloss", "auc"],
        early_stopping_rounds=500,
        tree_method="hist",
        device="cpu",
        random_state=RANDOM_STATE,
        n_jobs=8,
        scale_pos_weight=scale_pos_weight,
    )


def target_counts(y):
    y_arr = np.asarray(y, dtype=int)
    neg = int((y_arr == 0).sum())
    pos = int((y_arr == 1).sum())
    return {
        "negative": neg,
        "positive": pos,
        "positive_rate": float(pos / len(y_arr)) if len(y_arr) else None,
    }


def train_binary_model(task_name, X_train, y_train, X_test, y_test):
    counts = target_counts(y_train)
    if counts["positive"] == 0:
        raise ValueError(f"{task_name}: 训练集中没有正样本")
    if counts["negative"] == 0:
        raise ValueError(f"{task_name}: 训练集中没有负样本")

    scale_pos_weight = counts["negative"] / counts["positive"]
    model = build_xgb_binary_classifier(scale_pos_weight=scale_pos_weight)

    print("=" * 60)
    print(f"Start training {task_name}")
    print(json.dumps({
        "train_distribution": counts,
        "test_distribution": target_counts(y_test),
        "scale_pos_weight": scale_pos_weight,
    }, ensure_ascii=False, indent=2))
    print("=" * 60)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=200,
    )

    return model, {
        "train_distribution": counts,
        "test_distribution": target_counts(y_test),
        "scale_pos_weight": float(scale_pos_weight),
        "best_iteration": int(model.best_iteration) if hasattr(model, "best_iteration") else None,
        "best_score": float(model.best_score) if hasattr(model, "best_score") else None,
    }


def threshold_metrics(y_true, y_score, thresholds=THRESHOLDS):
    rows = []
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    total = int(len(y_true))
    pos_total = int((y_true == 1).sum())
    neg_total = int((y_true == 0).sum())

    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        pred_pos = int((y_pred == 1).sum())
        rows.append({
            "threshold": float(threshold),
            "total": total,
            "positive_total": pos_total,
            "negative_total": neg_total,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "recall": float(tp / pos_total) if pos_total else None,
            "precision": float(tp / pred_pos) if pred_pos else None,
            "fpr": float(fp / neg_total) if neg_total else None,
            "pred_positive_total": pred_pos,
            "pred_positive_ratio": float(pred_pos / total) if total else None,
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        })

    return rows


def choose_operating_point(rows, max_fpr=TARGET_NON_SDC_FPR):
    if not rows:
        return None

    df = pd.DataFrame(rows)
    feasible = df[df["fpr"] <= max_fpr].copy()
    selected_under_budget = True
    if feasible.empty:
        feasible = df.copy()
        selected_under_budget = False

    feasible = feasible.sort_values(
        by=["recall", "precision", "fpr", "threshold"],
        ascending=[False, False, True, False],
    )
    chosen = feasible.iloc[0].to_dict()
    chosen["selected_under_fpr_budget"] = selected_under_budget
    chosen["fpr_budget"] = float(max_fpr)
    return chosen


def choose_best_f1(rows):
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df = df.sort_values(by=["f1", "recall", "precision", "threshold"], ascending=[False, False, False, False])
    return df.iloc[0].to_dict()


def choose_best_recall(rows):
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df = df.sort_values(
        by=["recall", "precision", "f1", "fpr", "threshold"],
        ascending=[False, False, False, True, False],
    )
    chosen = df.iloc[0].to_dict()
    chosen["selection_mode"] = "max_recall"
    return chosen


def choose_threshold_point(rows, threshold_selection_mode):
    if threshold_selection_mode == "low_fpr_recall":
        return choose_operating_point(rows)
    if threshold_selection_mode == "max_recall":
        return choose_best_recall(rows)
    if threshold_selection_mode == "best_f1":
        chosen = choose_best_f1(rows)
        if chosen is not None:
            chosen["selection_mode"] = "best_f1"
        return chosen
    raise ValueError(f"Unsupported threshold_selection_mode: {threshold_selection_mode}")


def binary_metrics(y_true, y_score, threshold):
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    roc_auc = None
    if len(np.unique(y_true)) > 1:
        roc_auc = float(roc_auc_score(y_true, y_score))

    return {
        "threshold": float(threshold),
        "confusion_matrix": cm.tolist(),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
        "positive_total": int((y_true == 1).sum()),
        "negative_total": int((y_true == 0).sum()),
        "pred_positive_total": int((y_pred == 1).sum()),
    }


def save_prediction_files(df_eval, y_true, y_score, threshold, output_prefix):
    result = df_eval.copy().reset_index(drop=False)
    result.rename(columns={"index": "raw_row_index"}, inplace=True)
    result["y_true"] = np.asarray(y_true, dtype=int)
    result["y_score"] = np.asarray(y_score, dtype=float)
    result["y_pred"] = (result["y_score"] >= threshold).astype(int)
    result["is_correct"] = (result["y_true"] == result["y_pred"]).astype(int)

    wrong = result[result["is_correct"] == 0].copy()
    wrong["wrong_confidence"] = np.where(wrong["y_pred"] == 1, wrong["y_score"], 1.0 - wrong["y_score"])
    wrong = wrong.sort_values("wrong_confidence", ascending=False)

    all_path = f"{output_prefix}_predictions.csv"
    wrong_path = f"{output_prefix}_wrong_predictions.csv"
    atomic_write_csv(result, all_path)
    atomic_write_csv(wrong, wrong_path)
    return {
        "predictions_csv": all_path,
        "wrong_predictions_csv": wrong_path,
        "wrong_count": int(len(wrong)),
    }


def save_roc_points(y_true, y_score, output_prefix):
    if len(np.unique(y_true)) < 2:
        return None
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_df = pd.DataFrame({
        "fpr": fpr,
        "tpr": tpr,
        "threshold": thresholds,
    })
    path = f"{output_prefix}_roc_points.csv"
    atomic_write_csv(roc_df, path)
    return {
        "roc_points_csv": path,
        "roc_auc": float(auc(fpr, tpr)),
    }


def evaluate_model(task_name, model, df_eval, X_eval, y_eval, output_dir, split_name, threshold):
    y_score = model.predict_proba(X_eval)[:, 1]
    rows = threshold_metrics(y_eval, y_score)
    threshold_path = os.path.join(output_dir, f"{task_name}_{split_name}_threshold_metrics.csv")
    atomic_write_csv(pd.DataFrame(rows), threshold_path)

    prefix = os.path.join(output_dir, f"{task_name}_{split_name}")
    prediction_files = save_prediction_files(df_eval, y_eval, y_score, threshold, prefix)
    roc_info = save_roc_points(y_eval, y_score, prefix)

    metrics = binary_metrics(y_eval, y_score, threshold)
    metrics.update({
        "threshold_metrics_csv": threshold_path,
        "prediction_files": prediction_files,
        "roc_info": roc_info,
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


def run_binary_xgboost(
    train_csv,
    valid_csv,
    group_col="orig_id",
    threshold_selection_mode="low_fpr_recall",
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

    effective_drop_all_feature_nan = (
        DROP_ALL_FEATURE_NAN if drop_all_feature_nan is None else bool(drop_all_feature_nan)
    )

    train_df_full = add_targets(pd.read_csv(train_csv))
    valid_df_full = add_targets(pd.read_csv(valid_csv))
    feature_cols = get_feature_columns(train_df_full)

    train_nan_mask = all_feature_nan_mask(train_df_full, feature_cols)
    valid_nan_mask = all_feature_nan_mask(valid_df_full, feature_cols)
    filter_stats = {
        "drop_all_feature_nan": effective_drop_all_feature_nan,
        "train_removed": int(train_nan_mask.sum()),
        "valid_removed": int(valid_nan_mask.sum()),
        "train_removed_label_significance_counts": label_significance_counts(train_df_full.loc[train_nan_mask]),
        "valid_removed_label_significance_counts": label_significance_counts(valid_df_full.loc[valid_nan_mask]),
    }

    train_df_for_split = (
        train_df_full.loc[~train_nan_mask].copy()
        if effective_drop_all_feature_nan
        else train_df_full.copy()
    )
    train_df, test_df = split_train_test(train_df_for_split, group_col)
    valid_df_non_all_nan = valid_df_full.loc[~valid_nan_mask].copy()

    X_train = prepare_features(train_df, feature_cols)
    X_test = prepare_features(test_df, feature_cols)
    X_valid_full = prepare_features(valid_df_full, feature_cols)
    X_valid_non_all_nan = prepare_features(valid_df_non_all_nan, feature_cols)

    tasks = {
        "significant_sdc": {
            "target_col": "significant_sdc_target",
            "description": "predict significant SDC",
        },
    }

    summary = {
        "train_csv": train_csv,
        "valid_csv": valid_csv,
        "group_col": group_col,
        "threshold_selection_mode": threshold_selection_mode,
        "output_subdir": output_subdir,
        "target_significance": TARGET_SIGNIFICANCE,
        "target_non_sdc_fpr": TARGET_NON_SDC_FPR,
        "feature_count": len(feature_cols),
        "feature_columns": feature_cols,
        "all_feature_nan_filter_stats": filter_stats,
        "train_full_label_significance_counts": label_significance_counts(train_df_full),
        "valid_full_label_significance_counts": label_significance_counts(valid_df_full),
        "train_rows_after_filter": int(len(train_df_for_split)),
        "test_rows": int(len(test_df)),
        "valid_full_rows": int(len(valid_df_full)),
        "valid_non_all_nan_rows": int(len(valid_df_non_all_nan)),
        "tasks": {},
    }

    for task_name, task_cfg in tasks.items():
        target_col = task_cfg["target_col"]
        y_train = train_df[target_col].astype(int)
        y_test = test_df[target_col].astype(int)
        y_valid_full = valid_df_full[target_col].astype(int)
        y_valid_non_all_nan = valid_df_non_all_nan[target_col].astype(int)

        model, train_info = train_binary_model(task_name, X_train, y_train, X_test, y_test)
        test_scores = model.predict_proba(X_test)[:, 1]
        threshold_rows = threshold_metrics(y_test, test_scores)
        selected_point = choose_threshold_point(threshold_rows, threshold_selection_mode)
        operating_point = choose_operating_point(threshold_rows)
        max_recall_point = choose_best_recall(threshold_rows)
        best_f1_point = choose_best_f1(threshold_rows)
        selected_threshold = float(selected_point["threshold"]) if selected_point else 0.5

        print("=" * 60)
        print(f"{task_name} threshold selection mode: {threshold_selection_mode}")
        print(f"{task_name} selected threshold: {selected_threshold}")
        print(json.dumps({
            "selected_point": selected_point,
            "operating_point": operating_point,
            "max_recall_point": max_recall_point,
            "best_f1_point": best_f1_point,
        }, ensure_ascii=False, indent=2))
        print("=" * 60)

        task_summary = {
            "description": task_cfg["description"],
            "target_col": target_col,
            "train_info": train_info,
            "threshold_selection_mode": threshold_selection_mode,
            "selected_threshold_from_test": selected_threshold,
            "selected_point_from_test": selected_point,
            "operating_point_from_test": operating_point,
            "max_recall_point_from_test": max_recall_point,
            "best_f1_point_from_test": best_f1_point,
            "test_metrics": evaluate_model(
                task_name, model, test_df, X_test, y_test, output_dir, "test", selected_threshold
            ),
            "valid_full_metrics": evaluate_model(
                task_name, model, valid_df_full, X_valid_full, y_valid_full, output_dir, "valid_full", selected_threshold
            ),
            "valid_non_all_nan_metrics": evaluate_model(
                task_name,
                model,
                valid_df_non_all_nan,
                X_valid_non_all_nan,
                y_valid_non_all_nan,
                output_dir,
                "valid_non_all_nan",
                selected_threshold,
            ),
            "feature_importance_top20": feature_importance(
                model,
                feature_cols,
                os.path.join(output_dir, f"{task_name}_feature_importance.csv"),
            ),
        }
        summary["tasks"][task_name] = task_summary

    summary_path = os.path.join(output_dir, "metrics_summary.json")
    atomic_write_json(summary_path, summary)
    print(f"Saved metrics summary to: {summary_path}")
    return summary


def run_binary_xgboost_compare_nan_modes(
    train_csv,
    valid_csv,
    group_col="orig_id",
    threshold_selection_mode="low_fpr_recall",
):
    script_dir = os.path.dirname(os.path.abspath(train_csv))
    dataset_dir = os.path.dirname(script_dir)
    output_dir = os.path.join(dataset_dir, "output")
    clean_output_dir(output_dir)

    modes = {
        "keep_all_nan": False,
        "drop_all_feature_nan": True,
    }
    summaries = {}
    for mode_name, drop_all_feature_nan in modes.items():
        print("=" * 60)
        print(f"Run NaN mode: {mode_name}")
        print("=" * 60)
        summaries[mode_name] = run_binary_xgboost(
            train_csv=train_csv,
            valid_csv=valid_csv,
            group_col=group_col,
            threshold_selection_mode=threshold_selection_mode,
            drop_all_feature_nan=drop_all_feature_nan,
            output_subdir=mode_name,
            clean_output=False,
        )

    combined_summary = {
        "train_csv": train_csv,
        "valid_csv": valid_csv,
        "group_col": group_col,
        "threshold_selection_mode": threshold_selection_mode,
        "nan_modes": list(modes.keys()),
        "mode_summaries": summaries,
    }
    summary_path = os.path.join(output_dir, "metrics_summary.json")
    atomic_write_json(summary_path, combined_summary)
    print(f"Saved combined metrics summary to: {summary_path}")
    return combined_summary
