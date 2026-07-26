import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_curve,
    auc,
    classification_report
)

from xgboost import XGBClassifier


# =========================
# 配置区域
# =========================
TRAIN_CSV = "/data01/cd_workspace/Detect_SDC/llava-v1.5-7B/EarthVQA/train_data/llava-v1.5-7B_train_set.csv"
VALID_CSV = "/data01/cd_workspace/Detect_SDC/llava-v1.5-7B/EarthVQA/train_data/llava-v1.5-7B_valid_set.csv"
GROUP_COL = "sample_uid"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figure")
RANDOM_STATE = 42
TARGET_SIGNIFICANCE = 2
TARGET_NON_SDC_FPR = 0.02
POS_SIGNIFICANCE2_WEIGHT_MULTIPLIER = 3.0
HARD_NEG_SIGNIFICANCE2_WEIGHT_MULTIPLIER = 1.5
USE_TERNARY_SEVERITY_LABEL = True
TERNARY_SIGNIFICANT_CLASS_WEIGHT_MULTIPLIER = 3.0
LEAKAGE_PRONE_META_COLS = ["total_steps", "last_k_steps", "num_steps_used"]
DROP_ALL_FEATURE_NAN = True


# =========================
# 工具函数
# =========================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_predictions(
    df,
    y_true,
    y_pred,
    y_prob_all,
    task_type,
    extra_columns=None,
    save_all_path=os.path.join(OUTPUT_DIR, "predictions.csv"),
    save_wrong_path=os.path.join(OUTPUT_DIR, "wrong_predictions.csv")
):
    """
    保存预测结果，以及预测错误样本
    """
    result_df = df.copy().reset_index(drop=False)
    result_df.rename(columns={"index": "raw_row_index"}, inplace=True)

    result_df["y_true"] = np.array(y_true)
    result_df["y_pred"] = np.array(y_pred)
    result_df["is_correct"] = (result_df["y_true"] == result_df["y_pred"]).astype(int)

    if extra_columns:
        for col_name, col_value in extra_columns.items():
            result_df[col_name] = np.array(col_value)

    if y_prob_all.ndim == 2 and y_prob_all.shape[1] == 2:
        result_df["prob_class_0"] = y_prob_all[:, 0]
        result_df["prob_class_1"] = y_prob_all[:, 1]
        result_df["y_prob_positive"] = y_prob_all[:, 1]

        wrong_df = result_df[result_df["is_correct"] == 0].copy()
        if "final_decision_score" in wrong_df.columns:
            wrong_df["wrong_confidence"] = wrong_df["final_decision_score"]
        else:
            wrong_df["wrong_confidence"] = np.where(
                wrong_df["y_pred"] == 1,
                wrong_df["prob_class_1"],
                wrong_df["prob_class_0"]
            )
        wrong_df = wrong_df.sort_values("wrong_confidence", ascending=False)

    else:
        n_classes = y_prob_all.shape[1]
        for i in range(n_classes):
            result_df[f"prob_class_{i}"] = y_prob_all[:, i]

        result_df["y_prob_max"] = np.max(y_prob_all, axis=1)
        wrong_df = result_df[result_df["is_correct"] == 0].copy()
        wrong_df = wrong_df.sort_values("y_prob_max", ascending=False)

    save_all_dir = os.path.dirname(save_all_path)
    save_wrong_dir = os.path.dirname(save_wrong_path)
    if save_all_dir:
        ensure_dir(save_all_dir)
    if save_wrong_dir:
        ensure_dir(save_wrong_dir)

    result_df.to_csv(save_all_path, index=False, encoding="utf-8-sig")
    wrong_df.to_csv(save_wrong_path, index=False, encoding="utf-8-sig")

    print(f"Saved predictions to: {save_all_path}")
    print(f"Saved wrong predictions to: {save_wrong_path}")
    print(f"Total samples: {len(result_df)}")
    print(f"Wrong predictions: {len(wrong_df)}")
    print("=" * 60)

    return result_df, wrong_df


def find_best_threshold(y_true, y_prob, start=0.05, end=0.95, step=0.01):
    """
    仅适用于二分类
    """
    thresholds = np.arange(start, end + 1e-9, step)
    best_thr = 0.5
    best_f1 = -1.0

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thr = thr

    return best_thr, best_f1


def evaluate_significance_recall_by_threshold(
    df_eval,
    y_true,
    y_score,
    significance_value=2,
    start=0,
    end=0.95,
    step=0.05,
    save_path=None,
):
    """
    用于选择业务阈值：
    1. sdc_recall = 预测为正类的 label==1 样本数 / 全部 label==1 样本数
    2. significance2_sdc_recall = 预测为正类的 (label==1 且 significance==significance_value) 样本数 / 全部该类样本数
    3. non_sdc_fpr = 预测为正类的 label==0 样本数 / 全部 label==0 样本数
    """
    if "significance" not in df_eval.columns:
        print("[Warning] significance column not found, skip significance recall analysis.")
        return []

    significance = pd.to_numeric(df_eval["significance"], errors="coerce")
    y_true_arr = np.asarray(y_true)
    sdc_mask = y_true_arr == 1
    non_sdc_mask = y_true_arr == 0
    target_mask = sdc_mask & (significance == significance_value)

    sdc_total = int(sdc_mask.sum())
    non_sdc_total = int(non_sdc_mask.sum())
    target_total = int(target_mask.sum())
    total_samples = int(len(df_eval))
    thresholds = np.arange(start, end + 1e-9, step)

    rows = []
    for thr in thresholds:
        y_pred = (y_score >= thr).astype(int)
        pred_positive = y_pred == 1
        hit_sdc = int((sdc_mask & pred_positive).sum())
        hit_target = int((target_mask & pred_positive).sum())
        fp_non_sdc = int((non_sdc_mask & pred_positive).sum())
        pred_positive_total = int(pred_positive.sum())
        sdc_recall = hit_sdc / sdc_total if sdc_total > 0 else None
        target_recall = hit_target / target_total if target_total > 0 else None
        non_sdc_fpr = fp_non_sdc / non_sdc_total if non_sdc_total > 0 else None
        pred_positive_ratio = pred_positive_total / total_samples if total_samples > 0 else None
        target_alert_precision = hit_target / pred_positive_total if pred_positive_total > 0 else None

        rows.append({
            "threshold": float(round(thr, 6)),
            "sdc_total": sdc_total,
            "sdc_recalled": hit_sdc,
            "sdc_recall": sdc_recall,
            "target_significance_value": int(significance_value),
            "total_samples": total_samples,
            "target_sdc_total": target_total,
            "target_sdc_recalled": hit_target,
            "target_sdc_recall": target_recall,
            "non_sdc_total": non_sdc_total,
            "non_sdc_false_positive": fp_non_sdc,
            "non_sdc_fpr": non_sdc_fpr,
            "pred_positive_total": pred_positive_total,
            "pred_positive_ratio": pred_positive_ratio,
            "target_alert_precision": target_alert_precision,
        })

    result_df = pd.DataFrame(rows)
    print(f"SDC recall / significance == {significance_value} SDC recall / non-SDC FPR by threshold:")
    print(result_df.to_string(index=False))
    print("=" * 60)

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            ensure_dir(save_dir)
        result_df.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"Saved significance recall table to: {save_path}")
        print("=" * 60)

    return rows


def choose_threshold_by_target_recall(
    threshold_rows,
    max_non_sdc_fpr=TARGET_NON_SDC_FPR,
):
    if not threshold_rows:
        return None

    df = pd.DataFrame(threshold_rows)
    feasible_df = df[df["non_sdc_fpr"] <= max_non_sdc_fpr].copy()
    chosen_from_feasible = True
    if feasible_df.empty:
        feasible_df = df.copy()
        chosen_from_feasible = False

    feasible_df = feasible_df.sort_values(
        by=[
            "target_sdc_recall",
            "non_sdc_fpr",
            "target_alert_precision",
            "sdc_recall",
            "threshold",
        ],
        ascending=[False, True, False, False, False],
    )
    best_row = feasible_df.iloc[0].to_dict()
    best_row["selected_under_fpr_budget"] = chosen_from_feasible
    best_row["non_sdc_fpr_budget"] = float(max_non_sdc_fpr)

    print("Selected operating point:")
    print(json.dumps(best_row, ensure_ascii=False, indent=2))
    print("=" * 60)

    return best_row


def plot_significance_threshold_tradeoff(
    threshold_rows,
    save_path,
    title="SDC Recall / Target Recall / Non-SDC FPR Tradeoff",
):
    if not threshold_rows:
        print("[Warning] Empty threshold rows, skip significance tradeoff plot.")
        return

    df = pd.DataFrame(threshold_rows)
    if df.empty:
        print("[Warning] Empty threshold dataframe, skip significance tradeoff plot.")
        return

    thresholds = df["threshold"].to_numpy()
    sdc_recall = df["sdc_recall"].astype(float).to_numpy()
    target_sdc_recall = df["target_sdc_recall"].astype(float).to_numpy()
    non_sdc_fpr = df["non_sdc_fpr"].astype(float).to_numpy()

    save_dir = os.path.dirname(save_path)
    if save_dir:
        ensure_dir(save_dir)

    plt.figure(figsize=(8, 5))
    plt.plot(
        thresholds,
        sdc_recall,
        marker="o",
        linewidth=2,
        color="tab:blue",
        label="sdc_recall",
    )
    plt.plot(
        thresholds,
        target_sdc_recall,
        marker="s",
        linewidth=2,
        color="tab:orange",
        label="significance2_sdc_recall",
    )
    plt.plot(
        thresholds,
        non_sdc_fpr,
        marker="^",
        linewidth=2,
        color="tab:red",
        label="non_sdc_fpr",
    )
    plt.xlabel("Threshold")
    plt.ylabel("Rate")
    plt.xlim(0, 0.95)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="best")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"Saved significance tradeoff curve to: {save_path}")


def print_metrics_binary(y_true, y_pred, y_prob=None, title="Binary Metrics"):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(title)
    print(f"Accuracy:  {acc:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall:    {recall:.6f}")
    print(f"F1:        {f1:.6f}")

    auc_score = None
    if y_prob is not None and len(set(y_true)) > 1:
        auc_score = roc_auc_score(y_true, y_prob)
        print(f"ROC-AUC:   {auc_score:.6f}")

    print("=" * 60)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc_score
    }


def print_metrics_multiclass(y_true, y_pred, y_prob=None, title="Multiclass Metrics"):
    acc = accuracy_score(y_true, y_pred)

    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    precision_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(title)
    print(f"Accuracy:             {acc:.6f}")
    print(f"Precision (macro):    {precision_macro:.6f}")
    print(f"Recall (macro):       {recall_macro:.6f}")
    print(f"F1 (macro):           {f1_macro:.6f}")
    print(f"Precision (weighted): {precision_weighted:.6f}")
    print(f"Recall (weighted):    {recall_weighted:.6f}")
    print(f"F1 (weighted):        {f1_weighted:.6f}")

    auc_macro_ovr = None
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            auc_macro_ovr = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            print(f"ROC-AUC (macro, ovr): {auc_macro_ovr:.6f}")
        except Exception as e:
            print(f"ROC-AUC (macro, ovr) 计算失败: {e}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=6, zero_division=0))
    print("=" * 60)

    return {
        "accuracy": acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "roc_auc_macro_ovr": auc_macro_ovr
    }


def plot_roc_curve_binary(y_true, y_prob, save_path="roc_curve.png", title="ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"Saved ROC curve to: {save_path}")


def plot_roc_curve_multiclass(y_true, y_prob, n_classes, save_path="roc_curve_multiclass.png", title="Multiclass ROC Curve (OvR)"):
    from sklearn.preprocessing import label_binarize

    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    plt.figure(figsize=(8, 6))

    for i in range(n_classes):
        if len(np.unique(y_true_bin[:, i])) < 2:
            print(f"类别 {i} 在当前数据集中缺少正样本或负样本，跳过该类别 ROC。")
            continue

        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=2, label=f"Class {i} ROC (AUC={roc_auc:.4f})")

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"Saved multiclass ROC curve to: {save_path}")


def plot_confusion_matrix_fig(cm, save_path="confusion_matrix.png", class_names=None, title="Confusion Matrix"):
    n_classes = cm.shape[0]

    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]

    plt.figure(figsize=(6 + n_classes * 0.5, 5 + n_classes * 0.3))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(n_classes)
    plt.xticks(tick_marks, [f"Pred {x}" for x in class_names], rotation=45)
    plt.yticks(tick_marks, [f"True {x}" for x in class_names])

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=11
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"Saved confusion matrix to: {save_path}")


def plot_training_curves(evals_result, save_dir="."):
    train_key = "validation_0"
    val_key = "validation_1"

    if train_key not in evals_result or val_key not in evals_result:
        print("evals_result 中未找到 validation_0 / validation_1，跳过训练曲线绘制。")
        return

    if "mlogloss" in evals_result[train_key] and "mlogloss" in evals_result[val_key]:
        train_logloss = evals_result[train_key]["mlogloss"]
        val_logloss = evals_result[val_key]["mlogloss"]

        plt.figure(figsize=(8, 6))
        plt.plot(train_logloss, label="Train mlogloss", linewidth=2)
        plt.plot(val_logloss, label="Test mlogloss", linewidth=2)
        plt.xlabel("Boosting Round")
        plt.ylabel("mlogloss")
        plt.title("Training Curve - Multiclass Logloss")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        save_path = os.path.join(save_dir, "training_curve_mlogloss.png")
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"Saved training mlogloss curve to: {save_path}")

    if "logloss" in evals_result[train_key] and "logloss" in evals_result[val_key]:
        train_logloss = evals_result[train_key]["logloss"]
        val_logloss = evals_result[val_key]["logloss"]

        plt.figure(figsize=(8, 6))
        plt.plot(train_logloss, label="Train logloss", linewidth=2)
        plt.plot(val_logloss, label="Test logloss", linewidth=2)
        plt.xlabel("Boosting Round")
        plt.ylabel("Logloss")
        plt.title("Training Curve - Logloss")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        save_path = os.path.join(save_dir, "training_curve_logloss.png")
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"Saved training logloss curve to: {save_path}")

    if "auc" in evals_result[train_key] and "auc" in evals_result[val_key]:
        train_auc = evals_result[train_key]["auc"]
        val_auc = evals_result[val_key]["auc"]

        plt.figure(figsize=(8, 6))
        plt.plot(train_auc, label="Train AUC", linewidth=2)
        plt.plot(val_auc, label="Test AUC", linewidth=2)
        plt.xlabel("Boosting Round")
        plt.ylabel("AUC")
        plt.title("Training Curve - AUC")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        save_path = os.path.join(save_dir, "training_curve_auc.png")
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"Saved training AUC curve to: {save_path}")


def plot_feature_importance_fig(feat_imp, top_n=20, save_path="feature_importance_top20.png"):
    top_feat = feat_imp.head(top_n).copy().iloc[::-1]

    plt.figure(figsize=(10, 8))
    plt.barh(top_feat["feature"], top_feat["importance"])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"Saved feature importance figure to: {save_path}")


def build_binary_sample_weight(
    df_train,
    y_train,
    significance_value=TARGET_SIGNIFICANCE,
    pos_significance_multiplier=POS_SIGNIFICANCE2_WEIGHT_MULTIPLIER,
    hard_negative_multiplier=HARD_NEG_SIGNIFICANCE2_WEIGHT_MULTIPLIER,
):
    y_arr = np.asarray(y_train)
    weights = np.ones(len(df_train), dtype=float)

    neg = int((y_arr == 0).sum())
    pos = int((y_arr == 1).sum())
    pos_base_weight = neg / pos if pos > 0 else 1.0
    weights[y_arr == 1] = pos_base_weight

    if "significance" not in df_train.columns:
        return weights

    significance = pd.to_numeric(df_train["significance"], errors="coerce").to_numpy()
    weights[(y_arr == 1) & (significance == significance_value)] *= pos_significance_multiplier
    weights[(y_arr == 0) & (significance == significance_value)] *= hard_negative_multiplier
    return weights


def build_ternary_severity_label(df, significance_value=TARGET_SIGNIFICANCE):
    label = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int).to_numpy()
    significance = pd.to_numeric(df["significance"], errors="coerce").fillna(0).astype(int).to_numpy()
    ternary = np.zeros(len(df), dtype=int)
    ternary[(label == 1) & (significance != significance_value)] = 1
    ternary[(label == 1) & (significance == significance_value)] = 2
    return ternary


def build_ternary_sample_weight(
    y_train_ternary,
    significant_multiplier=TERNARY_SIGNIFICANT_CLASS_WEIGHT_MULTIPLIER,
):
    y_arr = np.asarray(y_train_ternary, dtype=int)
    weights = np.ones(len(y_arr), dtype=float)
    class_counts = np.bincount(y_arr, minlength=3)
    non_zero_counts = class_counts[class_counts > 0]
    base_count = float(non_zero_counts.max()) if non_zero_counts.size > 0 else 1.0

    for cls, count in enumerate(class_counts):
        if count > 0:
            weights[y_arr == cls] = base_count / float(count)

    weights[y_arr == 2] *= float(significant_multiplier)
    return weights


def compute_feature_signature_overlap_stats(df_a, df_b, feature_cols, round_digits=8):
    if not feature_cols:
        return {
            "common_signature_count": 0,
            "rows_in_a_with_common_signature": 0,
            "rows_in_b_with_common_signature": 0,
        }

    sig_a = pd.Series(
        list(map(tuple, df_a[feature_cols].round(round_digits).fillna(1e308).to_numpy().tolist())),
        index=df_a.index,
    )
    sig_b = pd.Series(
        list(map(tuple, df_b[feature_cols].round(round_digits).fillna(1e308).to_numpy().tolist())),
        index=df_b.index,
    )
    common = set(sig_a.unique()).intersection(set(sig_b.unique()))
    return {
        "common_signature_count": int(len(common)),
        "rows_in_a_with_common_signature": int(sig_a.isin(common).sum()),
        "rows_in_b_with_common_signature": int(sig_b.isin(common).sum()),
    }


def get_monitor_feature_cols(df):
    return [
        c for c in df.columns
        if c.startswith(("cos_sim_", "mean_diff_", "std_diff_", "l2_distance_"))
    ]


def compute_all_feature_nan_mask(df):
    feature_cols = get_monitor_feature_cols(df)
    if not feature_cols:
        return np.zeros(len(df), dtype=bool)
    return df[feature_cols].isna().all(axis=1).to_numpy()


def summarize_final_alert_subset(
    df_eval,
    subset_mask,
    final_y_pred,
    y_eval_binary,
    y_eval_significant,
    final_decision_score,
    subset_name,
    target_significance=TARGET_SIGNIFICANCE,
):
    subset_mask = np.asarray(subset_mask, dtype=bool)
    subset_size = int(subset_mask.sum())
    significance = pd.to_numeric(df_eval["significance"], errors="coerce").fillna(0).astype(int).to_numpy()

    raw_label_significance_counts = (
        df_eval.loc[subset_mask, ["label", "significance"]]
        .value_counts(dropna=False)
        .sort_index()
        .to_dict()
    )
    label_significance_counts = {
        f"label={k[0]}, significance={k[1]}": int(v)
        for k, v in raw_label_significance_counts.items()
    }

    if subset_size == 0:
        summary = {
            "subset_size": 0,
            "label_significance_counts": label_significance_counts,
        }
        print(f"{subset_name}: empty subset, skip metrics.")
        print("=" * 60)
        return summary

    subset_pred = np.asarray(final_y_pred)[subset_mask]
    subset_y_binary = np.asarray(y_eval_binary)[subset_mask]
    subset_y_significant = np.asarray(y_eval_significant)[subset_mask]
    subset_score = np.asarray(final_decision_score)[subset_mask]
    subset_significance = significance[subset_mask]

    cm_binary = confusion_matrix(subset_y_binary, subset_pred, labels=[0, 1]).tolist()
    binary_metrics = print_metrics_binary(
        subset_y_binary,
        subset_pred,
        y_prob=subset_score,
        title=f"{subset_name} Final Alert vs label"
    )
    significant_metrics = print_metrics_binary(
        subset_y_significant,
        subset_pred,
        y_prob=subset_score,
        title=f"{subset_name} Final Alert vs significant-SDC target"
    )

    sdc_mask = subset_y_binary == 1
    non_sdc_mask = subset_y_binary == 0
    target_mask = sdc_mask & (subset_significance == target_significance)
    pred_positive = subset_pred == 1

    sdc_total = int(sdc_mask.sum())
    non_sdc_total = int(non_sdc_mask.sum())
    target_total = int(target_mask.sum())
    sdc_recalled = int((sdc_mask & pred_positive).sum())
    target_recalled = int((target_mask & pred_positive).sum())
    non_sdc_false_positive = int((non_sdc_mask & pred_positive).sum())
    pred_positive_total = int(pred_positive.sum())

    summary = {
        "subset_size": subset_size,
        "label_significance_counts": label_significance_counts,
        "confusion_matrix_vs_label": cm_binary,
        "final_alert_vs_label_metrics": binary_metrics,
        "final_alert_vs_significant_target_metrics": significant_metrics,
        "sdc_total": sdc_total,
        "sdc_recalled": sdc_recalled,
        "sdc_recall": None if sdc_total == 0 else sdc_recalled / sdc_total,
        "target_sdc_total": target_total,
        "target_sdc_recalled": target_recalled,
        "target_sdc_recall": None if target_total == 0 else target_recalled / target_total,
        "non_sdc_total": non_sdc_total,
        "non_sdc_false_positive": non_sdc_false_positive,
        "non_sdc_fpr": None if non_sdc_total == 0 else non_sdc_false_positive / non_sdc_total,
        "pred_positive_total": pred_positive_total,
        "target_alert_precision": None if pred_positive_total == 0 else target_recalled / pred_positive_total,
    }

    print(f"{subset_name} summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("=" * 60)
    return summary


def build_xgb_binary_classifier(scale_pos_weight=1.0):
    return XGBClassifier(
        n_estimators=10000,
        learning_rate=0.01,
        max_depth=4,
        min_child_weight=1,
        scale_pos_weight=scale_pos_weight,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        early_stopping_rounds=500,
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=8
    )


def split_train_test_by_group(df, X, y, group_col="sample_uid", test_size=0.2, random_state=42):
    if group_col not in df.columns:
        raise ValueError(f"列 {group_col} 不存在，无法按组切分")

    groups = df[group_col]
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    split_info = {
        "split_mode": "group_shuffle",
        "group_col": group_col,
        "test_size": test_size,
        "random_state": random_state
    }
    return train_idx, test_idx, split_info

# def split_train_test(df, X, y, orig_id_col="orig_id"):
#     if orig_id_col not in df.columns:
#         raise ValueError(f"列 {orig_id_col} 不存在，无法按 orig_id 范围切分")

#     test_mask = (df[orig_id_col] > 3200) & (df[orig_id_col] < 4000)
#     test_idx = np.flatnonzero(test_mask.to_numpy())
#     train_idx = np.flatnonzero((~test_mask).to_numpy())

#     if len(test_idx) == 0:
#         raise ValueError("test_idx 为空，请检查 orig_id 条件是否正确")

#     if len(train_idx) == 0:
#         raise ValueError("train_idx 为空，请检查 orig_id 条件是否过于宽泛")

#     split_info = {
#         "split_mode": "orig_id_range",
#         "orig_id_col": orig_id_col,
#         "test_condition": "3000 < orig_id < 4000",
#         "train_size": len(train_idx),
#         "test_size": len(test_idx),
#     }

#     return train_idx, test_idx, split_info

def evaluate_dataset(
    model,
    df_eval,
    X_eval,
    y_eval,
    task_type,
    dataset_name,
    target_significance=TARGET_SIGNIFICANCE,
    max_non_sdc_fpr=TARGET_NON_SDC_FPR,
    severity_model=None,
    figure_dir=None,
    save_prefix=os.path.join(OUTPUT_DIR, "eval")
):
    print(f"Evaluate on {dataset_name}...")
    print("=" * 60)

    y_prob_all = model.predict_proba(X_eval)

    if task_type == "severity_ternary":
        y_eval_binary = pd.to_numeric(df_eval["label"], errors="coerce").fillna(0).astype(int).to_numpy()
        y_eval_significant = (
            (y_eval_binary == 1)
            & (pd.to_numeric(df_eval["significance"], errors="coerce").fillna(0).astype(int).to_numpy() == target_significance)
        ).astype(int)
        y_eval_ternary = np.asarray(y_eval).astype(int)
        y_pred_ternary_argmax = np.argmax(y_prob_all, axis=1)

        prob_any_sdc = y_prob_all[:, 1] + y_prob_all[:, 2]
        prob_significant_sdc = y_prob_all[:, 2]

        y_pred_default = (prob_significant_sdc >= 0.5).astype(int)
        default_metrics = print_metrics_binary(
            y_eval_significant,
            y_pred_default,
            y_prob=prob_significant_sdc,
            title=f"{dataset_name} Significant-SDC Metrics @ threshold=0.50"
        )

        best_thr, best_f1 = find_best_threshold(y_eval_significant, prob_significant_sdc, start=0, end=0.95, step=0.01)
        y_pred_best = (prob_significant_sdc >= best_thr).astype(int)

        print(f"{dataset_name} best threshold for significant-SDC F1: {best_thr:.2f}")
        print(f"{dataset_name} best significant-SDC F1 by threshold search: {best_f1:.6f}")
        print("=" * 60)

        best_metrics = print_metrics_binary(
            y_eval_significant,
            y_pred_best,
            y_prob=prob_significant_sdc,
            title=f"{dataset_name} Significant-SDC Metrics @ best threshold={best_thr:.2f}"
        )

        any_sdc_threshold_rows = evaluate_significance_recall_by_threshold(
            df_eval=df_eval,
            y_true=y_eval_binary,
            y_score=prob_any_sdc,
            significance_value=target_significance,
            start=0,
            end=0.95,
            step=0.05,
            save_path=f"{save_prefix}_ternary_prob_any_sdc_threshold_recall.csv",
        )
        any_sdc_operating_point = choose_threshold_by_target_recall(
            any_sdc_threshold_rows,
            max_non_sdc_fpr=max_non_sdc_fpr,
        )

        significant_threshold_rows = evaluate_significance_recall_by_threshold(
            df_eval=df_eval,
            y_true=y_eval_binary,
            y_score=prob_significant_sdc,
            significance_value=target_significance,
            start=0,
            end=0.95,
            step=0.05,
            save_path=f"{save_prefix}_ternary_prob_significant_sdc_threshold_recall.csv",
        )
        significant_operating_point = choose_threshold_by_target_recall(
            significant_threshold_rows,
            max_non_sdc_fpr=max_non_sdc_fpr,
        )

        extra_columns = {
            "prob_non_sdc": y_prob_all[:, 0],
            "prob_normal_sdc": y_prob_all[:, 1],
            "prob_significant_sdc": prob_significant_sdc,
            "prob_any_sdc": prob_any_sdc,
        }
        final_decision_source = "significant_sdc_best_threshold_f1"
        final_decision_threshold = float(best_thr)
        final_decision_score = prob_significant_sdc
        final_y_pred = y_pred_best

        if significant_operating_point is not None:
            final_decision_source = "prob_significant_sdc_target_operating_point"
            final_decision_threshold = float(significant_operating_point["threshold"])
            final_decision_score = prob_significant_sdc
            final_y_pred = (prob_significant_sdc >= final_decision_threshold).astype(int)

        extra_columns["final_decision_source"] = np.full(len(df_eval), final_decision_source, dtype=object)
        extra_columns["final_decision_threshold"] = np.full(len(df_eval), final_decision_threshold, dtype=float)
        extra_columns["final_decision_score"] = final_decision_score
        all_feature_nan_mask = compute_all_feature_nan_mask(df_eval)
        extra_columns["all_feature_nan"] = all_feature_nan_mask.astype(int)
        extra_columns["ternary_true"] = y_eval_ternary
        extra_columns["ternary_pred_argmax"] = y_pred_ternary_argmax

        print(f"{dataset_name} final decision source: {final_decision_source}")
        print(f"{dataset_name} final decision threshold: {final_decision_threshold:.6f}")
        print("=" * 60)

        subset_reports = {
            "all_feature_nan": summarize_final_alert_subset(
                df_eval=df_eval,
                subset_mask=all_feature_nan_mask,
                final_y_pred=final_y_pred,
                y_eval_binary=y_eval_binary,
                y_eval_significant=y_eval_significant,
                final_decision_score=final_decision_score,
                subset_name=f"{dataset_name} / all_feature_nan",
                target_significance=target_significance,
            ),
            "non_all_feature_nan": summarize_final_alert_subset(
                df_eval=df_eval,
                subset_mask=~all_feature_nan_mask,
                final_y_pred=final_y_pred,
                y_eval_binary=y_eval_binary,
                y_eval_significant=y_eval_significant,
                final_decision_score=final_decision_score,
                subset_name=f"{dataset_name} / non_all_feature_nan",
                target_significance=target_significance,
            ),
        }

        ternary_argmax_cm = confusion_matrix(y_eval_ternary, y_pred_ternary_argmax, labels=[0, 1, 2])
        significant_decision_cm = confusion_matrix(y_eval_significant, final_y_pred, labels=[0, 1])
        label_level_protection_cm = confusion_matrix(y_eval_binary, final_y_pred, labels=[0, 1])

        print(f"{dataset_name} Ternary Argmax Confusion Matrix (rows=true 0/1/2, cols=pred 0/1/2):")
        print(ternary_argmax_cm)
        print("=" * 60)
        print(f"{dataset_name} Significant-SDC Decision Confusion Matrix (rows=true non-target/target, cols=pred no-alert/alert):")
        print(significant_decision_cm)
        print("=" * 60)
        print(f"{dataset_name} Label-Level Protection Confusion Matrix (rows=true label 0/1, cols=pred no-alert/alert):")
        print(label_level_protection_cm)
        print("=" * 60)

        metrics_dict = {
            "ternary_default_threshold_metrics": default_metrics,
            "ternary_best_threshold": float(best_thr),
            "ternary_best_threshold_metrics": best_metrics,
            "ternary_argmax_confusion_matrix": ternary_argmax_cm.tolist(),
            "significant_sdc_decision_confusion_matrix": significant_decision_cm.tolist(),
            "label_level_protection_confusion_matrix": label_level_protection_cm.tolist(),
            "prob_any_sdc_metrics": {
                "score_name": "prob_any_sdc",
                "threshold_rows": any_sdc_threshold_rows,
                "selected_operating_point": any_sdc_operating_point,
            },
            "prob_significant_sdc_metrics": {
                "score_name": "prob_significant_sdc",
                "threshold_rows": significant_threshold_rows,
                "selected_operating_point": significant_operating_point,
            },
            "final_prediction_source": final_decision_source,
            "final_prediction_threshold": final_decision_threshold,
            "subset_reports_by_all_feature_nan": subset_reports,
        }

        cm = significant_decision_cm

        save_predictions(
            df=df_eval,
            y_true=y_eval_binary,
            y_pred=final_y_pred,
            y_prob_all=y_prob_all,
            task_type="binary",
            extra_columns=extra_columns,
            save_all_path=f"{save_prefix}_predictions.csv",
            save_wrong_path=f"{save_prefix}_wrong_predictions.csv"
        )

        if figure_dir is not None:
            figure_prefix = os.path.basename(save_prefix)
            plot_significance_threshold_tradeoff(
                significant_threshold_rows,
                save_path=os.path.join(figure_dir, f"{figure_prefix}_significance2_threshold_tradeoff.png"),
                title=f"{dataset_name} Significant-SDC Recall / Non-SDC FPR Tradeoff",
            )
            plot_confusion_matrix_fig(
                ternary_argmax_cm,
                save_path=os.path.join(figure_dir, f"{figure_prefix}_ternary_argmax_confusion_matrix.png"),
                class_names=["non-SDC", "SDC", "sig-SDC"],
                title=f"{dataset_name} Ternary Argmax Confusion Matrix"
            )
            plot_confusion_matrix_fig(
                significant_decision_cm,
                save_path=os.path.join(figure_dir, f"{figure_prefix}_significant_sdc_decision_confusion_matrix.png"),
                class_names=["non-target", "sig-SDC"],
                title=f"{dataset_name} Significant-SDC Decision Matrix"
            )
            plot_confusion_matrix_fig(
                label_level_protection_cm,
                save_path=os.path.join(figure_dir, f"{figure_prefix}_label_level_protection_confusion_matrix.png"),
                class_names=["label=0", "label=1"],
                title=f"{dataset_name} Label-Level Protection Matrix"
            )
            if len(np.unique(y_eval_binary)) > 1:
                plot_roc_curve_binary(
                    y_eval_binary,
                    prob_significant_sdc,
                    save_path=os.path.join(figure_dir, f"{figure_prefix}_roc_curve.png"),
                    title=f"{dataset_name} Significant-SDC ROC Curve"
                )

        return metrics_dict, cm

    if task_type == "binary":
        y_prob = y_prob_all[:, 1]

        y_pred_default = (y_prob >= 0.5).astype(int)
        default_metrics = print_metrics_binary(
            y_eval,
            y_pred_default,
            y_prob=y_prob,
            title=f"{dataset_name} Metrics @ threshold=0.50"
        )

        best_thr, best_f1 = find_best_threshold(y_eval, y_prob, start=0, end=0.95, step=0.01)
        y_pred_best = (y_prob >= best_thr).astype(int)

        print(f"{dataset_name} best threshold for F1: {best_thr:.2f}")
        print(f"{dataset_name} best F1 by threshold search: {best_f1:.6f}")
        print("=" * 60)

        best_metrics = print_metrics_binary(
            y_eval,
            y_pred_best,
            y_prob=y_prob,
            title=f"{dataset_name} Metrics @ best threshold={best_thr:.2f}"
        )
        threshold_recall_path = f"{save_prefix}_significance2_threshold_recall.csv"
        significance_recall_by_threshold = evaluate_significance_recall_by_threshold(
            df_eval=df_eval,
            y_true=y_eval.values,
            y_score=y_prob,
            significance_value=target_significance,
            start=0,
            end=0.95,
            step=0.05,
            save_path=threshold_recall_path,
        )
        target_operating_point = choose_threshold_by_target_recall(
            significance_recall_by_threshold,
            max_non_sdc_fpr=max_non_sdc_fpr,
        )

        extra_columns = {
            "prob_sdc": y_prob,
        }
        final_decision_source = "best_threshold_f1"
        final_decision_threshold = float(best_thr)
        final_decision_score = y_prob

        severity_metrics = None
        if severity_model is not None:
            y_prob_sig2 = severity_model.predict_proba(X_eval)[:, 1]
            severe_score = y_prob * y_prob_sig2
            extra_columns["prob_significance2_given_sdc"] = y_prob_sig2
            extra_columns["severe_score"] = severe_score

            severe_threshold_rows = evaluate_significance_recall_by_threshold(
                df_eval=df_eval,
                y_true=y_eval.values,
                y_score=severe_score,
                significance_value=target_significance,
                start=0,
                end=0.95,
                step=0.05,
                save_path=f"{save_prefix}_significance2_severe_score_threshold_recall.csv",
            )
            severe_operating_point = choose_threshold_by_target_recall(
                severe_threshold_rows,
                max_non_sdc_fpr=max_non_sdc_fpr,
            )

            severity_metrics = {
                "score_name": "prob_sdc * prob_significance2_given_sdc",
                "threshold_rows": severe_threshold_rows,
                "selected_operating_point": severe_operating_point,
            }
            if severe_operating_point is not None:
                final_decision_source = "severe_score_selected_operating_point"
                final_decision_threshold = float(severe_operating_point["threshold"])
                final_decision_score = severe_score
                final_y_pred = (severe_score >= final_decision_threshold).astype(int)
            else:
                final_y_pred = y_pred_best
        elif target_operating_point is not None:
            final_decision_source = "prob_sdc_target_operating_point"
            final_decision_threshold = float(target_operating_point["threshold"])
            final_decision_score = y_prob
            final_y_pred = (y_prob >= final_decision_threshold).astype(int)
        else:
            final_y_pred = y_pred_best

        extra_columns["final_decision_source"] = np.full(len(df_eval), final_decision_source, dtype=object)
        extra_columns["final_decision_threshold"] = np.full(len(df_eval), final_decision_threshold, dtype=float)
        extra_columns["final_decision_score"] = final_decision_score

        print(f"{dataset_name} final decision source: {final_decision_source}")
        print(f"{dataset_name} final decision threshold: {final_decision_threshold:.6f}")
        print("=" * 60)

        metrics_dict = {
            "default_threshold_metrics": default_metrics,
            "best_threshold": float(best_thr),
            "best_threshold_metrics": best_metrics,
            "significance2_recall_by_threshold": significance_recall_by_threshold,
            "target_operating_point": target_operating_point,
            "severe_score_metrics": severity_metrics,
            "final_prediction_source": final_decision_source,
            "final_prediction_threshold": final_decision_threshold,
        }

    else:
        final_y_pred = np.argmax(y_prob_all, axis=1)
        extra_columns = None

        multiclass_metrics = print_metrics_multiclass(
            y_eval,
            final_y_pred,
            y_prob=y_prob_all,
            title=f"{dataset_name} Metrics (Multiclass)"
        )

        metrics_dict = {
            "multiclass_metrics": multiclass_metrics
        }

    cm = confusion_matrix(y_eval, final_y_pred)
    print(f"{dataset_name} Confusion Matrix:")
    print(cm)
    print("=" * 60)

    save_predictions(
        df=df_eval,
        y_true=y_eval.values,
        y_pred=final_y_pred,
        y_prob_all=y_prob_all,
        task_type=task_type,
        extra_columns=extra_columns,
        save_all_path=f"{save_prefix}_predictions.csv",
        save_wrong_path=f"{save_prefix}_wrong_predictions.csv"
    )

    if figure_dir is not None:
        figure_prefix = os.path.basename(save_prefix)
        if task_type == "binary":
            plot_significance_threshold_tradeoff(
                significance_recall_by_threshold,
                save_path=os.path.join(figure_dir, f"{figure_prefix}_significance2_threshold_tradeoff.png"),
                title=f"{dataset_name} SDC Recall / Significance==2 SDC Recall / Non-SDC FPR Tradeoff",
            )

        plot_confusion_matrix_fig(
            cm,
            save_path=os.path.join(figure_dir, f"{figure_prefix}_confusion_matrix.png"),
            class_names=[str(i) for i in range(len(np.unique(pd.concat([pd.Series(y_eval), pd.Series(final_y_pred)]))))],
            title=f"{dataset_name} Confusion Matrix"
        )

        if len(set(y_eval)) > 1:
            if task_type == "binary":
                plot_roc_curve_binary(
                    y_eval,
                    y_prob,
                    save_path=os.path.join(figure_dir, f"{figure_prefix}_roc_curve.png"),
                    title=f"{dataset_name} ROC Curve"
                )
            else:
                plot_roc_curve_multiclass(
                    y_eval,
                    y_prob_all,
                    n_classes=y_prob_all.shape[1],
                    save_path=os.path.join(figure_dir, f"{figure_prefix}_roc_curve_multiclass.png"),
                    title=f"{dataset_name} ROC Curve (OvR)"
                )

    return metrics_dict, cm


# =========================
# 主流程
# =========================
def main():
    plt.rcParams["axes.unicode_minus"] = False
    ensure_dir(OUTPUT_DIR)
    ensure_dir(FIGURE_DIR)

    # -------------------------
    # 读取数据
    # -------------------------
    train_df_full = pd.read_csv(TRAIN_CSV)
    valid_df = pd.read_csv(VALID_CSV)

    print("=" * 60)
    print(f"Loaded train data from: {TRAIN_CSV}")
    print(f"Train rows: {len(train_df_full)}")
    print(f"Loaded valid data from: {VALID_CSV}")
    print(f"Valid rows: {len(valid_df)}")
    print("=" * 60)

    if "label" not in train_df_full.columns:
        raise ValueError("train_set.csv 中缺少 label 列")
    if "label" not in valid_df.columns:
        raise ValueError("valid_set.csv 中缺少 label 列")

    train_df_full["binary_label"] = pd.to_numeric(train_df_full["label"], errors="coerce").fillna(0).astype(int)
    valid_df["binary_label"] = pd.to_numeric(valid_df["label"], errors="coerce").fillna(0).astype(int)

    if USE_TERNARY_SEVERITY_LABEL:
        train_df_full["ternary_label"] = build_ternary_severity_label(train_df_full, significance_value=TARGET_SIGNIFICANCE)
        valid_df["ternary_label"] = build_ternary_severity_label(valid_df, significance_value=TARGET_SIGNIFICANCE)
        all_labels = pd.concat([train_df_full["ternary_label"], valid_df["ternary_label"]], axis=0)
        unique_labels = sorted(pd.Series(all_labels).dropna().unique().tolist())
        print("Unique ternary labels:", unique_labels)
        if unique_labels != [0, 1, 2]:
            raise ValueError(f"三分类标签应为 [0,1,2]，检测到: {unique_labels}")
        task_type = "severity_ternary"
        num_classes = 3
    else:
        # -------------------------
        # 任务类型检测（基于 train+valid 全部标签）
        # -------------------------
        all_labels = pd.concat([train_df_full["label"], valid_df["label"]], axis=0)
        unique_labels = sorted(pd.Series(all_labels).dropna().unique().tolist())

        print("Unique labels:", unique_labels)

        if unique_labels == [0, 1]:
            task_type = "binary"
            num_classes = 2
        elif unique_labels == [0, 1, 2]:
            task_type = "multiclass"
            num_classes = 3
        elif unique_labels == [0, 1, 2, 3]:
            task_type = "multiclass"
            num_classes = 4
        else:
            raise ValueError(f"当前仅支持标签 [0,1] / [0,1,2] / [0,1,2,3]，检测到: {unique_labels}")

    print(f"Detected task type: {task_type}")
    print(f"Number of classes: {num_classes}")
    print("=" * 60)

    # -------------------------
    # 特征列
    # -------------------------
    exclude_cols = [
        "label",
        "group_id",
        "group_uid",
        "step_start",
        "step_end",
        "sample_id",
        "sample_uid",
        "orig_id",
          "significance",
          "binary_label",
          "ternary_label",
    ]
    exclude_cols.extend(LEAKAGE_PRONE_META_COLS)

    if "num_steps_in_group" in train_df_full.columns:
        exclude_cols.append("num_steps_in_group")

    feature_cols = [c for c in train_df_full.columns if c not in exclude_cols]

    # 对齐 valid 特征列
    missing_cols = [c for c in feature_cols if c not in valid_df.columns]
    if missing_cols:
        raise ValueError(f"valid_set.csv 缺少以下特征列: {missing_cols}")

    train_all_feature_nan_mask = compute_all_feature_nan_mask(train_df_full)
    valid_all_feature_nan_mask = compute_all_feature_nan_mask(valid_df)
    valid_df_full_eval = valid_df.copy().reset_index(drop=True)
    valid_df_non_all_nan_eval = valid_df.loc[~valid_all_feature_nan_mask].copy().reset_index(drop=True)
    all_feature_nan_filter_stats = {
        "drop_all_feature_nan": bool(DROP_ALL_FEATURE_NAN),
        "train_removed": int(train_all_feature_nan_mask.sum()),
        "valid_removed": int(valid_all_feature_nan_mask.sum()),
        "train_removed_label_significance_counts": {
            f"label={k[0]}, significance={k[1]}": int(v)
            for k, v in train_df_full.loc[train_all_feature_nan_mask, ["label", "significance"]]
            .value_counts(dropna=False)
            .sort_index()
            .to_dict()
            .items()
        },
        "valid_removed_label_significance_counts": {
            f"label={k[0]}, significance={k[1]}": int(v)
            for k, v in valid_df.loc[valid_all_feature_nan_mask, ["label", "significance"]]
            .value_counts(dropna=False)
            .sort_index()
            .to_dict()
            .items()
        },
    }

    if DROP_ALL_FEATURE_NAN:
        train_df_full = train_df_full.loc[~train_all_feature_nan_mask].reset_index(drop=True)
        valid_df = valid_df_non_all_nan_eval.copy()
        print("Dropped all_feature_nan samples before training/evaluation:")
        print(json.dumps(all_feature_nan_filter_stats, ensure_ascii=False, indent=2))
        print(f"Train rows after drop: {len(train_df_full)}")
        print(f"Valid rows after drop: {len(valid_df)}")
        print("=" * 60)

    X_full = train_df_full[feature_cols]
    y_full = train_df_full["ternary_label"] if task_type == "severity_ternary" else train_df_full["label"]

    X_valid = valid_df[feature_cols]
    y_valid = valid_df["ternary_label"] if task_type == "severity_ternary" else valid_df["label"]
    X_valid_full_eval = valid_df_full_eval[feature_cols]
    y_valid_full_eval = valid_df_full_eval["ternary_label"] if task_type == "severity_ternary" else valid_df_full_eval["label"]
    X_valid_non_all_nan_eval = valid_df_non_all_nan_eval[feature_cols]
    y_valid_non_all_nan_eval = valid_df_non_all_nan_eval["ternary_label"] if task_type == "severity_ternary" else valid_df_non_all_nan_eval["label"]

    print(f"Split mode: group_shuffle")
    print(f"Group column: {GROUP_COL}")
    print(f"Number of features: {len(feature_cols)}")
    print("Feature columns:")
    print(feature_cols)
    print("=" * 60)

    print("Train(full) label distribution:")
    print(y_full.value_counts(dropna=False).sort_index())
    print("=" * 60)

    print("Valid label distribution:")
    print(y_valid.value_counts(dropna=False).sort_index())
    print("=" * 60)

    # -------------------------
    # 缺失值检查
    # -------------------------
    total_nan_train = int(X_full.isna().sum().sum())
    total_nan_valid = int(X_valid.isna().sum().sum())
    print(f"Total NaN in train features: {total_nan_train}")
    print(f"Total NaN in valid features: {total_nan_valid}")
    print("=" * 60)

    # -------------------------
    # train_set.csv 内部切 train/test
    # -------------------------
    train_idx, test_idx, split_info = split_train_test_by_group(
        df=train_df_full,
        X=X_full,
        y=y_full,
        group_col=GROUP_COL,
        test_size=0.15,
        random_state=RANDOM_STATE
    )
    # train_idx, test_idx, split_info = split_train_test(
    #     df=train_df_full, 
    #     X=X_full,
    #     y=y_full,
    #     orig_id_col="orig_id"
    # )


    X_train, X_test = X_full.iloc[train_idx], X_full.iloc[test_idx]
    y_train, y_test = y_full.iloc[train_idx], y_full.iloc[test_idx]

    train_df = train_df_full.iloc[train_idx].copy()
    test_df = train_df_full.iloc[test_idx].copy()

    print("Train/Test split inside train_set.csv:")
    print(f"  Train rows: {len(X_train)}")
    print(f"  Test rows:  {len(X_test)}")
    print(f"  Train unique groups: {train_df[GROUP_COL].nunique()}")
    print(f"  Test unique groups:  {test_df[GROUP_COL].nunique()}")
    if "orig_id" in train_df.columns:
        print(f"  Train orig_id unique: {train_df['orig_id'].nunique()}")
        print(f"  Test orig_id unique:  {test_df['orig_id'].nunique()}")
    print("=" * 60)

    train_test_overlap_stats = compute_feature_signature_overlap_stats(train_df, test_df, feature_cols)
    train_valid_full_overlap_stats = compute_feature_signature_overlap_stats(train_df_full, valid_df_full_eval, feature_cols)
    train_valid_non_all_nan_overlap_stats = compute_feature_signature_overlap_stats(train_df_full, valid_df_non_all_nan_eval, feature_cols)
    print("Feature signature overlap stats:")
    print(json.dumps({
        "train_vs_test": train_test_overlap_stats,
        "train_full_vs_valid_full": train_valid_full_overlap_stats,
        "train_full_vs_valid_non_all_nan": train_valid_non_all_nan_overlap_stats,
    }, ensure_ascii=False, indent=2))
    print("=" * 60)

    # -------------------------
    # 建模参数
    # -------------------------
    sample_weight_train = None
    if task_type == "binary":
        neg = int((y_train == 0).sum())
        pos = int((y_train == 1).sum())
        scale_pos_weight = neg / pos if pos > 0 else 1.0
        sample_weight_train = build_binary_sample_weight(
            train_df,
            y_train,
            significance_value=TARGET_SIGNIFICANCE,
            pos_significance_multiplier=POS_SIGNIFICANCE2_WEIGHT_MULTIPLIER,
            hard_negative_multiplier=HARD_NEG_SIGNIFICANCE2_WEIGHT_MULTIPLIER,
        )

        print(f"Train negatives: {neg}")
        print(f"Train positives: {pos}")
        print(f"Suggested scale_pos_weight = {scale_pos_weight:.4f}")
        print(f"Sample weight min/max: {sample_weight_train.min():.4f} / {sample_weight_train.max():.4f}")
        print("=" * 60)

        model = build_xgb_binary_classifier(scale_pos_weight=1.0)

    elif task_type == "severity_ternary":
        ternary_counts = pd.Series(y_train).value_counts(dropna=False).sort_index().to_dict()
        sample_weight_train = build_ternary_sample_weight(
            y_train,
            significant_multiplier=TERNARY_SIGNIFICANT_CLASS_WEIGHT_MULTIPLIER,
        )

        print("Train ternary class distribution:", ternary_counts)
        print(f"Ternary sample weight min/max: {sample_weight_train.min():.4f} / {sample_weight_train.max():.4f}")
        print("=" * 60)

        model = XGBClassifier(
            n_estimators=10000,
            learning_rate=0.01,
            max_depth=4,
            min_child_weight=1,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            early_stopping_rounds=500,
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=8
        )
    else:
        model = XGBClassifier(
            n_estimators=10000,
            learning_rate=0.01,
            max_depth=4,
            min_child_weight=1,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            num_class=num_classes,
            eval_metric="mlogloss",
            early_stopping_rounds=500,
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=8
        )

    print("Start training...")
    print("=" * 60)

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=200
    )

    print("=" * 60)
    print("Training finished.")
    print("=" * 60)

    if hasattr(model, "best_iteration"):
        print(f"Best iteration: {model.best_iteration}")
    if hasattr(model, "best_score"):
        print(f"Best score: {model.best_score}")
    print("=" * 60)

    severity_model = None
    severity_train_stats = None
    if task_type == "binary":
        pos_train_mask = (y_train == 1).to_numpy()
        pos_test_mask = (y_test == 1).to_numpy()

        sig_train_target = (
            pd.to_numeric(train_df.loc[pos_train_mask, "significance"], errors="coerce") == TARGET_SIGNIFICANCE
        ).astype(int)
        sig_test_target = (
            pd.to_numeric(test_df.loc[pos_test_mask, "significance"], errors="coerce") == TARGET_SIGNIFICANCE
        ).astype(int)

        if sig_train_target.nunique() >= 2 and sig_test_target.nunique() >= 2:
            sig_neg = int((sig_train_target == 0).sum())
            sig_pos = int((sig_train_target == 1).sum())
            severity_scale_pos_weight = sig_neg / sig_pos if sig_pos > 0 else 1.0
            severity_train_stats = {
                "train_positive_samples": int(pos_train_mask.sum()),
                "train_significance2_positive_samples": sig_pos,
                "train_other_sdc_samples": sig_neg,
                "severity_scale_pos_weight": float(severity_scale_pos_weight),
            }

            print("Start training significance==2 head on SDC positives...")
            print(json.dumps(severity_train_stats, ensure_ascii=False, indent=2))
            print("=" * 60)

            severity_model = build_xgb_binary_classifier(scale_pos_weight=severity_scale_pos_weight)
            severity_model.fit(
                X_train.loc[pos_train_mask],
                sig_train_target,
                eval_set=[
                    (X_train.loc[pos_train_mask], sig_train_target),
                    (X_test.loc[pos_test_mask], sig_test_target),
                ],
                verbose=200
            )
            print("Finished training significance==2 head.")
            print("=" * 60)
        else:
            print("[Warning] significance==2 head skipped: train/test positive split does not contain both classes.")
            print("=" * 60)

    # -------------------------
    # 先评估 test（来自 train_set 内部切分）
    # -------------------------
    test_metrics, test_cm = evaluate_dataset(
        model=model,
        df_eval=test_df,
        X_eval=X_test,
        y_eval=y_test,
        task_type=task_type,
        dataset_name="Test (split from train_set)",
        target_significance=TARGET_SIGNIFICANCE,
        max_non_sdc_fpr=TARGET_NON_SDC_FPR,
        severity_model=severity_model,
        figure_dir=FIGURE_DIR,
        save_prefix=os.path.join(OUTPUT_DIR, "test")
    )

    # -------------------------
    # 再评估 valid_set.csv（外部验证集）：先保留 all_feature_nan，再去掉 all_feature_nan
    # -------------------------
    valid_full_metrics, valid_full_cm = evaluate_dataset(
        model=model,
        df_eval=valid_df_full_eval,
        X_eval=X_valid_full_eval,
        y_eval=y_valid_full_eval,
        task_type=task_type,
        dataset_name="Valid full (external valid_set.csv, keep all_feature_nan)",
        target_significance=TARGET_SIGNIFICANCE,
        max_non_sdc_fpr=TARGET_NON_SDC_FPR,
        severity_model=severity_model,
        figure_dir=FIGURE_DIR,
        save_prefix=os.path.join(OUTPUT_DIR, "valid_full")
    )

    valid_non_all_nan_metrics, valid_non_all_nan_cm = evaluate_dataset(
        model=model,
        df_eval=valid_df_non_all_nan_eval,
        X_eval=X_valid_non_all_nan_eval,
        y_eval=y_valid_non_all_nan_eval,
        task_type=task_type,
        dataset_name="Valid non-all-feature-NaN (external valid_set.csv)",
        target_significance=TARGET_SIGNIFICANCE,
        max_non_sdc_fpr=TARGET_NON_SDC_FPR,
        severity_model=severity_model,
        figure_dir=FIGURE_DIR,
        save_prefix=os.path.join(OUTPUT_DIR, "valid_non_all_nan")
    )

    # -------------------------
    # 特征重要性
    # -------------------------
    importance = model.feature_importances_
    feat_imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": importance
    }).sort_values("importance", ascending=False)
    plot_feature_importance_fig(
        feat_imp,
        top_n=20,
        save_path=os.path.join(FIGURE_DIR, "feature_importance_top20.png")
    )

    # -------------------------
    # 训练曲线
    # -------------------------
    evals_result = model.evals_result()
    plot_training_curves(evals_result, save_dir=FIGURE_DIR)

    # -------------------------
    # 保存结果汇总
    # -------------------------
    result_dict = {
        "task_type": task_type,
        "num_classes": num_classes,
        "split_info": split_info,
        "excluded_meta_cols_due_to_leakage_risk": LEAKAGE_PRONE_META_COLS,
        "all_feature_nan_filter_stats": all_feature_nan_filter_stats,
        "feature_signature_overlap_stats": {
            "train_vs_test": train_test_overlap_stats,
            "train_full_vs_valid_full": train_valid_full_overlap_stats,
            "train_full_vs_valid_non_all_nan": train_valid_non_all_nan_overlap_stats,
        },
        "best_iteration": int(model.best_iteration) if hasattr(model, "best_iteration") else None,
        "best_score": float(model.best_score) if hasattr(model, "best_score") else None,
        "feature_count": len(feature_cols),
        "target_significance": TARGET_SIGNIFICANCE,
        "target_non_sdc_fpr_budget": TARGET_NON_SDC_FPR,
        "test_metrics": test_metrics,
        "valid_full_metrics": valid_full_metrics,
        "valid_non_all_nan_metrics": valid_non_all_nan_metrics,
        "valid_metrics": valid_non_all_nan_metrics,
    }

    if task_type == "binary":
        result_dict["suggested_scale_pos_weight"] = float(scale_pos_weight)
        result_dict["sample_weight_strategy"] = {
            "pos_significance2_weight_multiplier": POS_SIGNIFICANCE2_WEIGHT_MULTIPLIER,
            "hard_neg_significance2_weight_multiplier": HARD_NEG_SIGNIFICANCE2_WEIGHT_MULTIPLIER,
        }
        result_dict["severity_head_train_stats"] = severity_train_stats
    elif task_type == "severity_ternary":
        result_dict["sample_weight_strategy"] = {
            "use_ternary_severity_label": True,
            "ternary_significant_class_weight_multiplier": TERNARY_SIGNIFICANT_CLASS_WEIGHT_MULTIPLIER,
        }

    metrics_summary_path = os.path.join(OUTPUT_DIR, "metrics_summary.json")
    with open(metrics_summary_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)

    print(f"Saved metrics summary to: {metrics_summary_path}")
    print(f"All outputs have been saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
