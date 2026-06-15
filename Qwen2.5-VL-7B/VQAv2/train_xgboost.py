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
from sklearn.utils.class_weight import compute_sample_weight

# =========================
# 配置区域
# =========================
TRAIN_CSV = "./train_data/Qwen2.5_VQAv2_train_set.csv"
VALID_CSV = "./train_data/Qwen2.5_VQAv2_valid_set.csv"
GROUP_COL = "orig_id"
FIGURE_DIR = "./figure"
RANDOM_STATE = 42

# 是否使用自动平衡权重 (False 则使用 CLASS_WEIGHTS)
USE_BALANCED_WEIGHTS = True

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
    save_all_path="predictions.csv",
    save_wrong_path="wrong_predictions.csv"
):
    """
    保存预测结果，以及预测错误样本
    """
    result_df = df.copy().reset_index(drop=False)
    result_df.rename(columns={"index": "raw_row_index"}, inplace=True)

    result_df["y_true"] = np.array(y_true)
    result_df["y_pred"] = np.array(y_pred)
    result_df["is_correct"] = (result_df["y_true"] == result_df["y_pred"]).astype(int)

    if task_type == "binary":
        result_df["prob_class_0"] = y_prob_all[:, 0]
        result_df["prob_class_1"] = y_prob_all[:, 1]
        result_df["y_prob_positive"] = y_prob_all[:, 1]

        wrong_df = result_df[result_df["is_correct"] == 0].copy()
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


def split_train_test_by_group(df, X, y, group_col="orig_id", test_size=0.2, random_state=42):
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


def evaluate_dataset(
    model,
    df_eval,
    X_eval,
    y_eval,
    task_type,
    dataset_name,
    figure_dir=None,
    save_prefix="eval"
):
    print(f"Evaluate on {dataset_name}...")
    print("=" * 60)

    y_prob_all = model.predict_proba(X_eval)

    if task_type == "binary":
        y_prob = y_prob_all[:, 1]

        y_pred_default = (y_prob >= 0.5).astype(int)
        default_metrics = print_metrics_binary(
            y_eval,
            y_pred_default,
            y_prob=y_prob,
            title=f"{dataset_name} Metrics @ threshold=0.50"
        )

        best_thr, best_f1 = find_best_threshold(y_eval, y_prob, start=0.05, end=0.95, step=0.01)
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

        final_y_pred = y_pred_best
        metrics_dict = {
            "default_threshold_metrics": default_metrics,
            "best_threshold": float(best_thr),
            "best_threshold_metrics": best_metrics
        }

    else:
        final_y_pred = np.argmax(y_prob_all, axis=1)

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
        save_all_path=f"{save_prefix}_predictions.csv",
        save_wrong_path=f"{save_prefix}_wrong_predictions.csv"
    )

    if figure_dir is not None:
        plot_confusion_matrix_fig(
            cm,
            save_path=os.path.join(figure_dir, f"{save_prefix}_confusion_matrix.png"),
            class_names=[str(i) for i in range(len(np.unique(pd.concat([pd.Series(y_eval), pd.Series(final_y_pred)]))))],
            title=f"{dataset_name} Confusion Matrix"
        )

        if len(set(y_eval)) > 1:
            if task_type == "binary":
                plot_roc_curve_binary(
                    y_eval,
                    y_prob,
                    save_path=os.path.join(figure_dir, f"{save_prefix}_roc_curve.png"),
                    title=f"{dataset_name} ROC Curve"
                )
            else:
                plot_roc_curve_multiclass(
                    y_eval,
                    y_prob_all,
                    n_classes=y_prob_all.shape[1],
                    save_path=os.path.join(figure_dir, f"{save_prefix}_roc_curve_multiclass.png"),
                    title=f"{dataset_name} ROC Curve (OvR)"
                )

    return metrics_dict, cm


# =========================
# 主流程
# =========================
def main():
    plt.rcParams["axes.unicode_minus"] = False
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
        "total_steps",
        "dtel_score",
        "num_steps_used",
        'total_steps', 'last_k_steps', 'num_steps_used'
    ]

    if "num_steps_in_group" in train_df_full.columns:
        exclude_cols.append("num_steps_in_group")

    feature_cols = [c for c in train_df_full.columns if c not in exclude_cols]

    # 对齐 valid 特征列
    missing_cols = [c for c in feature_cols if c not in valid_df.columns]
    if missing_cols:
        raise ValueError(f"valid_set.csv 缺少以下特征列: {missing_cols}")

    X_full = train_df_full[feature_cols]
    y_full = train_df_full["label"]

    X_valid = valid_df[feature_cols]
    y_valid = valid_df["label"]

    # ✅【核心修复 1】：在切分前统一将 inf/-inf 替换为 np.nan
    # 使用 pandas replace 可保留 DataFrame 结构与列名，避免 np.where 转成 ndarray 导致后续对齐报错
    inf_count_full = int((X_full.isin([np.inf, -np.inf])).sum().sum())
    inf_count_valid = int((X_valid.isin([np.inf, -np.inf])).sum().sum())
    print(f"Replacing {inf_count_full} inf values in X_full with NaN...")
    print(f"Replacing {inf_count_valid} inf values in X_valid with NaN...")
    
    X_full = X_full.replace([np.inf, -np.inf], np.nan)
    X_valid = X_valid.replace([np.inf, -np.inf], np.nan)

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

    # -------------------------
    # 建模参数
    # -------------------------
    sample_weights = None  
    if task_type == "binary":
        neg = int((y_train == 0).sum())
        pos = int((y_train == 1).sum())
        scale_pos_weight = neg / pos if pos > 0 else 1.0

        print(f"Train negatives: {neg}")
        print(f"Train positives: {pos}")
        print(f"Suggested scale_pos_weight = {scale_pos_weight:.4f}")
        print("=" * 60)

        model = XGBClassifier(
            n_estimators=20000,
            learning_rate=0.01,
            max_depth=6,
            min_child_weight=1,          # ↑ 提高叶子最小样本权重，防强特征在极小子集上过拟合
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="logloss",
            early_stopping_rounds=500,
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=8
        )

    else:
        # # 多分类权重处理
        sample_weights = compute_sample_weight(
            class_weight="balanced",
            y=y_train
        )

        model = XGBClassifier(
            n_estimators=20000,
            learning_rate=0.01,
            max_depth=6,
            colsample_bytree=0.8,
            colsample_bylevel=0.7,
            colsample_bynode=0.7,
            reg_alpha=0.5,
            reg_lambda=1.0,
            subsample=0.9,
            objective="multi:softprob",
            num_class=num_classes,
            eval_metric="mlogloss",
            early_stopping_rounds=500,
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=16
        )

    print("Start training...")
    print("=" * 60)

    if task_type == "binary":
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=100
        )
    else:
        model.fit(
            X_train, y_train,
            sample_weight=sample_weights,  # ✅ 传入多分类的样本权重
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=100
        )

    print("=" * 60)
    print("Training finished.")
    print("=" * 60)

    if hasattr(model, "best_iteration"):
        print(f"Best iteration: {model.best_iteration}")
    if hasattr(model, "best_score"):
        print(f"Best score: {model.best_score}")
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
        figure_dir=FIGURE_DIR,
        save_prefix="test"
    )

    # -------------------------
    # 再评估 valid_set.csv（外部验证集）
    # -------------------------
    valid_metrics, valid_cm = evaluate_dataset(
        model=model,
        df_eval=valid_df,
        X_eval=X_valid,
        y_eval=y_valid,
        task_type=task_type,
        dataset_name="Valid (external valid_set.csv)",
        figure_dir=FIGURE_DIR,
        save_prefix="valid"
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
        "best_iteration": int(model.best_iteration) if hasattr(model, "best_iteration") else None,
        "best_score": float(model.best_score) if hasattr(model, "best_score") else None,
        "feature_count": len(feature_cols),
        "test_metrics": test_metrics,
        "valid_metrics": valid_metrics
    }

    if task_type == "binary":
        neg = int((y_train == 0).sum())
        pos = int((y_train == 1).sum())
        scale_pos_weight = neg / pos if pos > 0 else 1.0
        result_dict["suggested_scale_pos_weight"] = float(scale_pos_weight)

    with open("metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)

    print("Saved metrics summary to: metrics_summary.json")
    print("All figures have been saved.")
    print("=" * 60)

if __name__ == "__main__":
    main()
