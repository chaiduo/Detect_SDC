# sdc和no_sdc整体可视化
import math
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_jsonl(file_path):
    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def has_fault(sample):
    return sample.get("fault", None) is not None
def select_records_by_step(records, step_mode="all", k=None, step_range=None):
    if not records:
        return []

    # 先按 step 排序
    records = sorted(records, key=lambda x: x["step"])

    unique_steps = sorted({r["step"] for r in records})

    if step_mode == "all":
        selected_steps = set(unique_steps)

    elif step_mode == "last_k":
        if k is None or k <= 0:
            raise ValueError("When step_mode='last_k', k must be a positive integer")
        selected_steps = set(unique_steps[-k:])

    elif step_mode == "first_k":
        if k is None or k <= 0:
            raise ValueError("When step_mode='first_k', k must be a positive integer")
        selected_steps = set(unique_steps[:k])

    elif step_mode == "range":
        if step_range is None or len(step_range) != 2:
            raise ValueError("When step_mode='range', step_range must be (start, end)")
        start, end = step_range
        selected_steps = {s for s in unique_steps if start <= s <= end}

    else:
        raise ValueError(f"Unsupported step_mode: {step_mode}")

    return [r for r in records if r["step"] in selected_steps]


def is_valid_fault_sample(sample, after_threshold=1e30):
    fault = sample.get("fault", None)
    if not isinstance(fault, dict):
        return True

    after = fault.get("after", None)
    if after is None:
        return True

    try:
        after_val = float(after)
    except (TypeError, ValueError):
        return True

    if math.isnan(after_val):
        return False

    if after_val > after_threshold:
        return False

    if after_val < -after_threshold:
        return False

    return True


def filter_valid_samples(samples, after_threshold=1e9):
    filtered = [s for s in samples if is_valid_fault_sample(s, after_threshold=after_threshold)]
    skipped = len(samples) - len(filtered)

    print(f"[Info] total samples: {len(samples)}")
    print(f"[Info] valid samples: {len(filtered)}")
    print(f"[Info] skipped samples (fault.after is nan or abs(fault.after) > {after_threshold}): {skipped}")

    return filtered


def format_threshold(threshold):
    return f"{float(threshold):g}"


def parse_dtel_score(sample):
    dtel_score = sample.get("dtel_score", None)
    try:
        dtel_score = float(dtel_score)
    except (TypeError, ValueError):
        return None
    return abs(dtel_score)


def get_group_names(group_mode="ternary", dtel_threshold=0.3):
    if group_mode == "binary":
        return {
            "non_sdc": "non_sdc",
            "sdc": "sdc",
        }

    if group_mode == "ternary":
        thr_str = format_threshold(dtel_threshold)
        return {
            "non_sdc": "non_sdc",
            "lt": f"sdc_dtel_lt_{thr_str}",
            "ge": f"sdc_dtel_ge_{thr_str}",
        }

    if group_mode == "fault_aware":
        return {
            "no_fault": "no_fault",
            "fault_non_sdc": "fault_non_sdc",
            "fault_sdc": "fault_sdc",
        }

    raise ValueError(f"Unsupported group_mode: {group_mode}")


def get_group_order(group_mode="ternary", dtel_threshold=0.3):
    group_names = get_group_names(group_mode=group_mode, dtel_threshold=dtel_threshold)

    if group_mode == "binary":
        return [group_names["non_sdc"], group_names["sdc"]]

    if group_mode == "ternary":
        return [group_names["non_sdc"], group_names["lt"], group_names["ge"]]

    if group_mode == "fault_aware":
        return [
            group_names["no_fault"],
            group_names["fault_non_sdc"],
            group_names["fault_sdc"],
        ]

    raise ValueError(f"Unsupported group_mode: {group_mode}")


def get_sdc_group(sample, group_mode="ternary", dtel_threshold=0.3):
    is_sdc = int(sample.get("is_sdc", 0))
    fault_exists = has_fault(sample)
    group_names = get_group_names(group_mode=group_mode, dtel_threshold=dtel_threshold)

    if group_mode == "binary":
        return group_names["sdc"] if is_sdc == 1 else group_names["non_sdc"]

    if group_mode == "ternary":
        if is_sdc == 0:
            return group_names["non_sdc"]

        dtel_score = parse_dtel_score(sample)
        if dtel_score is not None and dtel_score < dtel_threshold:
            return group_names["lt"]
        else:
            return group_names["ge"]

    if group_mode == "fault_aware":
        if not fault_exists:
            return group_names["no_fault"]

        if is_sdc == 0:
            return group_names["fault_non_sdc"]

        return group_names["fault_sdc"]

    raise ValueError(f"Unsupported group_mode: {group_mode}")


def flatten_records(
    samples,
    group_mode="ternary",
    dtel_threshold=0.3,
    step_mode="all",
    k=None,
    step_range=None
):
    rows = []

    for sample_idx, sample in enumerate(samples):
        sample_id = sample.get("id", None)
        is_sdc = int(sample.get("is_sdc", 0))
        dtel_score = parse_dtel_score(sample)
        sdc_group = get_sdc_group(sample, group_mode=group_mode, dtel_threshold=dtel_threshold)

        mean_std_cos = sample.get("mean_std_cos", {})
        records = mean_std_cos.get("records", [])

        # 新增：按每个样本自己的 step 做筛选
        records = select_records_by_step(
            records,
            step_mode=step_mode,
            k=k,
            step_range=step_range
        )

        for r in records:
            src = r["src_layer"]
            tgt = r["tgt_layer"]
            step = r["step"]

            rows.append({
                "sample_idx": sample_idx,
                "sample_id": sample_id,
                "is_sdc": is_sdc,
                "has_fault": has_fault(sample),
                "dtel_score": dtel_score,
                "sdc_group": sdc_group,
                "step": step,
                "src_layer": src,
                "tgt_layer": tgt,
                "layer_pair": f"({src},{tgt})",
                "mean_diff": r["mean_diff"],
                "std_diff": r["std_diff"],
                "cos_sim": r["cos_sim"],
                "abs_mean_diff": abs(r["mean_diff"]),
                "abs_std_diff": abs(r["std_diff"]),
            })

    return pd.DataFrame(rows)



def plot_metric_by_layer_pair(df, metric, out_dir, group_mode="ternary", dtel_threshold=0.3):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    group_order = get_group_order(group_mode=group_mode, dtel_threshold=dtel_threshold)

    grouped = (
        df.groupby(["sdc_group", "src_layer", "tgt_layer", "layer_pair"], as_index=False)[metric]
        .mean()
    )

    layer_pair_order = (
        grouped[["src_layer", "tgt_layer", "layer_pair"]]
        .drop_duplicates()
        .sort_values(["src_layer", "tgt_layer"])["layer_pair"]
        .tolist()
    )

    grouped["layer_pair"] = pd.Categorical(
        grouped["layer_pair"],
        categories=layer_pair_order,
        ordered=True
    )

    grouped["sdc_group"] = pd.Categorical(
        grouped["sdc_group"],
        categories=group_order,
        ordered=True
    )

    grouped = grouped.sort_values(["src_layer", "tgt_layer", "sdc_group"])

    plt.figure(figsize=(14, 6))
    sns.lineplot(
        data=grouped,
        x="layer_pair",
        y=metric,
        hue="sdc_group",
        style="sdc_group",
        hue_order=group_order,
        style_order=group_order,
        markers=True,
        dashes=False,
    )
    plt.xticks(rotation=90)
    plt.xlabel("(src_layer, tgt_layer)")
    plt.ylabel(f"mean {metric}")

    if group_mode == "binary":
        title_suffix = " (binary)"
    elif group_mode == "ternary":
        title_suffix = f" (ternary, dtel_threshold={dtel_threshold})"
    else:
        title_suffix = " (fault_aware)"

    plt.title(f"Average {metric} by layer pair{title_suffix}")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(out_dir) / f"{metric}_by_layer_pair.png", dpi=150)
    plt.close()


def plot_metric_by_step(df, metric, out_dir, group_mode="ternary", dtel_threshold=0.3):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    group_order = get_group_order(group_mode=group_mode, dtel_threshold=dtel_threshold)

    grouped = (
        df.groupby(["sdc_group", "step"], as_index=False)[metric]
        .mean()
    )

    grouped["sdc_group"] = pd.Categorical(
        grouped["sdc_group"],
        categories=group_order,
        ordered=True
    )
    grouped = grouped.sort_values(["step", "sdc_group"])

    plt.figure(figsize=(12, 5))
    sns.lineplot(
        data=grouped,
        x="step",
        y=metric,
        hue="sdc_group",
        style="sdc_group",
        hue_order=group_order,
        style_order=group_order,
        markers=True,
        dashes=False,
    )
    plt.xlabel("step")
    plt.ylabel(f"mean {metric}")

    if group_mode == "binary":
        title_suffix = " (binary)"
    elif group_mode == "ternary":
        title_suffix = f" (ternary, dtel_threshold={dtel_threshold})"
    else:
        title_suffix = " (fault_aware)"

    plt.title(f"Average {metric} by decode step{title_suffix}")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(out_dir) / f"{metric}_by_step.png", dpi=150)
    plt.close()


def plot_metric_heatmap(df, metric, sdc_group, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    sub = df[df["sdc_group"] == sdc_group].copy()
    if len(sub) == 0:
        print(f"[Warn] no data for heatmap: metric={metric}, group={sdc_group}")
        return

    pivot = (
        sub.groupby(["step", "layer_pair"], as_index=False)[metric]
        .mean()
        .pivot(index="step", columns="layer_pair", values=metric)
    )

    ordered_cols = sorted(
        pivot.columns,
        key=lambda s: tuple(map(int, s.strip("()").split(",")))
    )
    pivot = pivot[ordered_cols]

    plt.figure(figsize=(16, 6))

    if metric == "cos_sim":
        sns.heatmap(
            pivot,
            cmap="coolwarm",
            vmin=0,
            vmax=1,
            center=None
        )
    else:
        sns.heatmap(
            pivot,
            cmap="coolwarm",
            center=0
        )

    plt.title(f"{metric} heatmap | group={sdc_group}")
    plt.xlabel("(src_layer, tgt_layer)")
    plt.ylabel("step")
    plt.tight_layout()
    plt.savefig(Path(out_dir) / f"{metric}_heatmap_{sdc_group}.png", dpi=150)
    plt.close()


def plot_metric_distribution(df, metric, out_dir, group_mode="ternary", dtel_threshold=0.3):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    group_order = get_group_order(group_mode=group_mode, dtel_threshold=dtel_threshold)

    plt.figure(figsize=(9, 5))
    sns.boxplot(
        data=df,
        x="sdc_group",
        y=metric,
        order=group_order
    )

    if group_mode == "binary":
        title_suffix = " (binary)"
    elif group_mode == "ternary":
        title_suffix = f" (ternary, dtel_threshold={dtel_threshold})"
    else:
        title_suffix = " (fault_aware)"

    plt.title(f"Distribution of {metric} by group{title_suffix}")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(out_dir) / f"{metric}_boxplot.png", dpi=150)
    plt.close()


def visualize_sdc_overall_difference(
    jsonl_path,
    out_dir="viz_overall",
    after_threshold=1e9,
    group_mode="ternary",
    dtel_threshold=0.3,
    step_mode="all",
    k=None,
    step_range=None
):

    if group_mode not in {"binary", "ternary", "fault_aware"}:
        raise ValueError(
            f"group_mode must be 'binary', 'ternary' or 'fault_aware', got: {group_mode}"
        )

    samples = load_jsonl(jsonl_path)
    samples = filter_valid_samples(samples, after_threshold=after_threshold)
    df = flatten_records(
        samples,
        group_mode=group_mode,
        dtel_threshold=dtel_threshold,
        step_mode=step_mode,
        k=k,
        step_range=step_range
    )


    if len(df) == 0:
        print("[Warn] no valid records found.")
        return

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    group_names = get_group_names(group_mode=group_mode, dtel_threshold=dtel_threshold)
    group_order = get_group_order(group_mode=group_mode, dtel_threshold=dtel_threshold)

    print(f"[Info] group_mode = {group_mode}")
    print(f"[Info] dtel_threshold = {dtel_threshold}")
    print(f"[Info] group names = {group_names}")

    print("[Info] valid sample counts by sdc_group:")
    sample_groups = [get_sdc_group(s, group_mode=group_mode, dtel_threshold=dtel_threshold) for s in samples]
    print(pd.Series(sample_groups).value_counts())

    print("[Info] flattened row counts by sdc_group:")
    print(df["sdc_group"].value_counts())

    print(f"[Info] flattened total rows: {len(df)}")
    print(f"[Info] unique sample_idx count: {df['sample_idx'].nunique()}")
    print(f"[Info] plot group order: {group_order}")

    metrics = ["cos_sim", "abs_mean_diff", "abs_std_diff"]

    for metric in metrics:
        plot_metric_by_layer_pair(df, metric, out_dir, group_mode=group_mode, dtel_threshold=dtel_threshold)
        plot_metric_by_step(df, metric, out_dir, group_mode=group_mode, dtel_threshold=dtel_threshold)

        for group in group_order:
            plot_metric_heatmap(df, metric, sdc_group=group, out_dir=out_dir)

        plot_metric_distribution(df, metric, out_dir, group_mode=group_mode, dtel_threshold=dtel_threshold)

    print(f"[Done] saved all plots to: {out_dir}")


if __name__ == "__main__":
    jsonl_path = "/data1/home/dataset_share/cd_data/Qwen2.5-VL-7B/EarthVQA/final/detect_EarthVQA_Qwen_with_sem_project.jsonl"
    # 1) 二分类
    visualize_sdc_overall_difference(
        jsonl_path=jsonl_path,
        out_dir="viz_overall_binary_last6",
        after_threshold=100,
        group_mode="binary",
        dtel_threshold=0.5,
        step_mode="last_k",
        k=6
    )

    # 2) 基于 dtel 的三分类
    visualize_sdc_overall_difference(
        jsonl_path=jsonl_path,
        out_dir="viz_overall_ternary-0.5_last6",
        after_threshold=100,
        group_mode="ternary",
        dtel_threshold=0.5,
        step_mode="last_k",
        k=6
    )

    # 3) fault-aware 三分类
    visualize_sdc_overall_difference(
        jsonl_path=jsonl_path,
        out_dir="viz_overall_fault_aware_last6",
        after_threshold=100,
        group_mode="fault_aware",
        dtel_threshold=0.5,
        step_mode="last_k",
        k=6
    )
