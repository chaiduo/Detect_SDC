import argparse
import json
import os
import random
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
grandparent_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.extend([parent_dir, grandparent_dir])

from llava_common import (
    append_jsonl,
    collect_parquet_files,
    generate_answer_from_pil,
    load_image_from_parquet_cell,
    load_llava_model,
    set_seed,
    to_jsonable,
)
from train_mapping_model import LayerAwareResidualMLP
from utils.fault_injector import FaultInjector
from utils.profiler import Profiler


def load_clean_answers(golden_json):
    with open(golden_json, "r", encoding="utf-8") as f:
        samples = json.load(f)
    return {int(item["id"]): item.get("pre_answer", "") for item in samples}


def build_result_record(idx, before_score, after_score, injector, source_file, row_idx, question, gt_answer, clean_pred, mean_std_cos, pred):
    return {
        "id": idx,
        "before_score": float(before_score),
        "after_score": float(after_score),
        "dtel_score": float(after_score - before_score),
        "is_sdc": int(clean_pred.strip() != pred.strip()),
        "fault": to_jsonable(getattr(injector, "fault_info", None)),
        "source_file": source_file,
        "row_idx": int(row_idx),
        "question": question,
        "gt_answer": gt_answer,
        "clean_answer": clean_pred,
        "pred_answer": pred,
        "mean_std_cos": mean_std_cos,
    }


def evaluate(
    model_path,
    parquet_path,
    output_jsonl,
    device,
    mapping_model_path,
    clean_answers,
    model_base=None,
    run_time=0,
    inject_fault=True,
    max_new_tokens=50,
    max_samples=5000,
):
    parquet_files = collect_parquet_files(parquet_path)
    if not parquet_files:
        raise ValueError(f"No parquet files found in: {parquet_path}")

    tokenizer, model, image_processor, model_name = load_llava_model(model_path, device, model_base)
    injector = FaultInjector(model, mode="activation")
    prof = Profiler(model, proj_dim=64, proj_method="project", seed=42)
    prof.register()

    mapping_model = LayerAwareResidualMLP(num_layers=32).to(device)
    mapping_model.load_state_dict(torch.load(mapping_model_path, map_location=device))
    mapping_model.eval()

    sample_id = 0
    for file_idx, val_file in enumerate(parquet_files):
        print(f"[Info] Loading parquet [{file_idx + 1}/{len(parquet_files)}]: {val_file}")
        df = pd.read_parquet(val_file)
        required_cols = {"question", "multiple_choice_answer", "image"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns in {val_file}: {missing_cols}")

        for row_idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Run={run_time} | {Path(val_file).name}"):
            question = str(row["question"]).strip()
            gt_answer = str(row["multiple_choice_answer"]).strip()
            pil_image = load_image_from_parquet_cell(row["image"])

            prof.reset(clear_stats=True)
            injector.reset()
            injector.register_step_hooks()
            if inject_fault:
                injector.set_num_bits(2)
                injector.inject()

            pred = generate_answer_from_pil(
                question=question,
                pil_image=pil_image,
                tokenizer=tokenizer,
                model=model,
                image_processor=image_processor,
                model_name=model_name,
                max_new_tokens=max_new_tokens,
            )

            prof.finalize()
            before_score = 0
            after_score = 0
            clean_pred = clean_answers.get(sample_id, "")
            sample_result = prof.get_attn_proj_model_compare_result(
                predictor_model=mapping_model,
                device=device,
                include_vectors=False,
            )
            result = build_result_record(
                idx=sample_id,
                before_score=before_score,
                after_score=after_score,
                injector=injector,
                source_file=str(val_file),
                row_idx=row_idx,
                question=question,
                gt_answer=gt_answer,
                clean_pred=clean_pred,
                mean_std_cos=sample_result,
                pred=pred,
            )

            if inject_fault is False or run_time < 1:
                append_jsonl(output_jsonl, result)
            else:
                if clean_pred.strip() != pred.strip():
                    append_jsonl(output_jsonl, result)

            injector.unregister_hooks()
            sample_id += 1
            if sample_id >= max_samples:
                break
        if sample_id >= max_samples:
            break

    prof.unregister()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--parquet_path", type=str, default="/data0/home/lc/cd/predict_error/Detect_SDC/Qwen2.5-VL-7B/VQAv2/train_data")
    parser.add_argument("--output_jsonl", type=str, default="/data1/home/dataset_share/cd_data/llava-v1.5-7B/VQAv2/final/detect_VQAv2_llava_with_sem_project.jsonl")
    parser.add_argument("--golden_json", type=str, default="./json/Golden_VQAv2_llava-v1.5-7B_30_new.json")
    parser.add_argument("--mapping_model", type=str, default="./model/best_mapping_model.pt")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_samples", type=int, default=5000)
    args = parser.parse_args()

    set_seed(42)
    device = torch.device(args.device)
    clean_answers = load_clean_answers(args.golden_json)

    evaluate(
        model_path=args.model_path,
        model_base=args.model_base,
        parquet_path=args.parquet_path,
        output_jsonl=args.output_jsonl,
        device=device,
        mapping_model_path=args.mapping_model,
        clean_answers=clean_answers,
        inject_fault=False,
        max_new_tokens=50,
        max_samples=args.max_samples,
    )
    for run in range(8):
        random.seed(42 + run)
        evaluate(
            model_path=args.model_path,
            model_base=args.model_base,
            parquet_path=args.parquet_path,
            output_jsonl=args.output_jsonl,
            device=device,
            mapping_model_path=args.mapping_model,
            clean_answers=clean_answers,
            run_time=run,
            inject_fault=True,
            max_new_tokens=50,
            max_samples=args.max_samples,
        )


if __name__ == "__main__":
    main()
