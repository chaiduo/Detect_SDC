import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
grandparent_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.extend([parent_dir, grandparent_dir])

from llava_common import collect_parquet_files, generate_answer_from_pil, load_image_from_parquet_cell, load_llava_model, set_seed


def evaluate(model_path, parquet_path, golden_json, device, model_base=None, max_samples=5000):
    tokenizer, model, image_processor, model_name = load_llava_model(model_path, device, model_base)
    parquet_files = collect_parquet_files(parquet_path)
    if not parquet_files:
        raise ValueError(f"No parquet files found in: {parquet_path}")

    all_samples = []
    sample_id = 0
    for file_idx, val_file in enumerate(parquet_files):
        print(f"[Info] Loading parquet [{file_idx + 1}/{len(parquet_files)}]: {val_file}")
        df = pd.read_parquet(val_file)
        required_cols = {"question", "multiple_choice_answer", "image"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns in {val_file}: {missing_cols}")

        for row_idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Evaluating {Path(val_file).name}"):
            question = str(row["question"]).strip()
            gt_answer = str(row["multiple_choice_answer"]).strip()
            pil_image = load_image_from_parquet_cell(row["image"])
            pred = generate_answer_from_pil(
                question=question,
                pil_image=pil_image,
                tokenizer=tokenizer,
                model=model,
                image_processor=image_processor,
                model_name=model_name,
                max_new_tokens=50,
            )
            all_samples.append(
                {
                    "id": sample_id,
                    "source_file": str(val_file),
                    "row_idx": int(row_idx),
                    "question": question,
                    "gt_answer": gt_answer,
                    "pre_answer": pred,
                    "score": 0,
                }
            )
            sample_id += 1
            if sample_id >= max_samples:
                break
        if sample_id >= max_samples:
            break

    os.makedirs(os.path.dirname(golden_json), exist_ok=True)
    with open(golden_json, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)

    print(f"[Done] Results saved successfully to: {golden_json}")
    print(f"[Done] Total saved samples: {len(all_samples)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--parquet_path", type=str, default="/data0/home/lc/cd/predict_error/Detect_SDC/Qwen2.5-VL-7B/VQAv2/")
    parser.add_argument("--golden_json", type=str, default="./json/Golden_VQAv2_llava-v1.5-7B_30_new.json")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_samples", type=int, default=5000)
    args = parser.parse_args()

    set_seed(42)
    evaluate(
        model_path=args.model_path,
        model_base=args.model_base,
        parquet_path=args.parquet_path,
        golden_json=args.golden_json,
        device=torch.device(args.device),
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
