import os
import sys
import json
import traceback

import pandas as pd
from tqdm import tqdm
import torch

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
grandparent_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.extend([parent_dir, grandparent_dir])

from llava_common import generate_answer_from_path, load_llava_model, set_seed
from utils.Lingo_judge import ScoreEvaluator


def evaluate(
    model_path: str,
    val_file: str,
    data_dir: str,
    golden_json: str,
    device,
    score_evaluator: ScoreEvaluator,
    model_base: str = None,
):
    print(f"[Info] Loading model from: {model_path}")
    tokenizer, model, image_processor, model_name = load_llava_model(
        model_path=model_path,
        device=device,
        model_base=model_base,
    )

    print(f"[Info] Reading parquet file: {val_file}")
    df = pd.read_parquet(val_file)

    sample_id = 0
    all_samples = []

    print(f"[Info] Start evaluation, total rows: {len(df)}")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        question_id = row["question_id"]
        segment_id = row["segment_id"]
        question = row["question"]
        gt_answer = row["answer"]

        images = row["images"]
        num_images = min(5, len(images))

        for i in range(num_images):
            img_rel_path = images[i]
            image_path = os.path.join(data_dir, img_rel_path)

            try:
                pred = generate_answer_from_path(
                    question=question,
                    image_path=image_path,
                    tokenizer=tokenizer,
                    model=model,
                    image_processor=image_processor,
                    model_name=model_name,
                    max_new_tokens=50,
                )
            except Exception:
                print(f"[Warning] Failed on image: {image_path}")
                traceback.print_exc()
                pred = ""

            try:
                score = score_evaluator.get_score(question, gt_answer, pred)
            except Exception:
                print(f"[Warning] Scoring failed for sample_id={sample_id}")
                traceback.print_exc()
                score = None

            sample_data = {
                "id": sample_id,
                "question_id": question_id,
                "segment_id": segment_id,
                "image_path": img_rel_path,
                "question": question,
                "gt_answer": gt_answer,
                "pre_answer": pred,
                "score": score,
            }
            all_samples.append(sample_data)
            sample_id += 1

    out_dir = os.path.dirname(golden_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(golden_json, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)

    print(f"[Info] Results saved successfully to: {golden_json}")
    print(f"[Info] Total saved samples: {len(all_samples)}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--val_file", type=str, default="/data0/home/lc/cd/predict_error/LingoQA-main/data/val/val.parquet")
    parser.add_argument("--data_dir", type=str, default="/data0/home/lc/cd/predict_error/LingoQA-main/data/val/")
    parser.add_argument("--golden_json", type=str, default="./json/Golden_LingoQA_llava-v1.5-7B.json")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    set_seed(42)

    device = torch.device(args.device)
    score_evaluator = ScoreEvaluator()

    evaluate(
        model_path=args.model_path,
        model_base=args.model_base,
        val_file=args.val_file,
        data_dir=args.data_dir,
        golden_json=args.golden_json,
        device=device,
        score_evaluator=score_evaluator,
    )


if __name__ == "__main__":
    main()
