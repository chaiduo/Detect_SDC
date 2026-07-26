import argparse
import json
import os
import sys

import pandas as pd
import torch
from tqdm import tqdm

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
grandparent_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.extend([parent_dir, grandparent_dir])

from llava_common import generate_answer_from_path, load_llava_model, set_seed

def evaluate(
    model_path,
    data_dir,
    dataset_json,
    golden_json,
    device,
    model_base: str = None,
    max_samples: int = 5000,
):
    tokenizer, model, image_processor, model_name = load_llava_model(model_path, device, model_base)

    print(f"Loading dataset from {dataset_json}...")
    with open(dataset_json, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    df = pd.DataFrame(list(raw_data.items()), columns=["image_filename", "qa_list"])

    sample_id = 0
    all_samples = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        image_filename = row["image_filename"]
        qa_list = row["qa_list"]
        image_path = os.path.join(data_dir, image_filename)

        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} does not exist. Skipping.")
            continue

        cur_cnt = 0
        for qa_pair in tqdm(qa_list, leave=False, desc=f"QAs for {image_filename}"):
            question = qa_pair.get("Question", "")
            answer = qa_pair.get("Answer", "")
            question_type = qa_pair.get("Type", "")
            if question_type != "Comprehensive Analysis":
                continue
            if cur_cnt >= 2:
                continue
            cur_cnt += 1

            pred = generate_answer_from_path(
                question=question,
                image_path=image_path,
                tokenizer=tokenizer,
                model=model,
                image_processor=image_processor,
                model_name=model_name,
                max_new_tokens=50,
            )

            all_samples.append(
                {
                    "id": sample_id,
                    "image_path": image_filename,
                    "type": question_type,
                    "question": question,
                    "gt_answer": answer,
                    "pre_answer": pred,
                    "scores": 0,
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

    print(f"Results saved successfully to: {golden_json}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="/data0/home/lc/cd/llm/datasets/EarthVLSet/EarthVQA/Train/images_png")
    parser.add_argument("--dataset_json", type=str, default="/data0/home/lc/cd/llm/datasets/EarthVLSet/EarthVQA/Train_QA.json")
    parser.add_argument("--golden_json", type=str, default="./json/Golden_EarthVQA_llava-v1.5-7B_30_CA.json")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_samples", type=int, default=5000)
    args = parser.parse_args()

    set_seed(42)
    evaluate(
        model_path=args.model_path,
        model_base=args.model_base,
        data_dir=args.data_dir,
        dataset_json=args.dataset_json,
        golden_json=args.golden_json,
        device=torch.device(args.device),
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
