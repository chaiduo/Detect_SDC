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
from utils.profiler import Profiler


def evaluate(
    model_path,
    data_dir,
    dataset_json,
    output_jsonl,
    device,
    model_base: str = None,
    max_samples: int = 5000,
):
    tokenizer, model, image_processor, model_name = load_llava_model(model_path, device, model_base)

    print(f"Loading dataset from {dataset_json}...")
    with open(dataset_json, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    df = pd.DataFrame(list(raw_data.items()), columns=["image_filename", "qa_list"])

    prof = Profiler(model, proj_dim=64, proj_method="project", seed=42)
    prof.register()

    sample_id = 0
    for _, row in tqdm(df.iterrows(), total=len(df)):
        image_filename = row["image_filename"]
        qa_list = row["qa_list"]
        image_path = os.path.join(data_dir, image_filename)

        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} does not exist. Skipping.")
            continue

        cur_cnt = 0
        for qa_pair in tqdm(qa_list, leave=False, desc=f"QAs for {image_filename}"):
            question = qa_pair.get("Question", "")
            question_type = qa_pair.get("Type", "")
            if question_type != "Comprehensive Analysis":
                continue
            if cur_cnt >= 2:
                continue
            cur_cnt += 1

            _ = generate_answer_from_path(
                question=question,
                image_path=image_path,
                tokenizer=tokenizer,
                model=model,
                image_processor=image_processor,
                model_name=model_name,
                max_new_tokens=50,
            )

            prof.finalize()
            prof.save_attn_proj_interlayer_jsonl(output_jsonl, sample_id=sample_id)
            prof.reset(clear_stats=True)
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
    parser.add_argument("--data_dir", type=str, default="/data0/home/lc/cd/llm/datasets/EarthVLSet/EarthVQA/Train/images_png")
    parser.add_argument("--dataset_json", type=str, default="/data0/home/lc/cd/llm/datasets/EarthVLSet/EarthVQA/Train_QA.json")
    parser.add_argument("--output_jsonl", type=str, default="/data1/home/dataset_share/cd_data/llava-v1.5-7B/EarthVQA/final/attn_proj_mapping_64_project.jsonl")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_samples", type=int, default=5000)
    args = parser.parse_args()

    set_seed(42)
    evaluate(
        model_path=args.model_path,
        model_base=args.model_base,
        data_dir=args.data_dir,
        dataset_json=args.dataset_json,
        output_jsonl=args.output_jsonl,
        device=torch.device(args.device),
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
