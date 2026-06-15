import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

current_dir = os.path.dirname(__file__)
grandparent_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(grandparent_dir)

from utils.Lingo_judge import ScoreEvaluator

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate(
    model_path, 
    val_file, 
    data_dir, 
    golden_json,
    device,
    score_evaluator:ScoreEvaluator
):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16).eval().to(device)
    processor = AutoProcessor.from_pretrained(model_path, min_pixels=256*28*28, max_pixels=1280*28*28)
    df = pd.read_parquet(val_file)

    sample_id = 0
    all_samples = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Profiling"):
        question_id = row["question_id"]
        segment_id = row["segment_id"]
        question = row["question"]
        gt_answer = row["answer"]

        for i in range(5):
            img_path = row["images"][i]
            image_path = os.path.join(data_dir, row["images"][i])

            messages = [
                {
                    "role": "system",
                    "content": "The answer must be limited to 30 words."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "file://" + image_path},
                        {"type": "text", "text": question},
                    ],
                }
            ]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt").to(device)
            out_ids = model.generate(**inputs, max_new_tokens=50, do_sample=False)
            trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
            pred = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
            
            # 写入golden数据
            score = score_evaluator.get_score(question, gt_answer, pred)
            sample_data = {
                "id": sample_id,
                "question_id": question_id,
                "segment_id": segment_id,
                "image_path": img_path,
                "question": question,
                "gt_answer": gt_answer,
                "pre_answer": pred,
                "score": score
            }
            all_samples.append(sample_data)
            sample_id += 1

    with open(golden_json, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)

    print(f"Results saved successfully to: {golden_json}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/data1/home/dataset_share/wsh_data/data/qwen/Qwen2___5-VL-7B-Instruct")
    parser.add_argument("--val_file", type=str, default="/data0/home/lc/cd/predict_error/LingoQA-main/data/val/val.parquet")
    parser.add_argument("--data_dir", type=str, default="/data0/home/lc/cd/predict_error/LingoQA-main/data/val/")
    parser.add_argument("--golden_json", type=str, default="./json/Golden_LingoQA_Qwen2.5-VL-7B.json")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device(args.device)
    score_evaluator = ScoreEvaluator()
    evaluate(
        model_path=args.model_path, 
        val_file=args.val_file, 
        data_dir=args.data_dir,  
        golden_json=args.golden_json,
        device=device,
        score_evaluator=score_evaluator
    )

if __name__ == "__main__":
    main()