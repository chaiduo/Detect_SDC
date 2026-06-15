import os, sys
import argparse
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import random
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

current_dir = os.path.dirname(__file__)
grandparent_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(grandparent_dir)

from utils.similarity_utils import SimilarityEvaluator


def set_seed(seed: int):
    """设置随机种子以确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate(
    model_path, 
    data_dir, 
    dataset_json, 
    golden_json,
    device,
    similarity_evaluator: SimilarityEvaluator,
    max_samples: int = 5000,
):
    print("Loading model and processor...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path,torch_dtype=torch.bfloat16,device_map=device).eval() 
    processor = AutoProcessor.from_pretrained(model_path, min_pixels=256*28*28, max_pixels=1280*28*28)

    print(f"Loading dataset from {dataset_json}...")
    with open(dataset_json, 'r') as f:
        raw_data = json.load(f)

    # 转换为DataFrame，其中 index 是图片名， value 是问答列表
    df = pd.DataFrame(list(raw_data.items()), columns=['image_filename', 'qa_list'])

    sample_id = 0
    all_samples = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        image_filename = row['image_filename']
        qa_list = row['qa_list']
        image_path = os.path.join(data_dir, image_filename)

        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} does not exist. Skipping.")
            continue
        
        cur_cnt = 0
        # 遍历当前图片的所有问答对
        for qa_pair in tqdm(qa_list, leave=False, desc=f"QAs for {image_filename}"):
            question = qa_pair.get('Question', '')
            answer = qa_pair.get('Answer', '')
            question_type = qa_pair.get('Type', '')
            if question_type != "Comprehensive Analysis":
                continue
            if cur_cnt >= 2:
                continue
            cur_cnt += 1
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
            generated_ids = model.generate(**inputs, max_new_tokens=50, do_sample=False)
            trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            pred = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
            
            result = {
                "id": sample_id,
                "image_path": image_filename,
                "type": question_type,
                "question": question,
                "gt_answer": answer,
                "pre_answer": pred,
                "scores": similarity_evaluator.score(answer, pred),
            }
            all_samples.append(result)

            sample_id += 1
            if sample_id >= max_samples:
                break
        if sample_id >= max_samples:
            break

    with open(golden_json, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)

    print(f"Results saved successfully to: {golden_json}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/data1/home/dataset_share/wsh_data/data/qwen/Qwen2___5-VL-7B-Instruct")
    parser.add_argument("--data_dir", type=str, default="/data0/home/lc/cd/llm/datasets/EarthVLSet/EarthVQA/Train/images_png")
    parser.add_argument("--dataset_json", type=str, default="/data0/home/lc/cd/llm/datasets/EarthVLSet/EarthVQA/Train_QA.json")
    parser.add_argument("--golden_json", type=str, default="./json/Golden_EarthVQA_Qwen2.5-VL-7B_30_CA.json")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_samples", type=int, default=5000)
    args = parser.parse_args()

    set_seed(42)
    device = torch.device(args.device)
    se = SimilarityEvaluator()
    evaluate(
        model_path=args.model_path,
        data_dir=args.data_dir,
        dataset_json=args.dataset_json,
        golden_json=args.golden_json,
        device=device,
        similarity_evaluator=se,
        max_samples=args.max_samples
    )

if __name__ == "__main__":
    main()