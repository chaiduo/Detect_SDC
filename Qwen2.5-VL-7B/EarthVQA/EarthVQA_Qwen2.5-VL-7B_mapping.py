import os
import sys
import argparse
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

current_dir = os.path.dirname(__file__)
grandparent_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(grandparent_dir)

from utils.profiler import Profiler

def set_model_seed(seed: int):
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
    output_jsonl,
    device,
    max_samples: int = 5000,
):
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path,torch_dtype=torch.bfloat16).eval().to(device)
    processor = AutoProcessor.from_pretrained(model_path,min_pixels=256 * 28 * 28,max_pixels=1280 * 28 * 28)

    print(f"Loading dataset from {dataset_json}...")
    with open(dataset_json, 'r') as f:
        raw_data = json.load(f)
    
    # 转换为DataFrame，其中 index 是图片名， value 是问答列表
    df = pd.DataFrame(list(raw_data.items()), columns=['image_filename', 'qa_list'])

    prof = Profiler(model, proj_dim=64, proj_method="project", seed=42)
    prof.register()

    sample_id = 0
    for _, row in tqdm(df.iterrows(), total=len(df)):
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
            inputs = processor(text=[text],images=image_inputs,videos=video_inputs,return_tensors="pt").to(device)
            out_ids = model.generate(**inputs, max_new_tokens=50, do_sample=False)
            trimmed = [o[len(inp):] for inp, o in zip(inputs.input_ids, out_ids)]
            pred = processor.batch_decode(trimmed, skip_special_tokens=True)[0]

            prof.finalize()
            prof.save_attn_proj_interlayer_jsonl(output_jsonl)
            prof.reset(clear_stats=True)
            sample_id += 1
            if sample_id >= max_samples:
                break
        if sample_id >= max_samples:
            break

    prof.unregister()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/data01/cd_workspace/llm/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--data_dir", type=str, default="/data01/cd_workspace/llm/EarthVQA/Train/images_png")
    parser.add_argument("--dataset_json", type=str, default="/data01/cd_workspace/llm/EarthVQA/Train_QA.json")
    parser.add_argument("--output_jsonl", type=str, default="./json/attn_proj_mapping_64_project.jsonl")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_samples", type=int, default=5000, help="Maximum number of samples to evaluate")
    args = parser.parse_args()

    set_model_seed(42)
    device = torch.device(args.device)
    evaluate(
        model_path=args.model_path,
        data_dir=args.data_dir,
        dataset_json=args.dataset_json,
        output_jsonl=args.output_jsonl,
        device=device,
        max_samples=args.max_samples,
    )
if __name__ == "__main__":
    main()
