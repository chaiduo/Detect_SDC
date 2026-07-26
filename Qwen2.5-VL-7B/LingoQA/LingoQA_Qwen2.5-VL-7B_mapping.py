import os
import sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

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
    val_file,
    data_dir,
    output_jsonl,
    device,
    max_new_tokens,
):
    
    output_dir = os.path.dirname(output_jsonl)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path,torch_dtype=torch.bfloat16).eval().to(device)
    processor = AutoProcessor.from_pretrained(model_path,min_pixels=256 * 28 * 28,max_pixels=1280 * 28 * 28)
    processor = AutoProcessor.from_pretrained(model_path,min_pixels=256 * 28 * 28,max_pixels=1280 * 28 * 28)

    prof = Profiler(model, proj_dim=64, seed=42)
    prof.register()

    sample_id = 0
    df = pd.read_parquet(val_file)
    for _, row in tqdm(df.iterrows(), total=len(df)):
        question = row["question"]

        for i in range(5):
            img_path = row["images"][i]
            image_path = os.path.join(data_dir, img_path)
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

            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False
                )
            
            trimmed = [o[len(inp):] for inp, o in zip(inputs.input_ids, out_ids)]
            pred = processor.batch_decode(trimmed, skip_special_tokens=True)[0]

            prof.save_attn_proj_interlayer_jsonl(output_jsonl)
            prof.reset(clear_stats=True)
            sample_id += 1
    prof.unregister()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/data01/cd_workspace/llm/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--val_file", type=str, default="/data01/cd_workspace/llm/LingoQA/val.parquet")
    parser.add_argument("--data_dir", type=str, default="/data01/cd_workspace/llm/LingoQA/")
    parser.add_argument("--output_jsonl", type=str, default="./json/attn_proj_interlayer.jsonl")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    args = parser.parse_args()

    set_model_seed(42)

    device = torch.device(args.device)
    evaluate(
        model_path=args.model_path,
        val_file=args.val_file,
        data_dir=args.data_dir,
        output_jsonl=args.output_jsonl,
        device=device,
        max_new_tokens=args.max_new_tokens,
    )
if __name__ == "__main__":
    main()
