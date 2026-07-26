import os
import sys
import io
import json
import glob
import argparse
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

current_dir = os.path.dirname(__file__)
grandparent_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(grandparent_dir)

from utils.similarity_utils import SimilarityEvaluator


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_image_from_parquet_cell(image_cell):
    """
    将 parquet 中的 image 单元统一转成 PIL.Image.
    支持以下格式：
    1) {"bytes": b"...", "path": "..."}
    2) bytes
    3) PIL.Image.Image
    4) {"path": "..."} 仅路径
    """
    if image_cell is None:
        raise ValueError("image cell is None")

    if isinstance(image_cell, Image.Image):
        return image_cell.convert("RGB")

    if isinstance(image_cell, bytes):
        return Image.open(io.BytesIO(image_cell)).convert("RGB")

    if isinstance(image_cell, dict):
        img_bytes = image_cell.get("bytes", None)
        img_path = image_cell.get("path", None)

        if img_bytes is not None:
            return Image.open(io.BytesIO(img_bytes)).convert("RGB")

        if img_path is not None and os.path.exists(img_path):
            return Image.open(img_path).convert("RGB")

        raise ValueError(f"Unsupported image dict format: keys={list(image_cell.keys())}")

    raise TypeError(f"Unsupported image cell type: {type(image_cell)}")


def collect_parquet_files(parquet_path):
    """
    支持：
    - 单个 parquet 文件
    - 一个目录下所有 parquet 文件
    """
    parquet_path = str(parquet_path)

    if os.path.isfile(parquet_path):
        return [parquet_path]

    if os.path.isdir(parquet_path):
        files = sorted(glob.glob(os.path.join(parquet_path, "*.parquet")))
        return files

    raise FileNotFoundError(f"parquet path not found: {parquet_path}")


def build_messages(question, pil_image):
    """
    为 Qwen2.5-VL 构造消息。
    这里直接传 PIL.Image，不走 file:// 路径。
    """
    messages = [
        {
            "role": "system",
            "content": "The answer must be limited to 30 words."
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": question},
            ],
        }
    ]
    return messages


def infer_one_sample(model, processor, device, question, pil_image, max_new_tokens=50):
    messages = build_messages(question, pil_image)

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=[pil_image],
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
    pred = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
    return pred


def evaluate(
    model_path,
    parquet_path,
    golden_json,
    device,
    similarity_evaluator: SimilarityEvaluator,
    max_samples=5000,
):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16
    ).eval().to(device)

    processor = AutoProcessor.from_pretrained(
        model_path,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28
    )

    parquet_files = collect_parquet_files(parquet_path)
    if len(parquet_files) == 0:
        raise ValueError(f"No parquet files found in: {parquet_path}")

    print(f"[Info] Found {len(parquet_files)} parquet file(s).")

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

            pred = infer_one_sample(
                model=model,
                processor=processor,
                device=device,
                question=question,
                pil_image=pil_image,
                max_new_tokens=50
            )

            sample_data = {
                "id": sample_id,
                "source_file": str(val_file),
                "row_idx": int(row_idx),
                "question": question,
                "gt_answer": gt_answer,
                "pre_answer": pred,
                "scores": similarity_evaluator.score(gt_answer, pred),
            }

            all_samples.append(sample_data)
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
    parser.add_argument("--model_path", type=str, default="/data01/cd_workspace/llm/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--parquet_path", type=str, default="/data01/cd_workspace/llm/VQAv2")
    parser.add_argument("--golden_json", type=str, default="./json/Golden_VQAv2_Qwen2.5-VL-7B_30_new.json")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_samples", type=int, default=5000)
    args = parser.parse_args()

    set_seed(42)
    device = torch.device(args.device)
    se = SimilarityEvaluator()

    evaluate(
        model_path=args.model_path,
        parquet_path=args.parquet_path,
        golden_json=args.golden_json,
        device=device,
        similarity_evaluator=se,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
