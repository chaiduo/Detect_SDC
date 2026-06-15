import pandas as pd
import numpy as np
import torch
import os,json,sys
from tqdm import tqdm
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from score_utils import ScoreEvaluator
score_evaluator = ScoreEvaluator(json_path="./json/LingoQA-InternVL3-8B.json")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def evaluate_on_vqa(model_path, val_file, data_dir, output_json, device):
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
        trust_remote_code=True,
    ).eval().to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    df = pd.read_parquet(val_file)

    idx = 0
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        question_id = row["question_id"]
        segment_id = row["segment_id"]
        gt_answer = row["answer"]

        for i in range(5):
            img_path = row["images"][i]
            img_path = os.path.join(data_dir, img_path)

            pixel_values = load_image(img_path, max_num=12).to(torch.bfloat16).to(device)
            generation_config = dict(max_new_tokens=512, do_sample=False, pad_token_id=tokenizer.eos_token_id)

            question = '<image>\n'+ row["question"]
            response = model.chat(tokenizer, pixel_values, question, generation_config)

            score = score_evaluator.get_score(question, gt_answer, response)
            correct = 1 if score > 0.5 else 0 
            results.append({
                "id": idx,
                "image": img_path,
                "question_id": question_id,
                "segment_id": segment_id,
                "question": question,
                "gt_answer": gt_answer,
                "pre_answer": response,
                "correct": correct,
                "score": score
            })
            idx += 1

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

def main():
    
    device = torch.device("cuda:4")
    model_path = "/data0/home/lc/cd/llm/InternVL/InternVL3-8B"
    val_file = "/data0/home/lc/cd/predict_significant_error/Tasks/LingoQA-main/data/val/val.parquet"
    data_dir = "/data0/home/lc/cd/predict_significant_error/Tasks/LingoQA-main/data/val/"
    output_json = "LingoQA-InternVL3-8B.json"

    # ==== 运行评估 ====
    evaluate_on_vqa(
        model_path=model_path,
        val_file=val_file,
        data_dir=data_dir,
        output_json=output_json,
        device=device
    )

if __name__ == "__main__":
    main()
