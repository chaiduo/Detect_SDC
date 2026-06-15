import pandas as pd
import numpy as np
import torch
import os,json,sys
os.environ["CUDA_VISIBLE_DEVICES"] = "4"  
from tqdm import tqdm
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from fault_injector import FaultInjector 
from score_utils import ScoreEvaluator

score_evaluator = ScoreEvaluator(json_path="LingoQA-InternVL3-8B.json")

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

def load_json_dataset(json_path):
    data = []
    if json_path.endswith(".jsonl"):
        with open(json_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
    elif json_path.endswith(".json"):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        raise ValueError("Only .json or .jsonl is supported!")

    print(f"✅ Loaded {len(data)} samples from {json_path}")
    return data

def evaluate(model_path, json_dataset_path, output_jsonl, device):
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
        trust_remote_code=True,
    ).eval().to(device)
    injector = FaultInjector(model, enable_injection=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    dataset = load_json_dataset(json_dataset_path)

    for sample in tqdm(dataset, desc="Faulting"):
        id = sample["id"]
        img_path = sample["image"]
        question = sample["question"]
        gt_answer = sample["gt_answer"]
        fault_idx = sample["fault"]["idx"]
        module_name = sample["fault"]["module"]
        fault_bit_pos = sample["fault"]["bit_pos"]

        injector.reset_and_select_target(module_name=module_name)
        injector.set_inject_info(bit_pos=fault_bit_pos, idx=fault_idx)

        pixel_values = load_image(img_path, max_num=12).to(torch.bfloat16).to(device)
        generation_config = dict(max_new_tokens=512, do_sample=False, pad_token_id=tokenizer.eos_token_id)

        question = '<image>\n'+ question
        prediction = model.chat(tokenizer, pixel_values, question, generation_config)

        before_score, after_score, clear_pred = score_evaluator.get_fault_scores(question, gt_answer, prediction, id)
        correct = 1 if after_score > 0.5 else 0 

        result = {
            "id": id,
            "image": img_path,
            "question": question,
            "gt_answer": gt_answer,
            "clean_answer": clear_pred,
            "pred_answer": prediction,
            "fault": injector.fault_info,
            "correct": correct,
            "before_score": before_score,
            "after_score": after_score,
            "dtel_score": after_score - before_score
        }

        with open(output_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


def main():
    device = torch.device("cuda:0")
    model_path = "/data0/home/lc/cd/llm/InternVL/InternVL3-8B"
    json_dataset_path = "Filtered_Fault_LingoQA-InternVL3-8B.json"
    output_jsonl = "ReEval_Fault_InternVL3-8B.jsonl"

    evaluate(
        model_path=model_path,
        json_dataset_path=json_dataset_path,
        output_jsonl=output_jsonl,
        device=device
    )

if __name__ == "__main__":
    main()
