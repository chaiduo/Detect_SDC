import os
import sys
import json
import re
import random
import traceback

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch

current_dir = os.path.dirname(__file__)
grandparent_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(grandparent_dir)

from utils.Lingo_judge import ScoreEvaluator

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_llava_model(model_path: str, model_base: str = None):
    disable_torch_init()

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        model_base,
        model_name
    )

    model.eval()
    return tokenizer, model, image_processor, model_name


def build_prompt(question: str, model, model_name: str):
    qs = question
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN

    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    return prompt


def generate_answer(
    question: str,
    image_path: str,
    tokenizer,
    model,
    image_processor,
    model_name: str,
    max_new_tokens: int = 50,
    temperature: float = 0.0,
    top_p=None,
    num_beams: int = 1,
):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")

    qs = question.strip() + "\nThe answer must be limited to 30 words."
    prompt = build_prompt(qs, model, model_name)

    images = [image]
    image_sizes = [x.size for x in images]

    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt"
    ).unsqueeze(0).to(model.device)

    gen_kwargs = {
        "images": images_tensor,
        "image_sizes": image_sizes,
        "num_beams": num_beams,
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
    }

    if temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        if top_p is not None:
            gen_kwargs["top_p"] = top_p
    else:
        gen_kwargs["do_sample"] = False

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            **gen_kwargs
        )

    # 注意：当前你的 llava 环境里 output_ids 直接 decode 即可
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    return outputs


def evaluate(
    model_path: str,
    val_file: str,
    data_dir: str,
    golden_json: str,
    score_evaluator: ScoreEvaluator,
    model_base: str = None,
):
    print(f"[Info] Loading model from: {model_path}")
    tokenizer, model, image_processor, model_name = load_llava_model(
        model_path=model_path,
        model_base=model_base
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
                pred = generate_answer(
                    question=question,
                    image_path=image_path,
                    tokenizer=tokenizer,
                    model=model,
                    image_processor=image_processor,
                    model_name=model_name,
                    max_new_tokens=50,
                    temperature=0.0,
                    top_p=None,
                    num_beams=1,
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
    set_seed(42)

    model_path = "liuhaotian/llava-v1.5-7b"
    model_base = None
    val_file = "/data0/home/lc/cd/predict_error/LingoQA-main/data/val/val.parquet"
    data_dir = "/data0/home/lc/cd/predict_error/LingoQA-main/data/val/"
    golden_json = "./json/Golden_LingoQA_llava-v1.5-7B.json"

    score_evaluator = ScoreEvaluator()

    evaluate(
        model_path=model_path,
        model_base=model_base,
        val_file=val_file,
        data_dir=data_dir,
        golden_json=golden_json,
        score_evaluator=score_evaluator,
    )


if __name__ == "__main__":
    main()
