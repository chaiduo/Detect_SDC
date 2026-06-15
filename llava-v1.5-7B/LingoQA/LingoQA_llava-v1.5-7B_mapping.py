import os
import sys
import re
import random

# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

current_dir = os.path.dirname(__file__)
grandparent_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(grandparent_dir)

from utils.profiler import Profiler

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


def set_model_seed(seed: int):
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
):
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

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

    # 这里按你当前跑通的 llava 环境，直接 decode output_ids
    pred = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return pred


def evaluate(
    model_path,
    val_file,
    data_dir,
    save_dir,
    device,
    max_new_tokens,
    model_base=None,
):
    tokenizer, model, image_processor, model_name = load_llava_model(model_path, model_base)
    model = model.to(device)

    prof = Profiler(model, proj_dim=64, seed=1234)
    prof.register()

    df = pd.read_parquet(val_file)

    os.makedirs(save_dir, exist_ok=True)

    sample_id = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        question = row["question"]
        images = row["images"]
        num_images = min(5, len(images))

        for i in range(num_images):
            img_path = images[i]
            image_path = os.path.join(data_dir, img_path)

            try:
                _ = generate_answer(
                    question=question,
                    image_path=image_path,
                    tokenizer=tokenizer,
                    model=model,
                    image_processor=image_processor,
                    model_name=model_name,
                    max_new_tokens=max_new_tokens,
                )

                prof.save_attn_proj_interlayer_jsonl(
                    os.path.join(save_dir, "attn_proj_interlayer.jsonl")
                )
            except Exception as e:
                print(f"[Warning] Failed on image: {image_path}")
                print(f"[Warning] Error: {e}")

            prof.reset(clear_stats=True)
            sample_id += 1

    prof.unregister()


def main():
    set_model_seed(42)

    device = torch.device("cuda:0")
    model_path = "liuhaotian/llava-v1.5-7b"
    model_base = None
    val_file = "/data0/home/lc/cd/predict_error/LingoQA-main/data/val/val.parquet"
    data_dir = "/data0/home/lc/cd/predict_error/LingoQA-main/data/val/"
    save_dir = "/data1/home/dataset_share/cd_data/llava-v1.5-7b/LingoQA"

    evaluate(
        model_path=model_path,
        model_base=model_base,
        val_file=val_file,
        data_dir=data_dir,
        save_dir=save_dir,
        device=device,
        max_new_tokens=50,
    )


if __name__ == "__main__":
    main()
