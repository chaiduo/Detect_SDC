import glob
import io
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from llava.utils import disable_torch_init


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_llava_model(model_path: str, device, model_base: str = None):
    disable_torch_init()
    if model_base == "":
        model_base = None

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path,
        model_base,
        model_name,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval().to(device)
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
    return conv.get_prompt()


def generate_answer_from_pil(
    question: str,
    pil_image,
    tokenizer,
    model,
    image_processor,
    model_name: str,
    max_new_tokens: int = 50,
):
    image = pil_image.convert("RGB")
    qs = question.strip() + "\nThe answer must be limited to 30 words."
    prompt = build_prompt(qs, model, model_name)

    images_tensor = process_images([image], image_processor, model.config).to(
        model.device,
        dtype=torch.float16,
    )
    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=[image.size],
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


def generate_answer_from_path(
    question: str,
    image_path: str,
    tokenizer,
    model,
    image_processor,
    model_name: str,
    max_new_tokens: int = 50,
):
    image = Image.open(image_path).convert("RGB")
    return generate_answer_from_pil(
        question,
        image,
        tokenizer,
        model,
        image_processor,
        model_name,
        max_new_tokens=max_new_tokens,
    )


def load_image_from_parquet_cell(image_cell):
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
    parquet_path = str(parquet_path)
    if os.path.isfile(parquet_path):
        return [parquet_path]
    if os.path.isdir(parquet_path):
        return sorted(glob.glob(os.path.join(parquet_path, "*.parquet")))
    raise FileNotFoundError(f"parquet path not found: {parquet_path}")


def to_jsonable(obj):
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, torch.Tensor):
        if obj.dim() == 0:
            return obj.item()
        return obj.detach().cpu().tolist()
    if isinstance(obj, (dict, defaultdict, Counter)):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def append_jsonl(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(to_jsonable(data), ensure_ascii=False) + "\n")
