import os
import json
import io
import pandas as pd
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from tqdm import tqdm


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
        return None

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

        return None

    return None

# =========================
# 配置区域
# =========================
PARQUET_FILE = "/data0/home/lc/cd/predict_error/Detect_SDC/Qwen2.5-VL-7B/VQAv2/train_data/validation-00000-of-00001-6c7328ff6c84284c.parquet"
MODEL_PATH = "/data1/home/dataset_share/wsh_data/data/qwen/Qwen2___5-VL-7B-Instruct"
OUTPUT_CSV = "/data0/home/lc/cd/predict_error/Detect_SDC/Qwen2.5-VL-7B/VQAv2/train_data/llm_inference_results.csv"
BATCH_SIZE = 1
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.1
TOP_P = 0.9

CHOICE_LETTERS = ["A", "B", "C", "D", "E", "F"]


def load_model():
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16
    ).eval()
    return processor, model


def build_prompt_direct(row):
    question = row["question"]
    choices = row["choices"]
    choice_str = "\n".join([f"{CHOICE_LETTERS[i]}. {c}" for i, c in enumerate(choices)])
    prompt = f"""Question: {question}

Choices:
{choice_str}

Please answer with only the letter (A, B, C, D, etc.).
Answer:"""
    return prompt


def build_prompt_cot(row):
    question = row["question"]
    choices = row["choices"]
    lecture = row.get("lecture", "")
    choice_str = "\n".join([f"{CHOICE_LETTERS[i]}. {c}" for i, c in enumerate(choices)])

    prompt = f"""Question: {question}

Choices:
{choice_str}
"""
    if lecture and str(lecture).strip():
        prompt += f"""
Reference knowledge:
{lecture}
"""
    prompt += """
Please think step by step, then give your final answer in the format: "The answer is X" where X is the letter.
"""
    return prompt


def extract_answer_direct(text):
    text = text.strip()
    for letter in CHOICE_LETTERS:
        if text.upper().startswith(letter):
            return letter
    for letter in CHOICE_LETTERS:
        if letter in text:
            return letter
    return None


def extract_answer_cot(text):
    import re
    match = re.search(r"[Tt]he answer is\s*([A-F])", text)
    if match:
        return match.group(1)
    return extract_answer_direct(text)


def generate(processor, model, prompt, image=None):
    messages = [{"role": "user", "content": prompt}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, images=image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return processor.decode(generated, skip_special_tokens=True)


def main():
    df = pd.read_parquet(PARQUET_FILE)
    print(f"Loaded {len(df)} samples")

    processor, model = load_model()
    print("Model loaded")

    results = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Inference"):
        # 检查是否有图片字段，没有图片的跳过
        if "image" not in df.columns or pd.isna(row.get("image")):
            continue

        # 加载图片
        image_data = row["image"]
        pil_image = load_image_from_parquet_cell(image_data)
        if pil_image is None:
            continue

        correct_letter = CHOICE_LETTERS[int(row["answer"])]
        choices = row["choices"]
        # 将 ndarray 转换为列表以便 JSON 序列化
        if hasattr(choices, 'tolist'):
            choices = choices.tolist()

        # 直接回答
        prompt_direct = build_prompt_direct(row)
        response_direct = generate(processor, model, prompt_direct, image=pil_image)
        pred_direct = extract_answer_direct(response_direct)

        # CoT 推理
        prompt_cot = build_prompt_cot(row)
        response_cot = generate(processor, model, prompt_cot, image=pil_image)
        pred_cot = extract_answer_cot(response_cot)

        results.append({
            "index": idx,
            "question": row["question"],
            "choices": json.dumps(choices, ensure_ascii=False),
            "correct_answer": correct_letter,
            "pred_direct": pred_direct,
            "correct_direct": pred_direct == correct_letter if pred_direct else False,
            "response_direct": response_direct,
            "pred_cot": pred_cot,
            "correct_cot": pred_cot == correct_letter if pred_cot else False,
            "response_cot": response_cot,
            "task": row["task"],
            "grade": row["grade"],
            "subject": row["subject"],
            "topic": row["topic"],
        })

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    acc_direct = result_df["correct_direct"].mean()
    acc_cot = result_df["correct_cot"].mean()

    print("=" * 60)
    print(f"Direct answer accuracy: {acc_direct:.4f}")
    print(f"CoT answer accuracy:    {acc_cot:.4f}")
    print(f"Results saved to: {OUTPUT_CSV}")
    print("=" * 60)


if __name__ == "__main__":
    main()
