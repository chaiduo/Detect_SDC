import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import sys
import io
import json
import glob
import random
from pathlib import Path
from collections import defaultdict, Counter

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

current_dir = os.path.dirname(__file__)
grandparent_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(grandparent_dir)

from utils.profiler import Profiler
from utils.Lingo_judge import ScoreEvaluator
from utils.fault_injector import FaultInjector
from utils.similarity_utils import SimilarityEvaluator
from train_mapping_model import LayerAwareResidualMLP


def set_model_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_jsonable(obj):
    """
    递归地把对象转换成 JSON 可序列化的 Python 原生类型
    """
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
    """
    把一条数据追加写入 jsonl 文件
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(to_jsonable(data), ensure_ascii=False) + "\n")


def build_result_record(
    idx,
    before_score,
    after_score,
    injector,
    source_file,
    row_idx,
    question,
    gt_answer,
    clean_pred,
    mean_std_cos,
    pred,
):
    result = {
        "id": idx,
        "before_score": float(before_score),
        "after_score": float(after_score),
        "dtel_score": float(after_score - before_score),
        "is_sdc": int(before_score != after_score),
        "fault": injector.fault_info,
        "source_file": source_file,
        "row_idx": int(row_idx),
        "question": question,
        "gt_answer": gt_answer,
        "clean_answer": clean_pred,
        "pred_answer": pred,
        "mean_std_cos": mean_std_cos,
    }
    return result


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
    为 Qwen2.5-VL 构造消息，直接传 PIL.Image
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


def evaluate(
    model_path,
    parquet_path,
    output_jsonl,
    device,
    similarity_evaluator: SimilarityEvaluator,
    run_time: int = 0,
    inject_fault: bool = True,
    max_new_tokens: int = 100,
    max_samples: int = 5000,
):
    parquet_files = collect_parquet_files(parquet_path)
    if len(parquet_files) == 0:
        raise ValueError(f"No parquet files found in: {parquet_path}")

    print(f"[Info] Found {len(parquet_files)} parquet file(s).")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16
    ).eval().to(device)

    processor = AutoProcessor.from_pretrained(
        model_path,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28
    )

    injector = FaultInjector(model, mode="activation")

    prof = Profiler(model, proj_dim=64, proj_method="project", seed=42)
    prof.register()

    mapping_model = LayerAwareResidualMLP(
        x_dim=64,
        num_layers=28,
        layer_emb_dim=16,
        hidden_dim=1024,
        num_blocks=8,
        dropout=0.1,
    ).to(device)

    state_dict = torch.load("best_model-project.pt", map_location=device)
    mapping_model.load_state_dict(state_dict)
    mapping_model.eval()

    sample_id = 0
    for file_idx, val_file in enumerate(parquet_files):
        print(f"[Info] Loading parquet [{file_idx + 1}/{len(parquet_files)}]: {val_file}")
        df = pd.read_parquet(val_file)

        required_cols = {"question", "multiple_choice_answer", "image"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns in {val_file}: {missing_cols}")

        for row_idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Run={run_time} | {Path(val_file).name}"):
            question = str(row["question"]).strip()
            gt_answer = str(row["multiple_choice_answer"]).strip()

            try:
                pil_image = load_image_from_parquet_cell(row["image"])
            except Exception as e:
                print(f"[Warning] Failed to load image at row {row_idx} in {val_file}: {e}")
                continue

            prof.reset(clear_stats=True)
            injector.reset()
            injector.register_step_hooks()

            if inject_fault:
                injector.set_num_bits(2)
                injector.inject()

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

            trimmed = [o[len(inp):] for inp, o in zip(inputs.input_ids, out_ids)]
            pred = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

            prof.finalize()
            before_score, after_score, clean_pred = similarity_evaluator.get_fault_scores(pred, sample_id)

            # sample_result = prof.get_attn_proj_model_diff_vector_result(
            #     predictor_model=mapping_model,
            #     device=device,
            #     include_pred_target_vectors=False,
            # )
            sample_result = prof.get_attn_proj_model_compare_result(
                predictor_model=mapping_model,
                device=device,
                include_vectors=False,
            )
            result = build_result_record(
                idx=sample_id,
                before_score=before_score,
                after_score=after_score,
                injector=injector,
                source_file=str(val_file),
                row_idx=row_idx,
                question=question,
                gt_answer=gt_answer,
                clean_pred=clean_pred,
                mean_std_cos=sample_result,
                pred=pred,
            )

            if (not inject_fault) or (run_time < 1):
                append_jsonl(output_jsonl, result)
            else:
                if before_score != after_score:
                    append_jsonl(output_jsonl, result)

            injector.unregister_hooks()
            sample_id += 1

            if sample_id >= max_samples:
                break

        if sample_id >= max_samples:
            break

    prof.unregister()


def main():
    set_model_seed(42)

    device = torch.device("cuda:0")
    model_path = "/data1/home/dataset_share/wsh_data/data/qwen/Qwen2___5-VL-7B-Instruct"
    parquet_path = "/data0/home/lc/cd/predict_error/Detect_SDC/Qwen2.5-VL-7B/VQAv2/train_data"
    output_jsonl = "/data1/home/dataset_share/cd_data/Qwen2.5-VL-7B/VQAv2/final/detect_VQAv2_Qwen_with_sem_project.jsonl"

    se = SimilarityEvaluator(json_path="./json/Golden_VQAv2_Qwen2.5-VL-7B_30_new.json")
    # clean run
    # evaluate(
    #     model_path=model_path,
    #     parquet_path=parquet_path,
    #     output_jsonl=output_jsonl,
    #     device=device,
    #     similarity_evaluator=se,
    #     inject_fault=False,
    #     max_new_tokens=50,
    #     max_samples=5000,
    # )

    # fault runs
    for run in range(2):
        run += 7
        random.seed(42 + run)
        evaluate(
            model_path=model_path,
            parquet_path=parquet_path,
            output_jsonl=output_jsonl,
            device=device,
            similarity_evaluator=se,
            run_time=run,
            inject_fault=True,
            max_new_tokens=50,
            max_samples=5000,
        )


if __name__ == "__main__":
    main()
