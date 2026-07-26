import os
import sys
import argparse
import json
import random
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Any, Dict
from collections import defaultdict, Counter
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

current_dir = os.path.dirname(__file__)
grandparent_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(grandparent_dir)

from utils.profiler import Profiler
from utils.fault_injector import FaultInjector
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
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(to_jsonable(data), ensure_ascii=False) + "\n")


def load_clean_answers(golden_json):
    with open(golden_json, "r", encoding="utf-8") as f:
        samples = json.load(f)
    return {int(item["id"]): item.get("pre_answer", "") for item in samples}



def build_result_record(
    idx,
    before_score,
    after_score,
    injector,
    img_path,
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
        "is_sdc": int(clean_pred.strip() != pred.strip()),
        "fault": to_jsonable(getattr(injector, "fault_info", None)),
        "image_path": img_path,
        "question": question,
        "gt_answer": gt_answer,
        "clean_answer": clean_pred,
        "pred_answer": pred,
        "mean_std_cos": mean_std_cos
    }
    return result


def evaluate(
    model_path,
    data_dir,
    dataset_json,
    output_jsonl,
    device,
    clean_answers,
    mapping_model_path: str = "best_model.pt",
    run_time: int = 0,
    inject_fault: bool = True,
    max_new_tokens: int = 100,
    max_samples: int = 5000,
):
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path,torch_dtype=torch.bfloat16).eval().to(device)
    processor = AutoProcessor.from_pretrained(model_path,min_pixels=256 * 28 * 28,max_pixels=1280 * 28 * 28)

    print(f"Loading dataset from {dataset_json}...")
    with open(dataset_json, 'r') as f:
        raw_data = json.load(f)

    df = pd.DataFrame(list(raw_data.items()), columns=['image_filename', 'qa_list'])

    injector = FaultInjector(model, mode="activation")

    prof = Profiler(model, proj_dim=64, proj_method="project",seed=42)
    prof.register()

    mapping_model = LayerAwareResidualMLP(
        x_dim=64,
        num_layers=28,
        layer_emb_dim=16,
        hidden_dim=64,
        num_blocks=4,
        dropout=0.1,
    ).to(device)

    state_dict = torch.load(mapping_model_path, map_location=device)
    mapping_model.load_state_dict(state_dict)
    mapping_model.eval()

    sample_id=0
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
            prof.reset(clear_stats=True)
            injector.reset()
            injector.register_step_hooks()
            if inject_fault is True:
                injector.set_num_bits(2)
                injector.inject()

            question = qa_pair.get('Question', '')
            gt_answer = qa_pair.get('Answer', '')
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

            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False
                )
            
            trimmed = [o[len(inp):] for inp, o in zip(inputs.input_ids, out_ids)]
            pred = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
            before_score = 0
            after_score = 0
            clean_pred = clean_answers.get(sample_id, "")

            prof.finalize()
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
                img_path=image_path,
                question=question,
                gt_answer=gt_answer,
                clean_pred=clean_pred,
                mean_std_cos=sample_result,
                pred=pred,
            )
            # 写入jsonl
            if inject_fault is False or run_time < 1:
                append_jsonl(output_jsonl, result)
            else:
                if clean_pred.strip() != pred.strip():
                    append_jsonl(output_jsonl, result)
                   
            injector.unregister_hooks()
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
    parser.add_argument("--output_jsonl", type=str, default="./json/detect_EarthVQA_Qwen_with_sem_project.jsonl")
    parser.add_argument("--golden_json", type=str, default="./json/Golden_EarthVQA_Qwen2.5-VL-7B_30_CA.json")
    parser.add_argument("--mapping_model", type=str, default="./model/best_mapping_model.pt")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_samples", type=int, default=5000, help="Maximum number of samples to evaluate")
    args = parser.parse_args()

    set_model_seed(42)

    device = torch.device(args.device)
    clean_answers = load_clean_answers(args.golden_json)

    evaluate(
        model_path=args.model_path,
        data_dir=args.data_dir,
        dataset_json=args.dataset_json,
        output_jsonl=args.output_jsonl,
        device=device,
        clean_answers=clean_answers,
        mapping_model_path=args.mapping_model,
        inject_fault=False,
        max_new_tokens=50,
        max_samples=args.max_samples,
    )
    for run in range(16):
        random.seed(42 + run)
        evaluate(
            model_path=args.model_path,
            data_dir=args.data_dir,
            dataset_json=args.dataset_json,
            output_jsonl=args.output_jsonl,
            device=device,
            clean_answers=clean_answers,
            mapping_model_path=args.mapping_model,
            run_time=run,
            inject_fault=True,
            max_new_tokens=50,
            max_samples=args.max_samples,
        )
if __name__ == "__main__":
    main()
