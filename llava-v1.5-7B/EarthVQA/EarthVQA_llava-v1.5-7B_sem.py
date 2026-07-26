import argparse
import json
import os
import random
import sys

import pandas as pd
import torch
from tqdm import tqdm

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
grandparent_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.extend([parent_dir, grandparent_dir])

from llava_common import append_jsonl, generate_answer_from_path, load_llava_model, set_seed, to_jsonable
from train_mapping_model import LayerAwareResidualMLP
from utils.fault_injector import FaultInjector
from utils.profiler import Profiler


def load_clean_answers(golden_json):
    with open(golden_json, "r", encoding="utf-8") as f:
        samples = json.load(f)
    return {int(item["id"]): item.get("pre_answer", "") for item in samples}


def build_result_record(idx, before_score, after_score, injector, img_path, question, gt_answer, clean_pred, mean_std_cos, pred):
    return {
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
        "mean_std_cos": mean_std_cos,
    }


def evaluate(
    model_path,
    data_dir,
    dataset_json,
    output_jsonl,
    device,
    mapping_model_path: str,
    clean_answers,
    model_base: str = None,
    run_time: int = 0,
    inject_fault: bool = True,
    max_new_tokens: int = 50,
    max_samples: int = 5000,
):
    tokenizer, model, image_processor, model_name = load_llava_model(model_path, device, model_base)

    with open(dataset_json, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    df = pd.DataFrame(list(raw_data.items()), columns=["image_filename", "qa_list"])

    injector = FaultInjector(model, mode="activation")
    prof = Profiler(model, proj_dim=64, proj_method="project", seed=42)
    prof.register()

    mapping_model = LayerAwareResidualMLP(num_layers=32).to(device)
    mapping_model.load_state_dict(torch.load(mapping_model_path, map_location=device))
    mapping_model.eval()

    sample_id = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Run={run_time}"):
        image_filename = row["image_filename"]
        image_path = os.path.join(data_dir, image_filename)
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} does not exist. Skipping.")
            continue

        cur_cnt = 0
        for qa_pair in tqdm(row["qa_list"], leave=False, desc=f"QAs for {image_filename}"):
            question = qa_pair.get("Question", "")
            gt_answer = qa_pair.get("Answer", "")
            question_type = qa_pair.get("Type", "")
            if question_type != "Comprehensive Analysis":
                continue
            if cur_cnt >= 2:
                continue
            cur_cnt += 1

            prof.reset(clear_stats=True)
            injector.reset()
            injector.register_step_hooks()
            if inject_fault:
                injector.set_num_bits(2)
                injector.inject()

            pred = generate_answer_from_path(
                question=question,
                image_path=image_path,
                tokenizer=tokenizer,
                model=model,
                image_processor=image_processor,
                model_name=model_name,
                max_new_tokens=max_new_tokens,
            )

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
                img_path=image_filename,
                question=question,
                gt_answer=gt_answer,
                clean_pred=clean_pred,
                mean_std_cos=sample_result,
                pred=pred,
            )

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
    parser.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="/data0/home/lc/cd/llm/datasets/EarthVLSet/EarthVQA/Train/images_png")
    parser.add_argument("--dataset_json", type=str, default="/data0/home/lc/cd/llm/datasets/EarthVLSet/EarthVQA/Train_QA.json")
    parser.add_argument("--output_jsonl", type=str, default="/data1/home/dataset_share/cd_data/llava-v1.5-7B/EarthVQA/final/detect_EarthVQA_llava_with_sem_project.jsonl")
    parser.add_argument("--golden_json", type=str, default="./json/Golden_EarthVQA_llava-v1.5-7B_30_CA.json")
    parser.add_argument("--mapping_model", type=str, default="./model/best_mapping_model.pt")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_samples", type=int, default=5000)
    args = parser.parse_args()

    set_seed(42)
    device = torch.device(args.device)
    clean_answers = load_clean_answers(args.golden_json)

    evaluate(
        model_path=args.model_path,
        model_base=args.model_base,
        data_dir=args.data_dir,
        dataset_json=args.dataset_json,
        output_jsonl=args.output_jsonl,
        device=device,
        mapping_model_path=args.mapping_model,
        clean_answers=clean_answers,
        inject_fault=False,
        max_new_tokens=50,
        max_samples=args.max_samples,
    )
    for run in range(8):
        random.seed(42 + run)
        evaluate(
            model_path=args.model_path,
            model_base=args.model_base,
            data_dir=args.data_dir,
            dataset_json=args.dataset_json,
            output_jsonl=args.output_jsonl,
            device=device,
            mapping_model_path=args.mapping_model,
            clean_answers=clean_answers,
            run_time=run,
            inject_fault=True,
            max_new_tokens=50,
            max_samples=args.max_samples,
        )


if __name__ == "__main__":
    main()
