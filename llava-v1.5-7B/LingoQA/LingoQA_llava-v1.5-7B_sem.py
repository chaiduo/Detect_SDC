import os
import sys
import pandas as pd
from tqdm import tqdm
import torch

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
grandparent_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.extend([parent_dir, grandparent_dir])

from llava_common import append_jsonl, generate_answer_from_path, load_llava_model, set_seed, to_jsonable
from utils.profiler import Profiler
from utils.Lingo_judge import ScoreEvaluator
from utils.fault_injector import FaultInjector
from train_mapping_model import LayerAwareResidualMLP


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
    return {
        "id": idx,
        "before_score": float(before_score),
        "after_score": float(after_score),
        "dtel_score": float(after_score - before_score),
        "is_sdc": int(before_score != after_score),
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
    val_file,
    data_dir,
    output_jsonl,
    device,
    score_evaluator: ScoreEvaluator,
    mapping_model_path: str = "./model/mapping_model.pt",
    model_base: str = None,
    run_time: int = 0,
    inject_fault: bool = True,
    max_new_tokens: int = 50,
):
    df = pd.read_parquet(val_file)

    tokenizer, model, image_processor, model_name = load_llava_model(
        model_path=model_path,
        model_base=model_base,
        device=device,
    )

    injector = FaultInjector(model, mode="activation")

    prof = Profiler(model, proj_dim=64, seed=1234)
    prof.register()

    mapping_model = LayerAwareResidualMLP(
        x_dim=64,
        num_layers=32,
        layer_emb_dim=16,
        hidden_dim=1024,
        num_blocks=8,
        dropout=0.1,
    ).to(device)

    state_dict = torch.load(mapping_model_path, map_location=device)
    mapping_model.load_state_dict(state_dict)
    mapping_model.eval()

    sample_id = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Run={run_time}"):
        question = row["question"]
        gt_answer = row["answer"]

        images = row["images"]
        num_images = min(5, len(images))

        for i in range(num_images):
            prof.reset(clear_stats=True)
            injector.reset()
            injector.register_step_hooks()
            if inject_fault is True:
                injector.set_num_bits(2)
                injector.inject()

            img_path = images[i]
            image_path = os.path.join(data_dir, img_path)

            pred = generate_answer_from_path(
                question=question,
                image_path=image_path,
                tokenizer=tokenizer,
                model=model,
                image_processor=image_processor,
                model_name=model_name,
                max_new_tokens=max_new_tokens,
            )

            before_score, after_score, clean_pred = score_evaluator.get_fault_scores(
                question,
                gt_answer,
                pred,
                sample_id,
            )

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
                img_path=img_path,
                question=question,
                gt_answer=gt_answer,
                clean_pred=clean_pred,
                mean_std_cos=sample_result,
                pred=pred,
            )

            if inject_fault is False or run_time < 1:
                append_jsonl(output_jsonl, result)
            else:
                if before_score != after_score:
                    append_jsonl(output_jsonl, result)

            injector.unregister_hooks()
            sample_id += 1

    prof.unregister()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--val_file", type=str, default="/data0/home/lc/cd/predict_error/LingoQA-main/data/val/val.parquet")
    parser.add_argument("--data_dir", type=str, default="/data0/home/lc/cd/predict_error/LingoQA-main/data/val/")
    parser.add_argument("--output_jsonl", type=str, default="/data1/home/dataset_share/cd_data/llava-v1.5-7B/LingoQA/detect_LingoQA_llava_with_sem.jsonl")
    parser.add_argument("--golden_json", type=str, default="./json/Golden_LingoQA_llava-v1.5-7B.json")
    parser.add_argument("--mapping_model", type=str, default="./model/lingoqa_mapping_model.pt")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    set_seed(42)

    device = torch.device(args.device)
    score_evaluator = ScoreEvaluator(json_path=args.golden_json)

    evaluate(
        model_path=args.model_path,
        model_base=args.model_base,
        val_file=args.val_file,
        data_dir=args.data_dir,
        output_jsonl=args.output_jsonl,
        device=device,
        score_evaluator=score_evaluator,
        mapping_model_path=args.mapping_model,
        inject_fault=False,
        max_new_tokens=50,
    )

    for run in range(8):
        set_seed(42 + run)
        evaluate(
            model_path=args.model_path,
            model_base=args.model_base,
            val_file=args.val_file,
            data_dir=args.data_dir,
            output_jsonl=args.output_jsonl,
            device=device,
            score_evaluator=score_evaluator,
            mapping_model_path=args.mapping_model,
            run_time=run,
            inject_fault=True,
            max_new_tokens=50,
        )


if __name__ == "__main__":
    main()
