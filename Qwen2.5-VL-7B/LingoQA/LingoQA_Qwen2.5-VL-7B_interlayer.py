import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys,json,random
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Any, Dict
from collections import defaultdict, Counter
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.Lingo_judge import ScoreEvaluator
from utils.fault_injector import FaultInjector
from utils.profiler import Profiler

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
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(to_jsonable(data), ensure_ascii=False) + "\n")


def collect_profiler_features(prof: Profiler) -> Dict[str, Any]:
    features = {
        "interlayer_attn": prof.get_interlayer_transition(branch="attn"),
        "interlayer_mlp": prof.get_interlayer_transition(branch="mlp"),
        "temporal_attn": prof.get_layerwise_temporal_transition(branch="attn"),
        "temporal_mlp": prof.get_layerwise_temporal_transition(branch="mlp"),
        "layerwise_attn": prof.get_layerwise_vector_stats(branch="attn"),
        "layerwise_mlp": prof.get_layerwise_vector_stats(branch="mlp"),
        "meta": {
            "num_steps": prof.get_steps(),
            "num_attn_layers": len(prof.decode_attn_by_layer),
            "num_mlp_layers": len(prof.decode_mlp_by_layer),
        }
    }

    return features


def build_result_record(
    idx,
    run_time,
    before_score,
    after_score,
    injector,
    img_path,
    question,
    gt_answer,
    clean_pred,
    pred,
    prof: Profiler,
):
    structured_feats = collect_profiler_features(prof)

    result = {
        "id": idx,
        "run_time": run_time,
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
        "features": to_jsonable(structured_feats),
    }
    return result


def evaluate(
    model_path,
    val_file,
    data_dir,
    output_jsonl,
    device,
    score_evaluator: ScoreEvaluator,
    run_time: int,
    max_new_tokens: int = 100,
):
    
    df = pd.read_parquet(val_file)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path,torch_dtype=torch.bfloat16).eval().to(device)
    processor = AutoProcessor.from_pretrained(model_path,min_pixels=256 * 28 * 28,max_pixels=1280 * 28 * 28)

    injector = FaultInjector(model, mode="activation")

    prof = Profiler(model, proj_dim=64, seed=1234)
    prof.register()

    sample_id=0
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Detecting run={run_time}"):
        question = row["question"]
        gt_answer = row["answer"]

        for i in range(5):
            prof.reset(clear_stats=True)
            injector.reset()
            injector.register_step_hooks()
            if run_time > 0:
                injector.set_num_bits(2)
                injector.inject()
                
            img_path = row["images"][i]
            image_path = os.path.join(data_dir, img_path)
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
            before_score, after_score, clean_pred = score_evaluator.get_fault_scores(
                question, gt_answer, pred, sample_id
            )
            result = build_result_record(
                idx=sample_id,
                run_time=run_time,
                before_score=before_score,
                after_score=after_score,
                injector=injector,
                img_path=img_path,
                question=question,
                gt_answer=gt_answer,
                clean_pred=clean_pred,
                pred=pred,
                prof=prof
            )

            append_jsonl(output_jsonl, result)
        
            injector.unregister_hooks()
            sample_id += 1
    prof.unregister()

def main():
    set_model_seed(42)

    device = torch.device("cuda:0")
    model_path = "/data1/home/dataset_share/wsh_data/data/qwen/Qwen2___5-VL-7B-Instruct"
    val_file = "/data0/home/lc/cd/predict_error/LingoQA-main/data/val/val.parquet"
    data_dir = "/data0/home/lc/cd/predict_error/LingoQA-main/data/val/"
    output_jsonl = "/data1/home/dataset_share/cd_data/Qwen2.5-VL-7B/LingoQA/detect_LingoQA_Qwen_with_inter_layer.jsonl"
    score_evaluator = ScoreEvaluator(json_path="./Golden_LingoQA_Qwen2.5-VL-7B.json")

    for run in range(2):
        random.seed(42 + run)
        evaluate(
            model_path=model_path,
            val_file=val_file,
            data_dir=data_dir,
            output_jsonl=output_jsonl,
            device=device,
            score_evaluator=score_evaluator,
            run_time=run,
            max_new_tokens=50,
        )
if __name__ == "__main__":
    main()
