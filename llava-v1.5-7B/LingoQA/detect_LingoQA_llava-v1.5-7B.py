import os, sys
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import random
import csv
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.score_utils import ScoreEvaluator
from utils.fault_injector import FaultInjector 
from epaa.profiler import Profiler

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
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_llava_model(model_path, device):
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=model_name
    )
    model.eval().to(device)
    return tokenizer, model, image_processor, model_name


def build_prompt_and_input(question, model, model_name):
    """构建多模态输入提示"""
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    qs = question

    # 插入图片占位符
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

    # 选择对话模板
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    else:
        conv_mode = "llava_v0"

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    return prompt

def calc_abnormal_score_2(
    stats: dict[int, dict[int, dict]],
    layers: dict[str, dict[str, dict]],
    a: float = 0.7,
    b: float = 0.2,
    c: float = 0.3,
) -> float:
    # 定义要检查的层和对应的 top 排名及权重
    layer_configs = [
        (30, "30", [1, 2, 3], [a, b, c]),
        (31, "31", [1, 2, 3], [a, b, c]),
    ]
    
    total_score = 0.0

    for stat_layer, layer_key, ranks, weights in layer_configs:
        for rank, weight in zip(ranks, weights):
            # 从 layers 中获取 ground truth 数据
            g_entry = layers[layer_key][f"top{rank}"]
            g_idx = g_entry["mode_idx"]
            g_diff = g_entry["diff_rate"]

            # 从 stats 中获取当前统计结果
            f_entry = stats[stat_layer][rank]
            f_idx = f_entry["mode_idx"]
            f_diff = f_entry["diff_rate"]

            # 计算该 rank 的异常分
            if g_idx != f_idx:
                score = 1.0
            else:
                score = abs(g_diff - f_diff)

            total_score += score * weight

    return total_score

def calc_abnormal_score_3(
    stats: dict[int, dict[int, dict]],
    layers: dict[str, dict[str, dict]], 
    a: float=0.7, 
    b: float=0.2, 
    c: float=0.3
):
    g_l30_top1_idx = layers['30']['top1']['mode_idx']
    g_l30_top2_idx = layers['30']['top2']['mode_idx']
    g_l30_top3_idx = layers['30']['top3']['mode_idx']
    g_l31_top1_idx = layers['31']['top1']['mode_idx']
    g_l31_top2_idx = layers['31']['top2']['mode_idx']
    g_l31_top3_idx = layers['31']['top3']['mode_idx']
    g_l30_top1_diff_rate = layers['30']['top1']['diff_rate']
    g_l30_top2_diff_rate = layers['30']['top2']['diff_rate']
    g_l30_top3_diff_rate = layers['30']['top3']['diff_rate']
    g_l31_top1_diff_rate = layers['31']['top1']['diff_rate']
    g_l31_top2_diff_rate = layers['31']['top2']['diff_rate']
    g_l31_top3_diff_rate = layers['31']['top3']['diff_rate']

    f_l30_top1_idx = stats[30][1]['mode_idx']
    f_l30_top2_idx = stats[30][2]['mode_idx']
    f_l30_top3_idx = stats[30][3]['mode_idx']
    f_l31_top1_idx = stats[31][1]['mode_idx']
    f_l31_top2_idx = stats[31][2]['mode_idx']
    f_l31_top3_idx = stats[31][3]['mode_idx']
    f_l30_top1_diff_rate = stats[30][1]['diff_rate']
    f_l30_top2_diff_rate = stats[30][2]['diff_rate']
    f_l30_top3_diff_rate = stats[30][3]['diff_rate']
    f_l31_top1_diff_rate = stats[31][1]['diff_rate']
    f_l31_top2_diff_rate = stats[31][2]['diff_rate']
    f_l31_top3_diff_rate = stats[31][3]['diff_rate']

    l30_abscore_top1, l30_abscore_top2, l30_abscore_top3 = 0, 0, 0
    l31_abscore_top1, l31_abscore_top2, l31_abscore_top3 = 0, 0, 0

    if g_l30_top1_idx != f_l30_top1_idx:
        l30_abscore_top1 = 1.0
    else:
        l30_abscore_top1 = abs(g_l30_top1_diff_rate - f_l30_top1_diff_rate)
    
    if g_l30_top2_idx != f_l30_top2_idx:
        l30_abscore_top2 = 1.0
    else:
        l30_abscore_top2 = abs(g_l30_top2_diff_rate - f_l30_top2_diff_rate)
    
    if g_l30_top3_idx != f_l30_top3_idx:
        l30_abscore_top3 = 1.0
    else:
        l30_abscore_top3 = abs(g_l30_top3_diff_rate - f_l30_top3_diff_rate)

    if g_l31_top1_idx != f_l31_top1_idx:
        l31_abscore_top1 = 1.0
    else:
        l31_abscore_top1 = abs(g_l31_top1_diff_rate - f_l31_top1_diff_rate)
    
    if g_l31_top2_idx != f_l31_top2_idx:
        l31_abscore_top2 = 1.0
    else:
        l31_abscore_top2 = abs(g_l31_top2_diff_rate - f_l31_top2_diff_rate)
    
    if g_l31_top3_idx != f_l31_top3_idx:
        l31_abscore_top3 = 1.0
    else:
        l31_abscore_top3 = abs(g_l31_top3_diff_rate - f_l31_top3_diff_rate)
    
    abscore=l30_abscore_top1*a + l30_abscore_top2*b + l30_abscore_top3*c + l31_abscore_top1*a + l31_abscore_top2*b + l31_abscore_top3*c
    return {

    }

def calc_abnormal_score(
    stats_mlp: dict[int, dict[int, dict]],
    stats_attn: dict[int, dict[int, dict]],
    layers: dict[str, dict[str, dict]], 
    a: float = 0.7, 
    b: float = 0.2, 
    c: float = 0.1
):
    # 定义要处理的层和对应的权重
    layer_info = [
        ("30_mlp", "30_mlp", [a, b, c]),
        ("31_mlp", "31_mlp", [a, b, c]),
    ]
    
    # 存储各层各 top 的异常分
    scores = {}
    total_abscore = 0.0

    for stat_layer, layer_key, weights in layer_info:
        layer_scores = {}
        for i, weight in enumerate(weights, start=1):  # i = 1, 2, 3
            rank = i
            g = layers[layer_key][f"top{rank}"]
            f = stats_mlp[stat_layer][rank]

            if g["mode_idx"] != f["mode_idx"]:
                score = 1.0
            else:
                score = abs(g["diff_rate"] - f["diff_rate"])
            
            key = f"l{layer_key}_abscore_top{i}"
            layer_scores[key] = score
            total_abscore += score * weight

        scores.update(layer_scores)

    layer_info = [
        ("30_attn_o", "30_attn_o", [a, b, c]),
        ("31_attn_o", "31_attn_o", [a, b, c]),
    ]

    for stat_layer, layer_key, weights in layer_info:
        layer_scores = {}
        for i, weight in enumerate(weights, start=1):  # i = 1, 2, 3
            rank = i
            g = layers[layer_key][f"top{rank}"]
            f = stats_attn[stat_layer][rank]

            if g["mode_idx"] != f["mode_idx"]:
                score = 1.0
            else:
                score = abs(g["diff_rate"] - f["diff_rate"])
            
            key = f"l{layer_key}_abscore_top{i}"
            layer_scores[key] = score
            total_abscore += score * weight

        scores.update(layer_scores)
    
    return {
        "abscore": total_abscore,
        **scores 
    }


def compute_diff(layer_top1_mlp, layer_top1_attn, g_layer_top1):
    f_l30_mlp = layer_top1_mlp['30_mlp']
    f_l31_mlp = layer_top1_mlp['31_mlp']
    f_l30_attn = layer_top1_attn['30_attn_o']
    f_l31_attn = layer_top1_attn['31_attn_o']

    g_l30_mlp = g_layer_top1['30_mlp']['top1_list']
    g_l31_mlp = g_layer_top1['31_mlp']['top1_list']
    g_l30_attn = g_layer_top1['30_attn_o']['top1_list']
    g_l31_attn = g_layer_top1['31_attn_o']['top1_list']

    min_len = min(len(f_l30_mlp), len(g_l30_mlp))
    l30_mlp_top1_diff = sum(1 for i in range(min_len) if f_l30_mlp[i] != g_l30_mlp[i])
    l31_mlp_top1_diff = sum(1 for i in range(min_len) if f_l31_mlp[i] != g_l31_mlp[i])
    l30_attn_top1_diff = sum(1 for i in range(min_len) if f_l30_attn[i] != g_l30_attn[i])
    l31_attn_top1_diff = sum(1 for i in range(min_len) if f_l31_attn[i] != g_l31_attn[i])

    return float(l30_mlp_top1_diff/min_len), float(l31_mlp_top1_diff/min_len), float(l30_attn_top1_diff/min_len), float(l31_attn_top1_diff/min_len)

def evaluate(
    model_path, 
    val_file, 
    data_dir, 
    output_jsonl, 
    device,
    profile_json,
    score_evaluator: ScoreEvaluator
):
    tokenizer, model, image_processor, model_name = load_llava_model(model_path, device)
    df = pd.read_parquet(val_file)
    
    with open(profile_json, 'r', encoding='utf-8') as f:
        profile_data = json.load(f)

    injector = FaultInjector(model)
    prof = Profiler(model)
    prof.register()

    idx = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Detecting"):
        question = row["question"]
        gt_answer = row["answer"]

        for i in range(5):
            prof.reset(clear_stats=True)
            injector.reset()
            injector.set_total_step(profile_data[idx]['layers']['30_mlp']['top1']['total'])
            injector.register_fault_hooks()
            injector.register_step_hooks()

            img_path = data_dir + row["images"][i]
            image = Image.open(img_path).convert("RGB")
            prompt = build_prompt_and_input(question + " The answer must be limited to 50 words.", model, model_name)
            images_tensor = process_images([image], image_processor, model.config).to(device, dtype=torch.float16)
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images_tensor,
                    image_sizes=[image.size],
                    do_sample=False,
                    max_new_tokens=100,
                    use_cache=True,
                )

            pred = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip().lower()

            before_score, after_score, clear_pred = score_evaluator.get_fault_scores(question, gt_answer, pred, idx)
            stats_mlp = prof.get_stats_mlp()
            stats_attn = prof.get_stats_attn()
            layer_top1_mlp, layer_top1_attn = prof.get_layer_top1()

            g_layer_top1 = profile_data[idx]['layers']
            l30_mlp_top1_diff, l31_mlp_top1_diff, l30_attn_top1_diff, l31_attn_top1_diff = compute_diff(layer_top1_mlp, layer_top1_attn, g_layer_top1)

            abnormal_scores = calc_abnormal_score(stats_mlp, stats_attn, profile_data[idx]['layers'], 1.0, 0, 0)
            result = {
                "l30_mlp_top1_diff": l30_mlp_top1_diff,
                "l31_mlp_top1_diff": l31_mlp_top1_diff,
                "l30_attn_top1_diff": l30_attn_top1_diff,
                "l31_attn_top1_diff": l31_attn_top1_diff,
                "id": idx,
                "before_score": float(before_score),
                "after_score": float(after_score),
                "dtel_score": float(after_score - before_score),
                "fault": injector.fault_info,
                "image_path": img_path,
                "question": question,
                "gt_answer": gt_answer,
                "clean_answer": clear_pred,
                "pre_answer": pred,
            }
            abnormal_scores.update(result)

            idx += 1
            if abs(after_score - before_score) > 0:
                with open(output_jsonl, "a", encoding="utf-8") as f:
                    f.write(json.dumps(abnormal_scores, ensure_ascii=False) + "\n")
            injector.unregister_hooks()

    prof.unregister()
    
def main():
    set_seed(42)

    device = torch.device("cuda:0")
    model_path = "liuhaotian/llava-v1.5-7b"
    val_file = "/data0/home/lc/cd/predict_significant_error/Tasks/LingoQA-main/data/val/val.parquet"
    data_dir = "/data0/home/lc/cd/predict_significant_error/Tasks/LingoQA-main/data/val/"
    output_jsonl = "./detect_LingoQA_llava-v1.5-7B.jsonl"
    profile_json = "./json/Profile_LingoQA_llava-v1.5-7B.json"
    score_evaluator = ScoreEvaluator(json_path="./json/Golden_LingoQA_llava-v1.5-7B.json")
    
    evaluate(
        model_path=model_path,
        val_file=val_file,
        data_dir=data_dir,
        output_jsonl=output_jsonl,
        device=device,
        profile_json=profile_json,
        score_evaluator=score_evaluator,
    )

if __name__ == "__main__":
    main()
