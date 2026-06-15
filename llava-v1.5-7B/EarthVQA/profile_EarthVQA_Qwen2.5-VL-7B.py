import os, sys
# 设置HF镜像环境变量
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import random
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# 如果需要，可以取消注释以下两行来添加自定义模块路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from epaa.profiler import Profiler
from utils.similarity_utils import SimilarityEvaluator

def set_seed(seed: int):
    """设置随机种子以确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate(
    model_path, 
    data_dir, 
    dataset_json, 
    profile_jsonl, 
    golden_jsonl,
    device,
    similarity_evaluator: SimilarityEvaluator
):
    print("Loading model and processor...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path,torch_dtype=torch.bfloat16,device_map=device).eval() 
    processor = AutoProcessor.from_pretrained(model_path,min_pixels=256*28*28,max_pixels=1280*28*28)

    print(f"Loading dataset from {dataset_json}...")
    with open(dataset_json, 'r') as f:
        raw_data = json.load(f)

    # 转换为DataFrame，其中 index 是图片名， value 是问答列表
    df = pd.DataFrame(list(raw_data.items()), columns=['image_filename', 'qa_list'])
    prof = Profiler(model)
    prof.register()

    idx = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Images"):
        image_filename = row['image_filename']
        qa_list = row['qa_list']
        image_path = os.path.join(data_dir, image_filename)

        if not os.path.exists(image_path):
             print(f"Warning: Image {image_path} does not exist. Skipping.")
             continue

        # 遍历当前图片的所有问答对
        for qa_pair in tqdm(qa_list, leave=False, desc=f"QAs for {image_filename}"):
            question_text = qa_pair.get('Question', '')
            answer_text = qa_pair.get('Answer', '')
            question_type = qa_pair.get('Type', '')
            if "area" in question_text:
                continue

            prof.reset(clear_stats=True)
            
            messages = [{"role": "user", "content": [
                {"type": "image", "image": "file://" + image_path},
                {"type": "text", "text": question_text + " The answer must be limited to 50 words."},
            ]}]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt").to(device)
            generated_ids = model.generate(**inputs,max_new_tokens=100,do_sample=False,)
            trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            pred = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
            
            result = {
                "id": idx,
                "image_path": image_filename,
                "type": question_type,
                "question": question_text,
                "gt_answer": answer_text,
                "pre_answer": pred,
                "scores": similarity_evaluator.score(answer_text, pred),
                "forwards": prof.get_steps()
            }
            
            with open(golden_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            
            stats_mlp = prof.get_stats_mlp()
            stats_attn = prof.get_stats_attn()
            layer_top1_mlp, layer_top1_attn = prof.get_layer_top1()
            profile = {
                "id": idx,
                "layers": {}
            }
            for layer_idx, rank_map in stats_mlp.items():
                layer_entry = {
                    "top1_list": layer_top1_mlp[layer_idx],
                }
                for rank, s in rank_map.items():
                    layer_entry[f"top{rank}"] = {
                        "mode_idx": s["mode_idx"],
                        "mode_rate": s["mode_rate"],
                        "diff_rate": s["diff_rate"],
                        "total": s["total"],
                    }
                profile["layers"][str(layer_idx)] = layer_entry
            
            for layer_idx, rank_map in stats_attn.items():
                layer_entry = {
                    "top1_list": layer_top1_attn[layer_idx],
                }
                for rank, s in rank_map.items():
                    layer_entry[f"top{rank}"] = {
                        "mode_idx": s["mode_idx"],
                        "mode_rate": s["mode_rate"],
                        "diff_rate": s["diff_rate"],
                        "total": s["total"],
                    }
                profile["layers"][str(layer_idx)] = layer_entry

            with open(profile_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(profile, ensure_ascii=False) + "\n")
            
            idx += 1
            if idx >= 5000:
                break
        if idx >= 5000:
            break

    print("Results saved successfully.")

def main():
    set_seed(42)
    device = torch.device("cuda:0")
    model_path = "/data1/home/dataset_share/wsh_data/data/qwen/Qwen2___5-VL-7B-Instruct"
    data_dir = "/data0/home/lc/cd/llm/datasets/EarthVLSet/EarthVQA/Train/images_png"
    dataset_json = "/data0/home/lc/cd/llm/datasets/EarthVLSet/EarthVQA/Train_QA.json"
    profile_jsonl = "Profile_EarthVQA_Qwen2.5-VL-7B.jsonl"
    golden_jsonl = "Golden_EarthVQA_Qwen2.5-VL-7B.jsonl"
    se = SimilarityEvaluator()
    evaluate(
        model_path=model_path,
        data_dir=data_dir,
        dataset_json=dataset_json,
        profile_jsonl=profile_jsonl, 
        golden_jsonl=golden_jsonl,
        device=device,
        similarity_evaluator = se
    )

if __name__ == "__main__":
    main()