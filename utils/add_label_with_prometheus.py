import json
import argparse
from pathlib import Path
import re

# 配置
DEFAULT_INPUT_FILE = "/data01/cd_workspace/Detect_SDC/Qwen2.5-VL-7B/EarthVQA/json/detect_EarthVQA_Qwen_with_sem_project.jsonl"
MODEL_PATH = "/data01/cd_workspace/llm/prometheus-7b-v2.0"
DEFAULT_DEVICE = "cuda:0"

# Prompt 模板
ABS_SYSTEM_PROMPT = "You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance."

ABSOLUTE_PROMPT = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer produced by the fault-free baseline, and a score rubric representing an evaluation criteria are given.
1. Write a detailed feedback that assesses the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 0 and 2. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 0 and 2)"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Reference Answer (fault-free baseline):
{reference_answer}

###Score Rubrics:
{rubric}

###Feedback: """

RUBRIC = """
Score 2: The response is semantically equivalent to the reference answer. It preserves the same core answer and would lead to the same decision or interpretation, even if the wording is different.
Score 1: The response has minor semantic deviations from the reference answer, but the core answer is still mostly preserved and the difference is unlikely to cause a wrong decision.
Score 0: The response has a major semantic deviation from the reference answer, changes or loses the core answer, contradicts the reference answer, or would likely cause a wrong decision.
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        "--input-file",
        default=DEFAULT_INPUT_FILE,
        help="Input JSONL file. Output will be saved as <input_stem>_labeled<input_suffix>.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="CUDA GPU index, e.g. --gpu 2 means cuda:2. Overrides --device.",
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help="Torch device, e.g. cuda:0, cuda:2, or cpu. Ignored when --gpu is set.",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for Prometheus inference.")
    return parser.parse_args()


def resolve_device(args):
    if args.gpu is not None:
        return f"cuda:{args.gpu}"
    return args.device


def load_model(device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading Prometheus model on {device}...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.to(device)
    model.eval()
    print("Model loaded.")
    return model, tokenizer


def build_output_file(input_file):
    input_path = Path(input_file)
    return str(input_path.with_name(f"{input_path.stem}_labeled{input_path.suffix}"))


def build_debug_file(input_file):
    input_path = Path(input_file)
    return str(input_path.with_name(f"{input_path.stem}_prometheus_parse_failed{input_path.suffix}"))


def extract_score(text):
    """严格从 Prometheus 输出中提取质量分数。

    只接受唯一的标准格式: [RESULT] 0/1/2。其他正文里的数字都不解析，
    避免把反馈中提到的 rubric 分数误当成最终结果。
    """
    result_matches = re.findall(r'\[RESULT\]\s*([0-2])\s*$', text.strip(), flags=re.IGNORECASE)
    if len(result_matches) == 1:
        return int(result_matches[0])
    return -1


def quality_score_to_significance(score):
    """Prometheus 质量分越高越好；下游 significance 越高表示错误越严重。"""
    if score not in (0, 1, 2):
        return -1
    return 2 - score


def build_prompt(instruction, response, reference_answer):
    """构建单个样本的 prompt"""
    sample_data = {
        "instruction": instruction,
        "response": response,
        "reference_answer": reference_answer,
        "rubric": RUBRIC
    }
    return ABSOLUTE_PROMPT.format(**sample_data)


def evaluate_batch(batch_items, model, tokenizer, device):
    """批量评估样本"""
    import torch

    # 构建 prompts
    prompts = []
    for item in batch_items:
        prompt = build_prompt(
            instruction=item["question"],
            response=item["pred_answer"],
            reference_answer=item["clean_answer"]
        )
        prompts.append(prompt)
    
    # 构建 messages 并 tokenize
    messages_list = [
        [
            {"role": "system", "content": ABS_SYSTEM_PROMPT},
            {"role": "user", "content": p},
        ]
        for p in prompts
    ]
    
    # 批量 tokenize（自动 padding）
    encodeds = tokenizer.apply_chat_template(
        messages_list, 
        return_tensors="pt", 
        padding=True,
        truncation=True,
        max_length=4096,
        return_dict=True
    )
    model_inputs = {k: v.to(device) for k, v in encodeds.items()}
    
    # 批量生成
    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs, 
            max_new_tokens=200, 
            do_sample=False, 
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 只解码新生成的部分
    input_len = model_inputs['input_ids'].shape[1]
    new_tokens = generated_ids[:, input_len:]
    decoded_list = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    
    # 提取分数
    scores = [extract_score(d) for d in decoded_list]
    
    return scores, decoded_list


def main():
    args = parse_args()
    from tqdm import tqdm

    input_file = args.input_file
    device = resolve_device(args)
    output_file = build_output_file(input_file)
    debug_file = build_debug_file(input_file)
    model, tokenizer = load_model(device)
    
    # 读取输入文件
    print(f"Reading {input_file}...")
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    print(f"Total samples: {len(data)}")
    open(output_file, 'w', encoding='utf-8').close()
    open(debug_file, 'w', encoding='utf-8').close()
    
    # 先全局过滤完全一致的答案，避免无效 tokenization，并让剩余样本组成更满的 batch。
    results = [None] * len(data)
    need_eval_items = []
    need_eval_indices = []
    for idx, item in enumerate(data):
        if str(item.get("pred_answer", "")) == str(item.get("clean_answer", "")):
            item["quality_score"] = 2
            item["significance"] = 0
            results[idx] = item
        else:
            need_eval_items.append(item)
            need_eval_indices.append(idx)

    print(f"Skipped identical answers: {len(data) - len(need_eval_items)}")
    print(f"Prometheus samples: {len(need_eval_items)}")

    next_write_idx = 0

    def flush_ready(output_f):
        nonlocal next_write_idx
        while next_write_idx < len(results) and results[next_write_idx] is not None:
            output_f.write(json.dumps(results[next_write_idx], ensure_ascii=False) + '\n')
            next_write_idx += 1

    # 只对非相同答案批量评估。Prometheus 返回 quality_score: 2=最好, 0=最差。
    with open(output_file, 'a', encoding='utf-8') as f, open(debug_file, 'a', encoding='utf-8') as debug_f:
        flush_ready(f)
        for i in tqdm(range(0, len(need_eval_items), args.batch_size), desc="Evaluating"):
            batch_items = need_eval_items[i:i + args.batch_size]
            batch_indices = need_eval_indices[i:i + args.batch_size]
            quality_scores, decoded_list = evaluate_batch(batch_items, model, tokenizer, device)

            for raw_idx, item, quality_score, feedback in zip(batch_indices, batch_items, quality_scores, decoded_list):
                significance = quality_score_to_significance(quality_score)
                item["quality_score"] = quality_score
                item["significance"] = significance
                results[raw_idx] = item
                if quality_score == -1:
                    debug_f.write(json.dumps({
                        "question": item.get("question"),
                        "clean_answer": item.get("clean_answer"),
                        "pred_answer": item.get("pred_answer"),
                        "quality_score": quality_score,
                        "significance": significance,
                        "prometheus_feedback": feedback,
                    }, ensure_ascii=False) + '\n')
            flush_ready(f)

    if any(item is None for item in results):
        raise RuntimeError("Internal error: unprocessed sample found")
    
    print(f"\nDone! Output saved to {output_file}")
    
    # 统计分数分布
    score_counts = {0: 0, 1: 0, 2: 0, -1: 0}
    for r in results:
        s = r["significance"]
        if s in score_counts:
            score_counts[s] += 1
    
    print(f"\nScore distribution:")
    print(f"  Score 0 (Correct): {score_counts[0]}")
    print(f"  Score 1 (Partial): {score_counts[1]}")
    print(f"  Score 2 (Wrong): {score_counts[2]}")
    print(f"  Score -1 (Parse Error): {score_counts[-1]}")


if __name__ == "__main__":
    main()
