import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import re

# 配置
INPUT_FILE = "/data1/home/dataset_share/cd_data/Qwen2.5-VL-7B/EarthVQA/final/detect_EarthVQA_Qwen_with_sem_project.jsonl"
OUTPUT_FILE = "/data1/home/dataset_share/cd_data/Qwen2.5-VL-7B/EarthVQA/final/detect_EarthVQA_Qwen_with_sem_project_labeled.jsonl"
DEVICE = "cuda:3"

# 加载模型
print("Loading Prometheus model...")
model = AutoModelForCausalLM.from_pretrained("/data0/home/lc/cd/llm/prometheus")
tokenizer = AutoTokenizer.from_pretrained("/data0/home/lc/cd/llm/prometheus")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
model.to(DEVICE)
print("Model loaded.")

# Prompt 模板
ABS_SYSTEM_PROMPT = "You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance."

ABSOLUTE_PROMPT = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 0, and a score rubric representing an evaluation criteria are given.
1. Write a detailed feedback that assesses the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 0 and 2. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 0 and 2)"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Reference Answer (Score 0):
{reference_answer}

###Score Rubrics:
{rubric}

###Feedback: """

RUBRIC = """
Score 0: The answer is completely correct and matches the reference answer.
Score 1: The answer has minor deviations from the reference answer but is semantically correct.
Score 2: The answer is completely wrong or contains semantic errors.
"""


def extract_score(text):
    """从模型输出中提取分数"""
    # 匹配 [RESULT] X 格式
    match = re.search(r'\[RESULT\]\s*(\d)', text)
    if match:
        return int(match.group(1))
    # 备用匹配：查找最后一个数字
    numbers = re.findall(r'\b[0-2]\b', text)
    if numbers:
        return int(numbers[-1])
    return -1


def build_prompt(instruction, response, reference_answer):
    """构建单个样本的 prompt"""
    sample_data = {
        "instruction": instruction,
        "response": response,
        "reference_answer": reference_answer,
        "rubric": RUBRIC
    }
    return ABS_SYSTEM_PROMPT + "\n\n" + ABSOLUTE_PROMPT.format(**sample_data)


def evaluate_batch(batch_items):
    """批量评估样本"""
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
    messages_list = [[{"role": "user", "content": p}] for p in prompts]
    
    # 批量 tokenize（自动 padding）
    encodeds = tokenizer.apply_chat_template(
        messages_list, 
        return_tensors="pt", 
        padding=True,
        return_dict=True
    )
    model_inputs = {k: v.to(DEVICE) for k, v in encodeds.items()}
    
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
    # 配置
    BATCH_SIZE = 64  # 根据 GPU 显存调整
    
    # 读取输入文件
    print(f"Reading {INPUT_FILE}...")
    data = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    print(f"Total samples: {len(data)}")
    
    # 批量处理
    results = []
    for i in tqdm(range(0, len(data), BATCH_SIZE), desc="Evaluating"):
        batch_items = data[i:i + BATCH_SIZE]
        
        # 批量评估
        scores, _ = evaluate_batch(batch_items)
        
        # 添加 label 字段并追加写入
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
            for item, score in zip(batch_items, scores):
                item["label"] = score
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
                results.append(item)
    
    print(f"\nDone! Output saved to {OUTPUT_FILE}")
    
    # 统计分数分布
    score_counts = {0: 0, 1: 0, 2: 0, -1: 0}
    for r in results:
        s = r["label"]
        if s in score_counts:
            score_counts[s] += 1
    
    print(f"\nScore distribution:")
    print(f"  Score 0 (Correct): {score_counts[0]}")
    print(f"  Score 1 (Partial): {score_counts[1]}")
    print(f"  Score 2 (Wrong): {score_counts[2]}")
    print(f"  Score -1 (Parse Error): {score_counts[-1]}")


if __name__ == "__main__":
    main()
