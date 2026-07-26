import os
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

class ScoreEvaluator:
    def __init__(self, judge_model_name=None, json_path=None, local_files_only=True):
        if judge_model_name is None:
            judge_model_name = os.environ.get("LINGO_JUDGE_MODEL", "wayveai/Lingo-Judge")
        tokenizer = AutoTokenizer.from_pretrained(judge_model_name, local_files_only=local_files_only)
        model = AutoModelForSequenceClassification.from_pretrained(
            judge_model_name,
            local_files_only=local_files_only,
            use_safetensors=False,
        )
        self.lingo_judge = pipeline("text-classification", model=model, tokenizer=tokenizer)
        self.data = None
        if json_path is not None:
            with open(json_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
    
    # 计算故障注入后预测答案的分数
    def get_fault_scores(self, question, answer, prediction, idx):
        input = f"[CLS]\nQuestion: {question}\nAnswer: {answer}\nStudent: {prediction}"
        result = self.lingo_judge(input)
        after_score = result[0]['score']

        before_score = None
        clear_pred = None
        if idx < len(self.data):
            before_score = self.data[idx]['score']
            clear_pred = self.data[idx]['pre_answer']
        else:
            before_score = None
        return before_score, after_score, clear_pred
    
    def get_score(self, question, gt_answer, prediction):
        input = f"[CLS]\nQuestion: {question}\nAnswer: {gt_answer}\nStudent: {prediction}"
        result = self.lingo_judge(input)
        return result[0]['score']

    def get_steps(self, idx):
        return self.data[idx]['forwards']
    

if __name__ == "__main__":
    se = ScoreEvaluator()
    print(se.get_score(
        "Where are the kids riding?",
        "carnival ride",
        "Kids are riding in fire truck-themed bumper cars at a fairground.",
    ))
