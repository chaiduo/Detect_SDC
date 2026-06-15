import os
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

class ScoreEvaluator:
    def __init__(self, judge_model_name='/data0/home/lc/cd/LingoQA', json_path=None):
        tokenizer = AutoTokenizer.from_pretrained(judge_model_name, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(judge_model_name, local_files_only=True,use_safetensors=False)
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