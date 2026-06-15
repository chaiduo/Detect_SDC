import json
from tqdm import tqdm
from sentence_transformers import CrossEncoder

class SimilarityEvaluator:
    def __init__(self, model_name='/data0/home/lc/cd/stsb-roberta-base', json_path=None):
        self.model = CrossEncoder(model_name)
        self.data = None
        if json_path is not None:
            with open(json_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)

    def score(self, text1, text2):
        return float(self.model.predict([(str(text1), str(text2))])[0])

    def get_fault_scores(self, pred, idx):
        answer = self.data[idx]['gt_answer']
        clean_pred = self.data[idx]['pre_answer']
        before_score = self.data[idx]['scores']
        after_score = self.score(answer, pred)
        return before_score, after_score, clean_pred


if __name__ == "__main__":
    se = SimilarityEvaluator()
    print(se.score(
        # "There is a car.",
        # "There are two cars."
        "There is one cyclist in the lane, riding a bicycle.",
        "There are no cyclists in the lane; it appears empty of them."
        # "The red car is driving away from the camera on a two-lane road.",
        # "The red car is driving forward on the road, heading towards the camera."
    ))
   