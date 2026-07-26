import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from ternary_xgboost_common import run_ternary_xgboost_compare_nan_modes


TRAIN_CSV = "/data01/cd_workspace/Detect_SDC/llava-v1.5-7B/LingoQA/train_data/llava-v1.5-7B_train_set.csv"
VALID_CSV = "/data01/cd_workspace/Detect_SDC/llava-v1.5-7B/LingoQA/train_data/llava-v1.5-7B_valid_set.csv"
GROUP_COL = "orig_id"


if __name__ == "__main__":
    run_ternary_xgboost_compare_nan_modes(
        train_csv=TRAIN_CSV,
        valid_csv=VALID_CSV,
        group_col=GROUP_COL,
    )
