## !/bin/bash
python ../similarity_utils.py \
    --input_jsonl /data0/home/lc/cd/predict_significant_error/Tasks/Code/pred_error/InternVL3-8B/json/Fault_LingoQA-InternVL3-8B.jsonl \
    --output_json /data0/home/lc/cd/predict_significant_error/Tasks/Code/pred_error/InternVL3-8B/json/Filtered_Fault_LingoQA-InternVL3-8B.json \
    --threshold 0.5