#!/bin/bash
set -e

# =========================
# 公共路径配置
# =========================
MODEL_PATH="/data01/cd_workspace/llm/llava-v1.5-7b"
MODEL_BASE=""
VAL_FILE="/data01/cd_workspace/llm/LingoQA/val.parquet"
DATA_DIR="/data01/cd_workspace/llm/LingoQA/"

# 设备配置
export CUDA_VISIBLE_DEVICES="0"
export HF_ENDPOINT="https://hf-mirror.com"
DEVICE="cuda:0"

# 中间文件路径
GOLDEN_JSON="./json/Golden_LingoQA_llava-v1.5-7B.json"
MAPPING_DIR="./json"
MAPPING_JSONL="${MAPPING_DIR}/attn_proj_interlayer.jsonl"
MAPPING_MODEL="./model/lingoqa_mapping_model.pt"
OUTPUT_SEM_JSONL="${MAPPING_DIR}/detect_LingoQA_llava_with_sem.jsonl"

# 确保输出目录存在
mkdir -p ./json
mkdir -p ./model
mkdir -p "${MAPPING_DIR}"

# =========================
# Step 1: Profile - 生成 golden 文件
# =========================
echo "========== Step 1: Profile =========="
echo "Input:  ${MODEL_PATH}, ${VAL_FILE}"
echo "Output: ${GOLDEN_JSON}"
echo "Device: ${DEVICE}"

python ./LingoQA_llava-v1.5-7B_profile.py \
    --model_path "${MODEL_PATH}" \
    --model_base "${MODEL_BASE}" \
    --val_file "${VAL_FILE}" \
    --data_dir "${DATA_DIR}" \
    --golden_json "${GOLDEN_JSON}" \
    --device "${DEVICE}"

echo "Step 1 done."

# =========================
# Step 2: Mapping - 生成层间映射数据
# =========================
echo "========== Step 2: Mapping =========="
echo "Input:  ${MODEL_PATH}, ${VAL_FILE}"
echo "Output: ${MAPPING_JSONL}"
echo "Device: ${DEVICE}"

python ./LingoQA_llava-v1.5-7B_mapping.py \
    --model_path "${MODEL_PATH}" \
    --model_base "${MODEL_BASE}" \
    --val_file "${VAL_FILE}" \
    --data_dir "${DATA_DIR}" \
    --save_dir "${MAPPING_DIR}" \
    --device "${DEVICE}"

echo "Step 2 done."

# =========================
# Step 3: Train mapping model - 训练映射模型
# =========================
echo "========== Step 3: Train Mapping Model =========="
echo "Input:  ${MAPPING_JSONL}"
echo "Output: ${MAPPING_MODEL}"
echo "Device: ${DEVICE}"

python ./train_mapping_model.py \
    --jsonl_path "${MAPPING_JSONL}" \
    --save_best_path "${MAPPING_MODEL}" \
    --device "${DEVICE}"

echo "Step 3 done."

# =========================
# Step 4: Semantic detection - 语义检测
# =========================
echo "========== Step 4: Semantic Detection =========="
echo "Input:  ${MODEL_PATH}, ${VAL_FILE}, ${MAPPING_MODEL}, ${GOLDEN_JSON}"
echo "Output: ${OUTPUT_SEM_JSONL}"
echo "Device: ${DEVICE}"

python ./LingoQA_llava-v1.5-7B_sem.py \
    --model_path "${MODEL_PATH}" \
    --model_base "${MODEL_BASE}" \
    --val_file "${VAL_FILE}" \
    --data_dir "${DATA_DIR}" \
    --golden_json "${GOLDEN_JSON}" \
    --mapping_model "${MAPPING_MODEL}" \
    --output_jsonl "${OUTPUT_SEM_JSONL}" \
    --device "${DEVICE}"

echo "Step 4 done."
echo "========== All steps completed! =========="
