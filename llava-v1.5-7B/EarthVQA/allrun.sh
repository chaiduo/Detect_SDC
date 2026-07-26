#!/bin/bash
set -e

# =========================
# 公共路径配置
# =========================
MODEL_PATH="/data01/cd_workspace/llm/llava-v1.5-7b"
MODEL_BASE=""
DATASET_DIR="/data01/cd_workspace/llm/EarthVQA/Train/images_png"
DATASET_JSON="/data01/cd_workspace/llm/EarthVQA/Train_QA.json"
MAX_SAMPLES=5000

# 设备配置
export CUDA_VISIBLE_DEVICES="1"
export HF_ENDPOINT="https://hf-mirror.com"
DEVICE="cuda:0"

# 中间文件路径
GOLDEN_JSON="./json/Golden_EarthVQA_llava-v1.5-7B.json"
MAPPING_DIR="./json"
MAPPING_JSONL="${MAPPING_DIR}/attn_proj_mapping_64_project.jsonl"
MAPPING_MODEL="./model/best_mapping_model.pt"
OUTPUT_SEM_JSONL="${MAPPING_DIR}/detect_EarthVQA_llava_with_sem_project.jsonl"

# 确保输出目录存在
mkdir -p ./json
mkdir -p ./model
mkdir -p "${MAPPING_DIR}"

# =========================
# Step 1: Profile - 生成 golden 文件
# =========================
echo "========== Step 1: Profile =========="
echo "Input:  ${MODEL_PATH}, ${DATASET_JSON}, ${DATASET_DIR}"
echo "Output: ${GOLDEN_JSON}"
echo "Device: ${DEVICE}"

python ./EarthVQA_llava-v1.5-7B_profile.py \
    --model_path "${MODEL_PATH}" \
    --model_base "${MODEL_BASE}" \
    --data_dir "${DATASET_DIR}" \
    --dataset_json "${DATASET_JSON}" \
    --golden_json "${GOLDEN_JSON}" \
    --device "${DEVICE}" \
    --max_samples "${MAX_SAMPLES}"

echo "Step 1 done."

# =========================
# Step 2: Mapping - 生成层间映射数据
# =========================
echo "========== Step 2: Mapping =========="
echo "Input:  ${MODEL_PATH}, ${DATASET_JSON}, ${DATASET_DIR}"
echo "Output: ${MAPPING_JSONL}"
echo "Device: ${DEVICE}"

python ./EarthVQA_llava-v1.5-7B_mapping.py \
    --model_path "${MODEL_PATH}" \
    --model_base "${MODEL_BASE}" \
    --data_dir "${DATASET_DIR}" \
    --dataset_json "${DATASET_JSON}" \
    --output_jsonl "${MAPPING_JSONL}" \
    --device "${DEVICE}" \
    --max_samples "${MAX_SAMPLES}"

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
echo "Input:  ${MODEL_PATH}, ${DATASET_JSON}, ${DATASET_DIR}, ${MAPPING_MODEL}, ${GOLDEN_JSON}"
echo "Output: ${OUTPUT_SEM_JSONL}"
echo "Device: ${DEVICE}"

python ./EarthVQA_llava-v1.5-7B_sem.py \
    --model_path "${MODEL_PATH}" \
    --model_base "${MODEL_BASE}" \
    --data_dir "${DATASET_DIR}" \
    --dataset_json "${DATASET_JSON}" \
    --golden_json "${GOLDEN_JSON}" \
    --mapping_model "${MAPPING_MODEL}" \
    --output_jsonl "${OUTPUT_SEM_JSONL}" \
    --device "${DEVICE}" \
    --max_samples "${MAX_SAMPLES}"

echo "Step 4 done."
echo "========== All steps completed! =========="
