#!/bin/bash
set -e  # 遇到错误立即退出

# =========================
# 公共路径配置
# =========================
MODEL_PATH="/data1/home/dataset_share/wsh_data/data/qwen/Qwen2___5-VL-7B-Instruct"
DATASET_DIR="/data0/home/lc/cd/llm/datasets/EarthVLSet/EarthVQA/Train/images_png"
DATASET_JSON="/data0/home/lc/cd/llm/datasets/EarthVLSet/EarthVQA/Train_QA.json"

# 设备配置
export CUDA_VISIBLE_DEVICES="6"
DEVICE="cuda:0"

# 中间文件路径
GOLDEN_JSON="./json/Golden_EarthVQA_Qwen2.5-VL-7B_30_CA.json"
MAPPING_DIR="/data1/home/dataset_share/cd_data/Qwen2.5-VL-7B/EarthVQA/final/"
MAPPING_JSONL="${MAPPING_DIR}attn_proj_mapping_64_project.jsonl"
MAPPING_MODEL="./model/best_mapping_model.pt"
OUTPUT_SEM_JSONL="${MAPPING_DIR}detect_EarthVQA_Qwen_with_sem_project.jsonl"

# 确保输出目录存在
mkdir -p ./json
mkdir -p ./model
mkdir -p ${MAPPING_DIR}

# =========================
# Step 1: Profile - 生成 golden 文件
# =========================
echo "========== Step 1: Profile =========="
echo "Input:  ${MODEL_PATH}, ${DATASET_JSON}"
echo "Output: ${GOLDEN_JSON}"
echo "Device: ${DEVICE}"

python ./EarthVQA_Qwen2.5-VL-7B_profile.py \
    --model_path "${MODEL_PATH}" \
    --data_dir "${DATASET_DIR}" \
    --dataset_json "${DATASET_JSON}" \
    --golden_json "${GOLDEN_JSON}" \
    --device "${DEVICE}"

echo "Step 1 done."

# =========================
# Step 2: Mapping - 生成层间映射数据
# =========================
echo "========== Step 2: Mapping =========="
echo "Input:  ${MODEL_PATH}, ${DATASET_JSON}"
echo "Output: ${MAPPING_JSONL}"
echo "Device: ${DEVICE}"

python ./EarthVQA_Qwen2.5-VL-7B_mapping.py \
    --model_path "${MODEL_PATH}" \
    --data_dir "${DATASET_DIR}" \
    --dataset_json "${DATASET_JSON}" \
    --output_jsonl "${MAPPING_JSONL}" \
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
echo "Input:  ${MODEL_PATH}, ${DATASET_JSON}, ${MAPPING_MODEL}, ${GOLDEN_JSON}"
echo "Output: ${OUTPUT_SEM_JSONL}"
echo "Device: ${DEVICE}"

python ./EarthVQA_Qwen2.5-VL-7B_sem.py \
    --model_path "${MODEL_PATH}" \
    --data_dir "${DATASET_DIR}" \
    --dataset_json "${DATASET_JSON}" \
    --golden_json "${GOLDEN_JSON}" \
    --mapping_model "${MAPPING_MODEL}" \
    --output_jsonl "${OUTPUT_SEM_JSONL}" \
    --device "${DEVICE}"

echo "Step 4 done."
echo "========== All steps completed! =========="
