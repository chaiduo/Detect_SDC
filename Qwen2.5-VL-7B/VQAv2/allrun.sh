#!/bin/bash
set -e

# =========================
# 公共路径配置
# =========================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

MODEL_PATH="/data01/cd_workspace/llm/Qwen2.5-VL-7B-Instruct"
PARQUET_PATH="/data01/cd_workspace/llm/VQAv2"
MAX_SAMPLES=5000
DEFAULT_PYTHON_BIN="${SCRIPT_DIR}/../.venv/bin/python"
if [[ -z "${PYTHON_BIN:-}" && -x "${DEFAULT_PYTHON_BIN}" ]]; then
    PYTHON_BIN="${DEFAULT_PYTHON_BIN}"
else
    PYTHON_BIN="${PYTHON_BIN:-python}"
fi

# 设备配置
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export HF_ENDPOINT="https://hf-mirror.com"
DEVICE="cuda:0"

# 中间文件路径
GOLDEN_JSON="./json/Golden_VQAv2_Qwen2.5-VL-7B_30_new.json"
MAPPING_DIR="./json"
MAPPING_JSONL="${MAPPING_DIR}/attn_proj_mapping_64_project.jsonl"
MAPPING_MODEL="./model/best_mapping_model.pt"
OUTPUT_SEM_JSONL="${MAPPING_DIR}/detect_VQAv2_Qwen_with_sem_project.jsonl"

# 确保输出目录存在
mkdir -p ./json
mkdir -p ./model
mkdir -p "${MAPPING_DIR}"

# =========================
# Step 1: Profile - 生成 golden 文件
# =========================
echo "========== Step 1: Profile =========="
echo "Input:  ${MODEL_PATH}, ${PARQUET_PATH}"
echo "Output: ${GOLDEN_JSON}"
echo "Device: ${DEVICE}"

"${PYTHON_BIN}" ./VQAv2_Qwen2.5-VL-7B_profile.py \
    --model_path "${MODEL_PATH}" \
    --parquet_path "${PARQUET_PATH}" \
    --golden_json "${GOLDEN_JSON}" \
    --device "${DEVICE}" \
    --max_samples "${MAX_SAMPLES}"

echo "Step 1 done."

# =========================
# Step 2: Mapping - 生成层间映射数据
# =========================
echo "========== Step 2: Mapping =========="
echo "Input:  ${MODEL_PATH}, ${PARQUET_PATH}"
echo "Output: ${MAPPING_JSONL}"
echo "Device: ${DEVICE}"

"${PYTHON_BIN}" ./VQAv2_Qwen2.5-VL-7B_mapping.py \
    --model_path "${MODEL_PATH}" \
    --parquet_path "${PARQUET_PATH}" \
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

"${PYTHON_BIN}" ./train_mapping_model.py \
    --jsonl_path "${MAPPING_JSONL}" \
    --save_best_path "${MAPPING_MODEL}" \
    --device "${DEVICE}"

echo "Step 3 done."

# =========================
# Step 4: Semantic detection - 语义检测
# =========================
echo "========== Step 4: Semantic Detection =========="
echo "Input:  ${MODEL_PATH}, ${PARQUET_PATH}, ${MAPPING_MODEL}, ${GOLDEN_JSON}"
echo "Output: ${OUTPUT_SEM_JSONL}"
echo "Device: ${DEVICE}"

"${PYTHON_BIN}" ./VQAv2_Qwen2.5-VL-7B_sem.py \
    --model_path "${MODEL_PATH}" \
    --parquet_path "${PARQUET_PATH}" \
    --golden_json "${GOLDEN_JSON}" \
    --mapping_model "${MAPPING_MODEL}" \
    --output_jsonl "${OUTPUT_SEM_JSONL}" \
    --device "${DEVICE}" \
    --max_samples "${MAX_SAMPLES}"

echo "Step 4 done."
echo "========== All steps completed! =========="
