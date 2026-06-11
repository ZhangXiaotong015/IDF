#!/bin/bash

# 设置输入和输出目录
INPUT_DIR="/data/test_vedio"
OUTPUT_DIR="/data/output_vedio_V2"

# 设置模型配置
CONFIG="/scratch/IDF/configs/models/idfnet.yaml"
CHECKPOINT="LocalImageLogger/unpaired_filteredCardiacNeuro_TissueBranch_LossV3Plus_PostProcessV1_iters3/checkpoints/epoch=57-step=50000.ckpt"

# 设置设备 (cuda 或 cpu)
DEVICE="cuda"

# 运行 Python 脚本
python /scratch/IDF/dicom_denoise_pipeline.py \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --device "$DEVICE"
