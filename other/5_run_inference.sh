#!/bin/bash
set -euo pipefail

# Step 5: Run inference (m_3 → m_1 translation)
# Local run (single GPU) on device 4.

export CUDA_VISIBLE_DEVICES=4

if command -v module >/dev/null 2>&1; then
  module load python/anaconda-3.14
fi

if [ -f "$HOME/.bashrc" ]; then
  source "$HOME/.bashrc"
fi

if command -v conda >/dev/null 2>&1; then
  conda activate f2fldm
fi

echo "=========================================="
echo "Local inference"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "=========================================="

# Prepare test set (copy some m_3 images)
echo "Preparing test set..."
mkdir -p data/ultrasound_dataset/test/m3
cp data/ultrasound_dataset/train/m3/m3_frame_0001.png data/ultrasound_dataset/test/m3/
cp data/ultrasound_dataset/train/m3/m3_frame_0050.png data/ultrasound_dataset/test/m3/
cp data/ultrasound_dataset/train/m3/m3_frame_0100.png data/ultrasound_dataset/test/m3/
cp data/ultrasound_dataset/train/m3/m3_frame_0200.png data/ultrasound_dataset/test/m3/

# Configuration
INPUT_DIR="data/ultrasound_dataset/test/m3"
OUTPUT_DIR="results/m3_to_m1"
SD_MODEL="stabilityai/stable-diffusion-xl-base-1.0"
LORA_PATH="checkpoints/m3_to_m1"
ET_MODEL="embedding_translation/checkpoints/m3_to_m1.ckpt"

# Translation parameters
HIGH_NOISE_FRAC=0.5
GUIDANCE_SCALE=12.0
ET_WEIGHT=0.25
REG="l0"

echo "Running m_3 → m_1 translation..."
python run.py \
  --input_path="$INPUT_DIR" \
  --output_path="$OUTPUT_DIR" \
  --sd_model_id="$SD_MODEL" \
  --lora_model_path="$LORA_PATH" \
  --high_noise_frac=$HIGH_NOISE_FRAC \
  --target_resolution=512 \
  --unet_addition_embed_type="text_latent_addembeddingft" \
  --unet_add_embedding_input=4096 \
  --unet_add_embedding_output=1280 \
  --guidance_scale=$GUIDANCE_SCALE \
  --feat_model_name="DINOv2" \
  --et_model_name="cyclegan" \
  --et_model_path="$ET_MODEL" \
  --reg="$REG" \
  --et_weight=$ET_WEIGHT

echo "✓ Translation complete!"
echo "Results: $OUTPUT_DIR"
