#!/bin/bash
set -euo pipefail

# Step 2: Train SDXL with PyTorch Lightning (both m_1 and m_3)
# Local run (single GPU) on device 4.

export CUDA_VISIBLE_DEVICES=0

module load python/anaconda-3.14
# if command -v module >/dev/null 2>&1; then
# fi

# if [ -f "$HOME/.bashrc" ]; then
#   source "$HOME/.bashrc"
# fi

if command -v conda >/dev/null 2>&1; then
  conda activate f2fldm
fi

echo "=========================================="
echo "Local SDXL training" 
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Start: $(date)"
echo "=========================================="

# Training config
DATASET_DIR="data/ultrasound_dataset/train"
OUTPUT_DIR="checkpoints/m3_to_m1"

python train_pl.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --dataset_dir="$DATASET_DIR" \
  --train_data_dir="$DATASET_DIR/metadata_combined.csv" \
  --output_dir="$OUTPUT_DIR" \
  --resolution=512 \
  --batch_size=2 \
  --accumulate_grad_batches=4 \
  --learning_rate=1e-4 \
  --max_steps=100000 \
  --val_check_interval=5000 \
  --rank=8 \
  --gpus=1 \
  --precision="bf16-mixed" \
  --num_workers=0 \
  --seed=42 \
  --stripe_weight=0.2 \
  --stripe_kernel_size=15

echo "=========================================="
echo "✓ SDXL training complete!"
echo "End: $(date)"
echo "Next: bash 3_extract_features.sh"
echo "=========================================="
