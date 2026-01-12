#!/bin/bash
set -euo pipefail

# Step 3: Extract DINO features from m_1 and m_3
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
echo "Local feature extraction"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "=========================================="

python extract_features_ultrasound.py \
  --m1_dir data/ultrasound_dataset/train/m1 \
  --m3_dir data/ultrasound_dataset/train/m3 \
  --output_base data/ultrasound_dataset/features \
  --model dinov2

echo "✓ Features extracted to data/ultrasound_dataset/features/"
echo "Next: sbatch 4_train_embedding_translation.sh"
