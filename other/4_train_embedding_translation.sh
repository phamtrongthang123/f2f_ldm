#!/bin/bash
set -euo pipefail

# Step 4: Train Embedding Translation (m_3 → m_1)
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
echo "Training ET for m_3 → m_1"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "=========================================="

# Create reversed feature directories (trainA=m_3, trainB=m_1)
echo "Creating reversed feature directories..."
mkdir -p data/ultrasound_dataset/features_m3_to_m1/trainA
mkdir -p data/ultrasound_dataset/features_m3_to_m1/trainB

# Copy m_3 features to trainA (source)
cp data/ultrasound_dataset/features/trainB/* data/ultrasound_dataset/features_m3_to_m1/trainA/

# Copy m_1 features to trainB (target)
cp data/ultrasound_dataset/features/trainA/* data/ultrasound_dataset/features_m3_to_m1/trainB/

echo "✓ Reversed features ready"
echo "  trainA (source): m_3"
echo "  trainB (target): m_1"

# Train CycleGAN
cd embedding_translation
python train.py \
  --dataroot="../data/ultrasound_dataset/features_m3_to_m1" \
  --feat_dim=384 \
  --batch_size=128 \
  --n_epochs=200 \
  --save_name="m3_to_m1" \
  --display_freq=100 \
  --print_freq=50

echo "=========================================="
echo "✓ ET training complete!"
echo "Model: embedding_translation/checkpoints/m3_to_m1.ckpt"
echo "Next: bash 5_run_inference.sh"
echo "=========================================="
