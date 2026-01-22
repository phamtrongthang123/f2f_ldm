#!/bin/bash
# Train embedding translation CycleGAN for m_1 -> m_3

FEAT_DIM=384  # DINOv2 feature dimension
BATCH_SIZE=128
EPOCHS=200
OUTPUT_NAME="ultrasound_m1_to_m3_et"

# Make sure you have extracted features first!
# See extract_features_ultrasound.sh

echo "Training embedding translation (m_1 -> m_3)..."
python train.py \
  --dataroot="../data/ultrasound_dataset/features" \
  --feat_dim=$FEAT_DIM \
  --batch_size=$BATCH_SIZE \
  --n_epochs=$EPOCHS \
  --save_name=$OUTPUT_NAME \
  --display_freq=100 \
  --print_freq=50

echo "✓ Embedding translation training complete!"
echo "Checkpoint saved to: checkpoints/${OUTPUT_NAME}.ckpt"
