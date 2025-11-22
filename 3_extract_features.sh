#!/bin/bash
# Step 3: Extract DINO features from m_1 and m_3

module load python/anaconda-3.14
conda activate f2fldm

python extract_features_ultrasound.py \
  --m1_dir data/ultrasound_dataset/train/m1 \
  --m3_dir data/ultrasound_dataset/train/m3 \
  --output_base data/ultrasound_dataset/features \
  --model dinov2

echo "✓ Features extracted to data/ultrasound_dataset/features/"
echo "Next: sbatch 4_train_embedding_translation.sh"
