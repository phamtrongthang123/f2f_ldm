#!/bin/bash
# Step 1: Prepare ultrasound dataset (m_3 → m_1)

python setup_ultrasound_dataset.py \
  --source_m1 data/20250814InvivoData/Videos/20250815_07_46_16_m_1 \
  --source_m3 data/20250814InvivoData/Videos/20250815_07_52_18_m_3 \
  --output data/ultrasound_dataset \
  --resolution 512

echo "✓ Data prepared in data/ultrasound_dataset/"
echo "Next: sbatch 2_train_sdxl.sh"
