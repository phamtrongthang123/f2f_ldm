#!/bin/bash
#SBATCH --job-name=m3_to_m1_et
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --partition=gpu

# Step 4: Train Embedding Translation (m_3 → m_1)
# This swaps trainA and trainB so CycleGAN learns the reverse direction

mkdir -p slurm_logs

echo "=========================================="
echo "Training ET for m_3 → m_1"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

# Activate conda
module load python/anaconda-3.14
source ~/.bashrc
conda activate f2fldm

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
