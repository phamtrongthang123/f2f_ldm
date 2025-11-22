#!/bin/bash
#SBATCH --job-name=m3_to_m1_sdxl
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --partition=agpu
#SBATCH --constraint=public&1a100

# Step 2: Train SDXL with PyTorch Lightning (both m_1 and m_3)

mkdir -p slurm_logs

echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=========================================="

# Activate conda
module load python/anaconda-3.14
source ~/.bashrc
conda activate f2fldm

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
  --num_workers=4 \
  --seed=42

echo "=========================================="
echo "✓ SDXL training complete!"
echo "End: $(date)"
echo "Next: bash 3_extract_features.sh"
echo "=========================================="
