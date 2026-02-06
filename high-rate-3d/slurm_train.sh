#!/bin/bash
#SBATCH --job-name=train-diffusion
#SBATCH --time=3-00:00:00
#SBATCH --output=/scrfs/storage/tp030/home/f2f_ldm/slurm_logs/train_diff_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --partition=agpu72
#SBATCH --constraint=1a100

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running: Diffusion Model Training"

mkdir -p /scrfs/storage/tp030/home/f2f_ldm/slurm_logs

SCRIPT_DIR="/scrfs/storage/tp030/home/f2f_ldm/high-rate-3d"

apptainer exec --nv --writable-tmpfs \
  --bind /scrfs/storage/tp030/home:/scrfs/storage/tp030/home \
  "$HOME/qwen3vl-cu128.sif" \
  bash "$SCRIPT_DIR/run_train.sh" "$@"

echo "Job finished at: $(date)"
