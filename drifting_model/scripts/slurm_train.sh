#!/bin/bash
#SBATCH --job-name=drifting-imagenet
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=3-00:00:00
#SBATCH --output=/scrfs/storage/tp030/home/f2f_ldm/drifting_model/slurm_logs/train_%j.out
#SBATCH --gres=gpu:4
#SBATCH --partition=qgpu72
#SBATCH --constraint=4a100

set -euo pipefail

# Adjust paths as needed
CONFIG="${CONFIG:-configs/ablation_default.yaml}"
LATENT_DIR="${LATENT_DIR:-./data/imagenet_latents}"
IMAGENET_DIR="${IMAGENET_DIR:-/path/to/imagenet}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints}"

mkdir -p logs "$OUTPUT_DIR"

echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "Config: $CONFIG"
echo "Latent dir: $LATENT_DIR"
echo "Output dir: $OUTPUT_DIR"

# Launch with torchrun
srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$(hostname):29500 \
    train_imagenet.py \
    --config "$CONFIG" \
    --latent_dir "$LATENT_DIR" \
    --imagenet_dir "$IMAGENET_DIR" \
    --output_dir "$OUTPUT_DIR"
