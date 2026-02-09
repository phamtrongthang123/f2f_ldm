#!/bin/bash
#SBATCH --job-name=precompute
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/precompute_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --partition=agpu
#SBATCH --constraint=1a100

set -euo pipefail

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Define project root and environment
PROJECT_ROOT=$(pwd)
CONDA_ENV_NAME="drifting"
CONTAINER="$HOME/qwenvl-2.5-cu121.sif"

# Create output directories
mkdir -p slurm_logs

# Default paths - adjust if needed
IMAGENET_DIR="${IMAGENET_DIR:-./dataset/imagenet}"
OUTPUT_DIR="${OUTPUT_DIR:-./dataset/imagenet_latents}"

echo "Project Root: $PROJECT_ROOT"
echo "ImageNet Dir: $IMAGENET_DIR"
echo "Output Dir: $OUTPUT_DIR"

apptainer exec --nv --writable-tmpfs \
    --bind /scrfs/storage/tp030/home:/scrfs/storage/tp030/home \
    --bind /home/tp030:/home/tp030 \
    --bind /share/apps:/share/apps \
    "${CONTAINER}" bash -c "
source /share/apps/python/anaconda-3.14/etc/profile.d/conda.sh
conda activate '${CONDA_ENV_NAME}'

cd ${PROJECT_ROOT}

echo '=== Pre-computing Latents ==='
python data/imagenet.py \
    --imagenet_dir '$IMAGENET_DIR' \
    --output_dir '$OUTPUT_DIR' \
    --batch_size 64 \
    --num_workers 16
"

echo "Job finished at: $(date)"
