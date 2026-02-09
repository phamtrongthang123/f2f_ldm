#!/bin/bash
#SBATCH --job-name=train_mae
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm_logs/mae_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --partition=qgpu72
#SBATCH --constraint=4a100

# set -euo pipefail

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Define project root and environment
PROJECT_ROOT=$(pwd)
CONDA_ENV_NAME="drifting"
CONTAINER="$HOME/qwenvl-2.5-cu121.sif"

# Default paths
LATENT_DIR="${LATENT_DIR:-./dataset/imagenet_latents}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/mae}"
EPOCHS="${EPOCHS:-192}"
BASE_WIDTH="${BASE_WIDTH:-256}"

mkdir -p slurm_logs "$OUTPUT_DIR"

echo "Latent Dir: $LATENT_DIR"
echo "Output Dir: $OUTPUT_DIR"
echo "Epochs: $EPOCHS"

apptainer exec --nv --writable-tmpfs \
    --bind /scrfs/storage/tp030/home:/scrfs/storage/tp030/home \
    --bind /home/tp030:/home/tp030 \
    --bind /share/apps:/share/apps \
    "${CONTAINER}" bash -c "
source /share/apps/python/anaconda-3.14/etc/profile.d/conda.sh
conda activate '${CONDA_ENV_NAME}'
export PYTHONUNBUFFERED=1
export TORCH_DISTRIBUTED_DEBUG=INFO
cd ${PROJECT_ROOT}
echo \"Current Python: \$(which python)\"
echo \"Current Torchrun: \$(which torchrun)\"
python -c \"import torch; print(f'Torch available: {torch.cuda.is_available()}')\" || echo \"PYTHON IMPORT FAILED\"

echo '=== Training Latent-MAE (Multi-GPU DDP) ==='
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    train_mae.py \
    --latent_dir '$LATENT_DIR' \
    --output_dir '$OUTPUT_DIR' \
    --base_width '$BASE_WIDTH' \
    --epochs '$EPOCHS' \
    --effective_batch 8192 \
    --save_every 20
"

echo "Job finished at: $(date)"