#!/bin/bash
#SBATCH --job-name=drifting-robotics
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/robotics_%j.out
#SBATCH --partition=agpu72
#SBATCH --constraint=1a100

set -euo pipefail

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Define project root and environment
PROJECT_ROOT=$(pwd)
CONDA_ENV_NAME="drifting_robotics"
CONTAINER="$HOME/qwenvl-2.5-cu121.sif"

# Create output directories
mkdir -p slurm_logs checkpoints/robotics

# Default paths and settings
TASK="${TASK:-pusht}"
if [ -d "./data/pusht/pusht_cchi_v7_replay.zarr" ]; then
    DATA_DIR="${DATA_DIR:-./data/pusht/pusht_cchi_v7_replay.zarr}"
else
    DATA_DIR="${DATA_DIR:-./data/pusht_cchi_v7_replay.zarr}"
fi
SAVE_DIR="${SAVE_DIR:-./checkpoints/robotics}"
EPOCHS="${EPOCHS:-3050}"

echo "Project Root: $PROJECT_ROOT"
echo "Task: $TASK"
echo "Data Dir: $DATA_DIR"
echo "Save Dir: $SAVE_DIR"
echo "Epochs: $EPOCHS"

# Execute training within Apptainer
apptainer exec --nv --writable-tmpfs \
    --bind "$HOME:$HOME" \
    --bind /share/apps:/share/apps \
    "${CONTAINER}" bash -c "
source /share/apps/python/anaconda-3.14/etc/profile.d/conda.sh
conda activate '${CONDA_ENV_NAME}'
python -c \"import torch; print('CUDA available:', torch.cuda.is_available())\"
if python -c \"import torch; exit(0 if torch.cuda.is_available() else 1)\"; then
    echo \"CUDA detected.\"
else
    echo \"CUDA NOT detected. Attempting hot-fix...\"
    pip uninstall torch torchvision -y 
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
fi

cd ${PROJECT_ROOT}

echo '=== Training Drifting Robotics Policy ==='
python train_robotics.py \
    --task '$TASK' \
    --data_dir '$DATA_DIR' \
    --save_dir '$SAVE_DIR' \
    --epochs $EPOCHS \
    --batch_size 256
"

echo "Job finished at: $(date)"
