#!/bin/bash
#SBATCH --job-name=train-diffusion-policy
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/train_%j.out
#SBATCH --partition=agpu72
#SBATCH --constraint=1a100

set -euo pipefail

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Define project root and environment
PROJECT_ROOT=$(pwd)
CONDA_ENV_NAME="robodiff"
CONTAINER="$HOME/qwenvl-2.5-cu121.sif"

# Create output directories
mkdir -p slurm_logs

# Execute training within Apptainer
apptainer exec --nv --writable-tmpfs \
    --bind "$HOME:$HOME" \
    --bind /share/apps:/share/apps \
    "${CONTAINER}" bash -c "
source /share/apps/python/anaconda-3.14/etc/profile.d/conda.sh
cd ${PROJECT_ROOT}

# Try to activate the environment, install if it fails
if ! conda activate '${CONDA_ENV_NAME}' 2>/dev/null; then
    echo \"Environment '${CONDA_ENV_NAME}' not found. Installing from conda_environment.yaml...\"
    conda env create -f conda_environment.yaml
    conda activate '${CONDA_ENV_NAME}'
fi


echo '=== Launching Diffusion Policy Training ==='
python train.py \
    --config-dir=. \
    --config-name=image_pusht_diffusion_policy_cnn.yaml \
    training.seed=42 \
    training.device=cuda:0 \
    hydra.run.dir='data/outputs/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_\${name}_\${task_name}'
"

echo "Job finished at: $(date)"
