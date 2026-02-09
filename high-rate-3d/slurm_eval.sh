#!/bin/bash
#SBATCH --job-name=eval-diffusion
#SBATCH --time=0-02:00:00
#SBATCH --output=/scrfs/storage/tp030/home/f2f_ldm/slurm_logs/eval_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --partition=agpu06
#SBATCH --constraint=1a100

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"

mkdir -p /scrfs/storage/tp030/home/f2f_ldm/slurm_logs

SCRIPT_DIR="/scrfs/storage/tp030/home/f2f_ldm/high-rate-3d"

apptainer exec --nv --writable-tmpfs \
  --bind /scrfs/storage/tp030/home:/scrfs/storage/tp030/home \
  "$HOME/qwen3vl-cu128.sif" \
  bash "$SCRIPT_DIR/run_eval.sh" "$@"

echo "Job finished at: $(date)"
