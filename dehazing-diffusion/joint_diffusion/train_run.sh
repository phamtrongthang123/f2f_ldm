#!/bin/bash
set -euo pipefail

ROOT_DIR="/scrfs/storage/tp030/home/f2f_ldm"
SCRIPT_DIR="$ROOT_DIR/dehazing-diffusion/joint_diffusion"

source "$ROOT_DIR/.venv_joint/bin/activate"
cd "$SCRIPT_DIR"

echo "=== Environment ==="
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "=== Running training ==="
python train.py \
    -c configs/training/score_zea_tissue.yaml \
    --data_root "$ROOT_DIR/data"
