#!/bin/bash
set -euo pipefail

ROOT_DIR="/scrfs/storage/tp030/home/f2f_ldm"
SCRIPT_DIR="$ROOT_DIR/high-rate-3d"

source "$ROOT_DIR/.venv_zea/bin/activate"
cd "$SCRIPT_DIR"
export KERAS_BACKEND=jax
export ZEA_CACHE_DIR="$SCRIPT_DIR/cache"

mkdir -p outputs/eval

echo "=== Environment ==="
python --version
python -c "import jax; print(f'JAX: {jax.__version__}, Devices: {jax.devices()}')"

echo ""
echo "=== Evaluating Diffusion Model ==="

python eval_diffusion.py "$@"

echo ""
echo "=== Evaluation Complete ==="
