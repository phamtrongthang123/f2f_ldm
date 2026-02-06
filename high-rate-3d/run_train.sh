#!/bin/bash
set -euo pipefail

ROOT_DIR="/scrfs/storage/tp030/home/f2f_ldm"
SCRIPT_DIR="$ROOT_DIR/high-rate-3d"

source "$ROOT_DIR/.venv_zea/bin/activate"
cd "$SCRIPT_DIR"
export KERAS_BACKEND=jax
export ZEA_CACHE_DIR="$SCRIPT_DIR/cache"

# Ensure optax is available
echo "Installing optax..."
uv pip install optax
python -c "import optax; print(f'optax installed: {optax.__version__}')"

mkdir -p outputs/training

echo "=== Environment ==="
python --version
python -c "import jax; print(f'JAX: {jax.__version__}')"
python -c "import jax; print(f'Devices: {jax.devices()}')"
python -c "import keras; print(f'Keras: {keras.__version__}, backend: {keras.backend.backend()}')"
python -c "import optax; print(f'optax: {optax.__version__}')"

echo ""
echo "=== Training Diffusion Model ==="

# Pass through any CLI arguments
python train_diffusion.py "$@"

echo ""
echo "=== Training Complete ==="
echo "Outputs saved to: $SCRIPT_DIR/outputs/training/"
