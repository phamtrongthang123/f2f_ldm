#!/bin/bash
set -euo pipefail

export KERAS_BACKEND=jax

ROOT_DIR="/scrfs/storage/tp030/home/f2f_ldm"
SCRIPT_DIR="$ROOT_DIR/dehazing-diffusion/reproduce_helpers"

source "$ROOT_DIR/.venv_zea/bin/activate"
cd "$ROOT_DIR"

# echo "=== Testing ZEA imports ==="
# python -c "
# from zea.probes import Probe
# from zea.scan import Scan
# from zea.ops import Beamform
# from zea.simulator import simulate_rf
# from zea.beamform.delays import compute_t0_delays_planewave
# print('All ZEA imports successful')
# "

# echo "=== Running small test synthesis (10 train, 2 val) ==="
# python "$SCRIPT_DIR/zea_synthesize_dataset.py" \
#     --output-root "$ROOT_DIR/data/zea_synth_test" \
#     --n-train 10 \
#     --n-val 2 \
#     --seed 42

# echo "=== Visualizing synthesized data ==="
# python "$SCRIPT_DIR/visualize_dataset.py" \
#     --data-root "$ROOT_DIR/data/zea_synth_test" \
#     --n-samples 4

# echo "=== Test synthesis complete ==="

# Uncomment below for full synthesis:
# echo "=== Running full synthesis (1000 train, 100 val) ==="
# python "$SCRIPT_DIR/zea_synthesize_dataset.py" \
#     --output-root "$ROOT_DIR/data/zea_synth" \
#     --n-train 1000 \
#     --n-val 100 \
#     --seed 42

python "$SCRIPT_DIR/visualize_dataset.py" \
    --data-root "$ROOT_DIR/data/zea_synth" \
    --n-samples 4