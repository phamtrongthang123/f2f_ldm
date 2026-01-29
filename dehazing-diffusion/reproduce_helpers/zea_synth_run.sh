#!/bin/bash
set -euo pipefail

export KERAS_BACKEND=jax

ROOT_DIR="/scrfs/storage/tp030/home/f2f_ldm"
SCRIPT_DIR="$ROOT_DIR/dehazing-diffusion/reproduce_helpers"

source "$ROOT_DIR/.venv_zea/bin/activate"
cd "$ROOT_DIR"

echo "=== Testing ZEA imports ==="
python -c "
from zea.probes import Probe
from zea.scan import Scan
from zea.ops import Beamform
from zea.simulator import simulate_rf
from zea.beamform.delays import compute_t0_delays_planewave
print('All ZEA imports successful')
"

echo "=== Running small test synthesis (10 train, 2 val) ==="
python "$SCRIPT_DIR/zea_synthesize_dataset.py" \
  --output-root "$ROOT_DIR/data/zea_synth_test" \
  --n-train 10 --n-val 2 \
  --seed 123

echo "=== Verifying test output ==="
python -c "
import numpy as np
from pathlib import Path
root = Path('$ROOT_DIR/data/zea_synth_test')
for kind in ['tissue', 'haze']:
    train = np.load(root / kind / 'train.npz')['rf']
    val = np.load(root / kind / 'val.npz')['rf']
    print(f'{kind}: train={train.shape}, val={val.shape}, range=[{train.min():.1f}, {train.max():.1f}]')
"

echo "=== Running full synthesis (150 train, 38 val) ==="
python "$SCRIPT_DIR/zea_synthesize_dataset.py" \
  --output-root "$ROOT_DIR/data/zea_synth" \
  --n-train 150 --n-val 38 \
  --seed 123

echo "=== Verifying full output ==="
python -c "
import numpy as np
from pathlib import Path
root = Path('$ROOT_DIR/data/zea_synth')
for kind in ['tissue', 'haze']:
    train = np.load(root / kind / 'train.npz')['rf']
    val = np.load(root / kind / 'val.npz')['rf']
    print(f'{kind}: train={train.shape}, val={val.shape}, range=[{train.min():.1f}, {train.max():.1f}]')
"

rm -rf "$ROOT_DIR/data/zea_synth_test"
echo "=== Done ==="
