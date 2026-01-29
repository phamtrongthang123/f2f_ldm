# Reproduction Plan: Dehazing Ultrasound using Diffusion Models

This plan outlines the steps to reproduce the paper method using synthetic ultrasound data generated with ZEA.

---

## Overview

**Goal**: Train two diffusion models (tissue + haze) and use joint posterior sampling to dehaze ultrasound images.

**Workflow**:
```
ZEA Synthesis → Train Models → Joint Inference → Evaluate
```

---

## Phase 1: Environment Setup

### Step 1.1: Rename folder (remove spaces) [done]
```bash
cd /home/tp030/f2f_ldm
mv "reproduce helpers" dehazing-diffusion/reproduce_helpers
```

### Step 1.2: Create ZEA environment [done]
```bash
cd /home/tp030/f2f_ldm
uv venv .venv_zea
uv pip install --python .venv_zea zea "jax[cuda12]" numpy
```

### Step 1.3: Create joint_diffusion environment [done]
```bash
cd /home/tp030/f2f_ldm
uv venv .venv_joint
uv pip install --python .venv_joint torch torchvision
uv pip install --python .venv_joint -r dehazing-diffusion/joint_diffusion/requirements/requirements_pytorch.txt
```

---

## Phase 2: Code Modifications (TF → PyTorch Port)

### Step 2.1: Add ZEA dataset loader to `datasets.py` [done]

**File**: `/home/tp030/f2f_ldm/dehazing-diffusion/joint_diffusion/datasets.py`

`ZeaDataset` loads stored RF data with shape `(N, n_tx, n_ax, n_el)` and uses all transmits as channels:

```python
data = np.load(npz_path)[npz_key].astype(np.float32)
# Stored shape: (N, n_tx, n_ax, n_el) — use all transmits as channels
# Already in (N, C, H, W) format where C=n_tx
```

`image_shape` is set dynamically from the data: `[n_tx, *image_size]`. This means changing `n_tx` in the synthesis script (e.g. from 3 to 15) requires no code changes downstream — the dataset loader, model `in_channels`, and everything else adapts automatically.

---

### Step 2.2: Add Haze Corruptor to `corruptors.py` [So you don't need it right now if you're still working on synthesis and training (Phases 4-5). You'll need it before running inference in Phase 6]

**File**: `/home/tp030/f2f_ldm/dehazing-diffusion/joint_diffusion/utils/corruptors.py`

**Action**: Add at the end of the file (after line 291):
```python
import torch

@register_corruptor(name="haze")
class HazeCorruptor(Corruptor):
    """Additive haze corruptor for ultrasound dehazing.

    Measurement model: y = x + gamma * h
    where x is clean tissue, h is haze, gamma is haze strength.
    """

    def __init__(self, config, train=True):
        super().__init__(config, dataset_name="zea_haze", model=True)
        self.load_corruptor_dataset(train)
        self.blend_factor = config.noise_stddev  # gamma (haze strength)
        self.noise_stddev = self.blend_factor

    def corrupt(self, images):
        """Add haze to clean tissue images."""
        batch_size = images.shape[0]
        self.noise = next(self.gen)[:batch_size]
        noisy_images = images + self.blend_factor * self.noise
        return noisy_images
```

---

### Step 2.3: Create training config for tissue model [done]

**File**: `/home/tp030/f2f_ldm/dehazing-diffusion/joint_diffusion/configs/training/score_zea_tissue.yaml`

Key parameters (from paper Section 3.2.4):
- `batch_size: 8`, `lr: 1e-4`, `epochs: 100`
- `channels: 32`, `kernel_size: 3` (NCSNv2)
- `image_size: [1024, 64]`, `dataset_name: zea_tissue`

---

### Step 2.4: Create training config for haze model [done]

**File**: `/home/tp030/f2f_ldm/dehazing-diffusion/joint_diffusion/configs/training/score_zea_haze.yaml`

Same as tissue config, only `dataset_name: zea_haze`.

---

### Step 2.5: Create inference config [done]

**File**: `/home/tp030/f2f_ldm/dehazing-diffusion/joint_diffusion/configs/inference/paper/zea_dehaze_pigdm.yaml`

Key parameters (from paper Section 3.2.4):
- `lambda_coeff: 0.5`, `kappa_coeff: 0.5` (paper recommends ~0.5 for both)
- `num_scales: 200` (T=200 diffusion steps)
- `guidance: pigdm`, `corruptor: haze`
- `run_id.sgm` and `sgm.corruptor_run_id` need to be filled after training

---

### Step 2.6: Shell scripts 

- `/home/tp030/f2f_ldm/dehazing-diffusion/reproduce_helpers/zea_synth_run.sh` - Runs synthesis then visualization
- `/home/tp030/f2f_ldm/dehazing-diffusion/reproduce_helpers/slurm_zea_synth.sh` - SLURM job submission

---

## Phase 3: Sanity Test the PyTorch Port

Run these tests **before** generating data or training. Each layer catches progressively deeper issues. Stop and fix before moving on if any layer fails.

### Step 3.1: Layer 1 — Import smoke test

Verify all ported modules import without errors.

```bash
cd /home/tp030/f2f_ldm
source .venv_joint/bin/activate
cd dehazing-diffusion/joint_diffusion
python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

# Core modules
from datasets import get_dataset
from generators.models import get_model
from generators.layers import ConvBlock, ResidualBlock, DownSample, UpSample
from generators.SGM.SGM import NCSNv2
from generators.SGM.sde_lib import get_sde
from generators.SGM.sampling import get_predictor, get_corrector
from utils.corruptors import get_corruptor
from utils.inverse import get_denoiser

print('All imports OK')
"
```

### Step 3.2: Layer 2 — Component shape tests

Verify each ported component produces correct tensor shapes.

```bash
cd /home/tp030/f2f_ldm/dehazing-diffusion/joint_diffusion
source /home/tp030/f2f_ldm/.venv_joint/bin/activate
python -c "
import torch
import numpy as np

# --- Test 1: Layers ---
from generators.layers import ConvBlock, ResidualBlock
x = torch.randn(2, 1, 128, 64)  # (B, C, H, W)
conv = ConvBlock(in_channels=1, out_channels=32, kernel_size=3)
out = conv(x)
assert out.shape == (2, 32, 128, 64), f'ConvBlock: expected (2,32,128,64), got {out.shape}'
print(f'ConvBlock OK: {x.shape} -> {out.shape}')

# --- Test 2: Score network forward pass ---
from generators.SGM.SGM import NCSNv2
from utils.utils import AttrDict
config = AttrDict({
    'channels': 32,
    'image_size': [128, 64],
    'num_scales': 10,
    'sigma': 25.0,
    'normalization': 'batch',
    'kernel_size': 3,
    'activation': 'relu',
    'drop_prob': None,
    'upmode': 'upconv',
    'embed_dim': 256,
})
model = NCSNv2(config)
x = torch.randn(2, 1, 128, 64)
t = torch.randint(0, 10, (2,))
score = model(x, t)
assert score.shape == x.shape, f'NCSNv2: expected {x.shape}, got {score.shape}'
print(f'NCSNv2 OK: input {x.shape} -> score {score.shape}')

# --- Test 3: SDE ---
from generators.SGM.sde_lib import get_sde
sde = get_sde(config)
t = torch.rand(2)
mean, std = sde.marginal_prob(x, t)
assert mean.shape == x.shape, f'SDE marginal_prob mean: expected {x.shape}, got {mean.shape}'
print(f'SDE OK: marginal_prob shapes correct')

print()
print('All component shape tests PASSED')
"
```

### Step 3.3: Layer 3 — Dataset loading round-trip

Verify the ZEA dataset loader returns correct shapes and value ranges. Requires a small synthesized dataset (generate with `zea_synth_run.sh` first).

```bash
cd /home/tp030/f2f_ldm
source .venv_joint/bin/activate
cd dehazing-diffusion/joint_diffusion
python -c "
import torch
from datasets import get_dataset
from utils.utils import AttrDict

for name in ['zea_tissue', 'zea_haze']:
    config = AttrDict({
        'dataset_name': name,
        'data_root': '../../data',
        'batch_size': 4,
        'image_range': [0, 1],
        'shuffle': True,
        'seed': 42,
        'npz_key': 'rf',
    })
    train, val = get_dataset(config)
    batch = next(iter(train))
    assert isinstance(batch, torch.Tensor), f'Expected torch.Tensor, got {type(batch)}'
    assert batch.shape[1] == 3, f'Expected channel dim=n_tx=3, got {batch.shape[1]}'
    assert batch.shape[2:] == (1024, 64), f'Expected spatial (1024,64), got {batch.shape[2:]}'
    print(f'{name}: batch={batch.shape}, range=[{batch.min():.3f}, {batch.max():.3f}] OK')

print()
print('Dataset loading tests PASSED')
"
```

### Step 3.4: Layer 4 — Training smoke test (2 steps)

Run the training loop for 2 steps on dummy data to verify the full pipeline works end-to-end.

```bash
cd /home/tp030/f2f_ldm
source .venv_joint/bin/activate
cd dehazing-diffusion/joint_diffusion

export WANDB_MODE=disabled
python train.py -c configs/training/score_zea_tissue.yaml \
  --data_root ../../data \
  --epochs 1 \
  --limit_n_samples 8 \
  --batch_size 2
```

**Expected**: Completes 1 epoch without errors. Loss value is printed and is finite (not NaN/Inf).

### Step 3.5: Layer 5 — Corruptor test

Verify the HazeCorruptor produces valid corrupted images.

```bash
cd /home/tp030/f2f_ldm/dehazing-diffusion/joint_diffusion
source /home/tp030/f2f_ldm/.venv_joint/bin/activate
python -c "
import torch
from utils.corruptors import get_corruptor
from utils.utils import AttrDict

config = AttrDict({
    'dataset_name': 'zea_tissue',
    'data_root': '../../data',
    'batch_size': 4,
    'image_range': [0, 1],
    'image_size': [128, 64],
    'color_mode': 'grayscale',
    'shuffle': True,
    'seed': 42,
    'npz_key': 'rf',
    'noise_stddev': 0.5,
    'corruptor': 'haze',
})

corruptor = get_corruptor(config, train=True)
clean = torch.randn(4, 1, 128, 64)
corrupted = corruptor.corrupt(clean)
assert corrupted.shape == clean.shape, f'Shape mismatch: {corrupted.shape} vs {clean.shape}'
assert not torch.isnan(corrupted).any(), 'NaN in corrupted output'
assert not torch.equal(clean, corrupted), 'Corrupted should differ from clean'
print(f'Corruptor OK: {clean.shape} -> {corrupted.shape}')
print(f'Clean range: [{clean.min():.3f}, {clean.max():.3f}]')
print(f'Corrupted range: [{corrupted.min():.3f}, {corrupted.max():.3f}]')
print()
print('Corruptor test PASSED')
"
```

---

## Phase 4: Generate Synthetic Data [done]

### Step 4.1: Test ZEA API compatibility [done]

```bash
cd /home/tp030/f2f_ldm
source .venv_zea/bin/activate
KERAS_BACKEND=jax python -c "
from zea.probes import Probe
from zea.scan import Scan
from zea.ops import Beamform
from zea.simulator import simulate_rf
from zea.beamform.delays import compute_t0_delays_planewave
print('All ZEA imports successful')
"
```

### Step 4.2: Synthesis script [done]

**File**: `/home/tp030/f2f_ldm/dehazing-diffusion/reproduce_helpers/zea_synthesize_dataset.py`

- Uses `phantoms.fish()` for tissue (same as simulation reference, ~104 scatterers)
- Haze adds 30-60 extra random scatterers with lower magnitudes (0.3-0.8)
- Stores raw float32 RF data from `simulate_rf`, no normalization
- Stores all transmits: shape `(N, n_tx, n_ax, n_el)` — e.g. `(N, 3, 1024, 64)` with default `n_tx=3`
- `n_tx` is configurable in the synthesis script; downstream code adapts automatically
- Output structure: `{output_root}/tissue/train.npz`, `val.npz`, `{output_root}/haze/train.npz`, `val.npz`
- NPZ key: `rf`

### Step 4.3: Run small test synthesis

```bash
cd /home/tp030/f2f_ldm
bash dehazing-diffusion/reproduce_helpers/zea_synth_run.sh
```

This runs 10 train + 2 val samples and then visualization.

### Step 4.4: Verify output

```bash
python -c "
import numpy as np
from pathlib import Path

root = Path('data/zea_synth_test')
for kind in ['tissue', 'haze']:
    train = np.load(root / kind / 'train.npz')['rf']
    val = np.load(root / kind / 'val.npz')['rf']
    print(f'{kind}: train={train.shape}, val={val.shape}, dtype={train.dtype}')
"
```

Expected output:
```
tissue: train=(10, 3, 1024, 64), val=(2, 3, 1024, 64), dtype=float32
haze: train=(10, 3, 1024, 64), val=(2, 3, 1024, 64), dtype=float32
```

### Step 4.5: Visualization script [done]

**File**: `/home/tp030/f2f_ldm/dehazing-diffusion/reproduce_helpers/visualize_dataset.py`

- Loads stored RF data from npz files
- Adds channel dim `[:, :, :, np.newaxis]` to get `(n_tx, n_ax, n_el, 1)`
- Feeds directly into `zea.Pipeline` for B-mode reconstruction (same flow as simulation reference)
- Shows tissue (top row) and haze (bottom row) side by side
- Saves to `{data_root}/visualization.png`

### Step 4.6: Run full synthesis

```bash
cd /home/tp030/f2f_ldm
source .venv_zea/bin/activate
KERAS_BACKEND=jax python dehazing-diffusion/reproduce_helpers/zea_synthesize_dataset.py \
  --output-root data/zea_synth \
  --n-train 1000 --n-val 100 \
  --seed 42
```

---

## Phase 5: Train Diffusion Models

### Step 5.1: Test dataset loading

```bash
cd /home/tp030/f2f_ldm
source .venv_joint/bin/activate
cd dehazing-diffusion/joint_diffusion
python -c "
from datasets import get_dataset
from utils.utils import AttrDict

config = AttrDict({
    'dataset_name': 'zea_tissue',
    'data_root': '../../data',
    'batch_size': 4,
    'image_range': [0, 1],
    'shuffle': True,
    'seed': 1234,
    'npz_key': 'rf',
})
train, val = get_dataset(config)
batch = next(iter(train))
print(f'Batch shape: {batch.shape}')
print(f'Value range: [{batch.min():.3f}, {batch.max():.3f}]')
"
```

### Step 5.2: Train tissue model

```bash
cd /home/tp030/f2f_ldm
source .venv_joint/bin/activate
cd dehazing-diffusion/joint_diffusion

export WANDB_MODE=offline
python train.py -c configs/training/score_zea_tissue.yaml \
  --data_root ../../data
```

### Step 5.3: Train haze model

```bash
python train.py -c configs/training/score_zea_haze.yaml \
  --data_root ../../data
```

---

## Phase 6: Run Joint Inference

### Step 6.1: Update inference config with trained model paths

```bash
cd /home/tp030/f2f_ldm
source .venv_joint/bin/activate
python reproduce_helpers/update_zea_inference_config.py \
  --wandb-dir dehazing-diffusion/joint_diffusion/wandb \
  --inference-config dehazing-diffusion/joint_diffusion/configs/inference/paper/zea_dehaze_pigdm.yaml
```

### Step 6.2: Run dehazing inference

```bash
cd dehazing-diffusion/joint_diffusion
python inference.py -e paper/zea_dehaze_pigdm -t denoise -m sgm \
  --data_root ../../data
```

---

## Phase 7: Evaluate Results

### Step 7.1: Check output files

Results will be saved in `dehazing-diffusion/joint_diffusion/results/`.

### Step 7.2: Compute metrics

Use the gCNR (generalized contrast-to-noise ratio) from `processing.py`:
```python
from processing import gcnr

# Compare dehazed vs original
score = gcnr(dehazed_region, background_region)
```

---

## Troubleshooting

### GPU OOM during ZEA synthesis

The `simulate_rf` function memory scales with scatterer count. The fish phantom uses ~104 scatterers which is well within GPU limits. The haze phantom adds 30-60 more (total ~134-164), still safe.

If OOM occurs with larger phantoms:
- Reduce scatterer count
- Set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` to let JAX use more GPU memory
- Request a larger GPU in SLURM
