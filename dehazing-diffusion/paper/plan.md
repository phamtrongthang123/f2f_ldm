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
uv pip install --python .venv_zea zea torch numpy
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

### Step 2.1: Add ZEA dataset loader to `datasets.py`

**File**: `/home/tp030/f2f_ldm/dehazing-diffusion/joint_diffusion/datasets.py`

**Action**: Add `zea_tissue` and `zea_haze` to `_DATASETS` list (line 18):
```python
_DATASETS = [
    "mnist",
    "celeba",
    "sinenoise",
    "sinenoise1d",
    "tmnist",
    "zea_tissue",
    "zea_haze",
]
```

**Action**: Add the dataset loader function (after line 453):
```python
import torch
from torch.utils.data import Dataset, DataLoader

class ZeaDataset(Dataset):
    """PyTorch Dataset for ZEA synthetic RF data (tissue or haze)."""

    def __init__(self, npz_path, npz_key="rf", image_range=(0, 1), limit_n=None):
        data = np.load(npz_path)[npz_key].astype(np.float32) / 255.0
        # Add channel dim: (N, H, W) -> (N, 1, H, W)
        self.data = data[:, np.newaxis, :, :]
        if limit_n:
            self.data = self.data[:limit_n]
        # Normalize to image_range
        lo, hi = image_range
        self.data = self.data * (hi - lo) + lo
        self.data = torch.from_numpy(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def _get_zea_dataset(config, kind: str):
    """Load ZEA synthetic RF dataset (tissue or haze).

    Returns:
        Tuple of (train_loader, val_loader)
    """
    data_root = Path(config.data_root)
    npz_key = config.get("npz_key", "rf")
    image_range = config.get("image_range", [0, 1])
    batch_size = config.get("batch_size", 16)
    shuffle = config.get("shuffle", True)
    seed = config.get("seed", None)
    limit_n = config.get("limit_n_samples", None)

    train_path = data_root / "zea_synth" / kind / "train.npz"
    val_path = data_root / "zea_synth" / kind / "val.npz"

    if not train_path.exists():
        raise FileNotFoundError(f"ZEA dataset not found: {train_path}")

    train_ds = ZeaDataset(train_path, npz_key, image_range, limit_n)
    val_ds = ZeaDataset(val_path, npz_key, image_range, limit_n)

    print(f"Using {len(train_ds)} files for training.")
    print(f"Using {len(val_ds)} files for validation.")

    g = torch.Generator()
    if seed:
        g.manual_seed(seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
```

**Action**: Update `get_dataset()` function (around line 56, add before the `datasets = train, test` line):
```python
    if dataset_name.lower() == "zea_tissue":
        train, test = _get_zea_dataset(config, "tissue")
    if dataset_name.lower() == "zea_haze":
        train, test = _get_zea_dataset(config, "haze")
```

---

### Step 2.2: Add Haze Corruptor to `corruptors.py`

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

### Step 2.3: Create training config for tissue model

**File**: `/home/tp030/f2f_ldm/dehazing-diffusion/joint_diffusion/configs/training/score_zea_tissue.yaml`

**Action**: Create new file with content:
```yaml
# Training parameters
epochs:
  desc: Number of epochs to train
  value: 100
batch_size:
  desc: Size of each mini-batch
  value: 16
seed:
  desc: Random seed
  value: 1234
lr:
  desc: Learning rate
  value: 0.0002
ema:
  desc: Exponential moving average of weights
  value: 0.99999
embed_dim:
  desc: Size of embedded time vector
  value: 256
save_freq:
  desc: Save frequency (epochs)
  value: 10
eval_freq:
  desc: Evaluation frequency (epochs)
  value: 5
num_img:
  desc: Number of images to plot
  value: 16

# Model parameters
model_name:
  desc: Name of the generative model
  value: score
score_backbone:
  desc: Backbone architecture
  value: NCSNv2
sde:
  desc: SDE type
  value: simple
sigma:
  desc: Sigma value in SDE
  value: 25.0
num_scales:
  desc: Number of noise scales
  value: 1000
reduce_mean:
  value: false
likelihood_weighting:
  value: false
normalization:
  desc: Normalization type
  value: batch
channels:
  desc: Number of channels
  value: 32
kernel_size:
  value: 3
upmode:
  value: upconv
activation:
  value: relu
drop_prob:
  value: null

# Sampling parameters
sampling_method:
  value: pc
predictor:
  value: euler_maruyama
corrector:
  value: none
n_steps_each:
  value: 1
noise_removal:
  value: true
probability_flow:
  value: false
snr:
  value: 0.17

# Data parameters
data_root:
  desc: Path to data root
  value: null
dataset_name:
  desc: Dataset name
  value: zea_tissue
image_size:
  desc: Image size [axial, lateral]
  value: [128, 64]
image_range:
  value: [0, 1]
color_mode:
  value: grayscale
npz_key:
  desc: Key in NPZ file
  value: rf
```

---

### Step 2.4: Create training config for haze model

**File**: `/home/tp030/f2f_ldm/dehazing-diffusion/joint_diffusion/configs/training/score_zea_haze.yaml`

**Action**: Copy `score_zea_tissue.yaml` and change only:
```yaml
dataset_name:
  desc: Dataset name
  value: zea_haze
```

---

### Step 2.5: Create inference config

**File**: `/home/tp030/f2f_ldm/dehazing-diffusion/joint_diffusion/configs/inference/paper/zea_dehaze_pigdm.yaml`

**Action**: Create the `paper/` directory and file:
```yaml
# Inference config for ZEA dehazing with ΠGDM

run_id:
  sgm: null  # Filled by update_zea_inference_config.py

# Data parameters
data_root: null
batch_size: 8
image_size: [128, 64]
image_range: [0, 1]
color_mode: grayscale
npz_key: rf

# Corruption parameters
paired_data: true
corruptor: haze
noise_stddev: 0.5  # gamma (haze strength)

# SGM parameters
sgm:
  corruptor_run_id: null  # Filled by update_zea_inference_config.py
  n_steps: 1000
  snr: 0.17

# Denoiser
denoiser: sgm
```

---

### Step 2.6: Fix synthesis script paths

**File**: `/home/tp030/f2f_ldm/reproduce_helpers/zea_synthesize_dataset.py`

**Action**: Update lines 24-26 to handle the renamed folder:
```python
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "dehazing-diffusion"))
```

Verify this resolves to `/home/tp030/f2f_ldm/dehazing-diffusion`.

---

### Step 2.7: Update shell scripts for renamed folder

**File**: `/home/tp030/f2f_ldm/reproduce_helpers/run_zea_synthesis.sh`
```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$ROOT_DIR/.venv_zea/bin/activate"

python "$SCRIPT_DIR/zea_synthesize_dataset.py" \
  --output-root "$ROOT_DIR/data/zea_synth"
```

**File**: `/home/tp030/f2f_ldm/reproduce_helpers/train_zea_models.sh`
```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JD_DIR="$ROOT_DIR/dehazing-diffusion/joint_diffusion"

source "$ROOT_DIR/.venv_joint/bin/activate"

export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=${WANDB_DIR:-$JD_DIR/wandb}

pushd "$JD_DIR" >/dev/null
python train.py -c configs/training/score_zea_tissue.yaml --data_root "$ROOT_DIR/data"
python train.py -c configs/training/score_zea_haze.yaml --data_root "$ROOT_DIR/data"
popd >/dev/null
```

**File**: `/home/tp030/f2f_ldm/reproduce_helpers/run_zea_inference.sh`
```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JD_DIR="$ROOT_DIR/dehazing-diffusion/joint_diffusion"

source "$ROOT_DIR/.venv_joint/bin/activate"

pushd "$JD_DIR" >/dev/null
python inference.py -e paper/zea_dehaze_pigdm -t denoise -m sgm --data_root "$ROOT_DIR/data"
popd >/dev/null
```

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

Verify the ZEA dataset loader returns correct shapes and value ranges. Requires a small dummy dataset (create one inline).

```bash
cd /home/tp030/f2f_ldm
source .venv_joint/bin/activate
python -c "
import numpy as np
from pathlib import Path

# Create tiny dummy dataset for testing
dummy_root = Path('data/zea_synth_test')
for kind in ['tissue', 'haze']:
    (dummy_root / kind).mkdir(parents=True, exist_ok=True)
    train = np.random.randint(0, 256, (8, 128, 64), dtype=np.uint8)
    val = np.random.randint(0, 256, (2, 128, 64), dtype=np.uint8)
    np.savez(dummy_root / kind / 'train.npz', rf=train)
    np.savez(dummy_root / kind / 'val.npz', rf=val)
print('Dummy dataset created')
"

cd dehazing-diffusion/joint_diffusion
python -c "
import torch
from datasets import get_dataset
from utils.utils import AttrDict

for name in ['zea_tissue', 'zea_haze']:
    config = AttrDict({
        'dataset_name': name,
        'data_root': '../../data_test',
        'batch_size': 4,
        'image_range': [0, 1],
        'shuffle': True,
        'seed': 42,
        'npz_key': 'rf',
    })
    # Point to test data
    config.data_root = '../../data/zea_synth_test/..'
    train, val = get_dataset(config)
    batch = next(iter(train))
    assert isinstance(batch, torch.Tensor), f'Expected torch.Tensor, got {type(batch)}'
    assert batch.shape[1] == 1, f'Expected channel dim=1, got {batch.shape[1]}'
    assert batch.shape[2:] == (128, 64), f'Expected spatial (128,64), got {batch.shape[2:]}'
    assert 0.0 <= batch.min() <= batch.max() <= 1.0, f'Values out of [0,1]: [{batch.min():.3f}, {batch.max():.3f}]'
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
  --data_root ../../data/zea_synth_test/.. \
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
    'data_root': '../../data/zea_synth_test/..',
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

### Step 3.6: Layer 6 — Cleanup test data

```bash
rm -rf /home/tp030/f2f_ldm/data/zea_synth_test
```

---

## Phase 4: Generate Synthetic Data

### Step 3.1: Test ZEA API compatibility

```bash
cd /home/tp030/f2f_ldm
source .venv_zea/bin/activate
python -c "
from zea.probes import Probe
from zea.scan import Scan
from zea.ops import Beamform
from zea.simulator import simulate_rf
from zea.beamform.delays import compute_t0_delays_planewave
print('All ZEA imports successful')
"
```

If imports fail, check ZEA documentation for correct module paths.

### Step 3.2: Run synthesis (small test first)

```bash
cd /home/tp030/f2f_ldm
source .venv_zea/bin/activate
python reproduce_helpers/zea_synthesize_dataset.py \
  --output-root data/zea_synth \
  --n-train 10 --n-val 2 \
  --seed 123
```

### Step 3.3: Verify output

```bash
python -c "
import numpy as np
from pathlib import Path

root = Path('data/zea_synth')
for kind in ['tissue', 'haze']:
    train = np.load(root / kind / 'train.npz')['rf']
    val = np.load(root / kind / 'val.npz')['rf']
    print(f'{kind}: train={train.shape}, val={val.shape}, range=[{train.min():.1f}, {train.max():.1f}]')
"
```

Expected output:
```
tissue: train=(10, 128, 64), val=(2, 128, 64), range=[0.0, 255.0]
haze: train=(10, 128, 64), val=(2, 128, 64), range=[0.0, 255.0]
```

### Step 3.4: Run full synthesis

```bash
python reproduce_helpers/zea_synthesize_dataset.py \
  --output-root data/zea_synth \
  --n-train 150 --n-val 38 \
  --seed 123
```

---

## Phase 5: Train Diffusion Models

### Step 4.1: Test dataset loading

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

### Step 4.2: Train tissue model

```bash
cd /home/tp030/f2f_ldm
source .venv_joint/bin/activate
cd dehazing-diffusion/joint_diffusion

export WANDB_MODE=offline
python train.py -c configs/training/score_zea_tissue.yaml \
  --data_root ../../data
```

### Step 4.3: Train haze model

```bash
python train.py -c configs/training/score_zea_haze.yaml \
  --data_root ../../data
```

---

## Phase 6: Run Joint Inference

### Step 5.1: Update inference config with trained model paths

```bash
cd /home/tp030/f2f_ldm
source .venv_joint/bin/activate
python reproduce_helpers/update_zea_inference_config.py \
  --wandb-dir dehazing-diffusion/joint_diffusion/wandb \
  --inference-config dehazing-diffusion/joint_diffusion/configs/inference/paper/zea_dehaze_pigdm.yaml
```

### Step 5.2: Run dehazing inference

```bash
cd dehazing-diffusion/joint_diffusion
python inference.py -e paper/zea_dehaze_pigdm -t denoise -m sgm \
  --data_root ../../data
```

---

## Phase 7: Evaluate Results

### Step 6.1: Check output files

Results will be saved in `dehazing-diffusion/joint_diffusion/results/`.

### Step 6.2: Compute metrics

Use the gCNR (generalized contrast-to-noise ratio) from `processing.py`:
```python
from processing import gcnr

# Compare dehazed vs original
score = gcnr(dehazed_region, background_region)
```

---

## Troubleshooting

### ZEA import errors
- Check ZEA version: `uv pip show --python .venv_zea zea`
- The API might differ between versions
- Consult ZEA documentation for your version

### CUDA/GPU errors
- Ensure PyTorch sees GPU: `python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"`
- Set `CUDA_VISIBLE_DEVICES=0` if needed

### Memory errors during training
- Reduce `batch_size` in config
- Reduce `image_size` if needed
- Use `limit_n_samples` for quick tests

### wandb errors
- Use `export WANDB_MODE=offline` to avoid login requirements
- Check `WANDB_DIR` is writable

---

## File Checklist

After completing all steps, you should have:

- [ ] `/home/tp030/f2f_ldm/.venv_zea/` - ZEA virtual environment
- [ ] `/home/tp030/f2f_ldm/.venv_joint/` - Joint diffusion virtual environment
- [ ] `/home/tp030/f2f_ldm/reproduce_helpers/` - Renamed from "reproduce helpers"
- [ ] `/home/tp030/f2f_ldm/data/zea_synth/tissue/{train,val}.npz` - Tissue data
- [ ] `/home/tp030/f2f_ldm/data/zea_synth/haze/{train,val}.npz` - Haze data
- [ ] `/home/tp030/f2f_ldm/data/zea_synth/metadata.json` - Synthesis metadata
- [ ] `joint_diffusion/configs/training/score_zea_tissue.yaml` - Tissue training config
- [ ] `joint_diffusion/configs/training/score_zea_haze.yaml` - Haze training config
- [ ] `joint_diffusion/configs/inference/paper/zea_dehaze_pigdm.yaml` - Inference config
- [ ] Modified `joint_diffusion/datasets.py` - Added ZEA loaders
- [ ] Modified `joint_diffusion/utils/corruptors.py` - Added HazeCorruptor
