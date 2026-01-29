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
mv "reproduce helpers" reproduce_helpers
```

### Step 1.2: Create ZEA environment
```bash
cd /home/tp030/f2f_ldm
python -m venv .venv_zea
source .venv_zea/bin/activate
pip install --upgrade pip
pip install zea tensorflow numpy
```

### Step 1.3: Verify ZEA installation
```bash
source .venv_zea/bin/activate
python -c "import zea; print('ZEA installed successfully')"
```

### Step 1.4: Create joint_diffusion environment
```bash
cd /home/tp030/f2f_ldm
python -m venv .venv_joint
source .venv_joint/bin/activate
pip install --upgrade pip
pip install "tensorflow<2.11"
pip install -r dehazing-diffusion/joint_diffusion/requirements/requirements.txt
```

---

## Phase 2: Code Modifications

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
def _get_zea_dataset(config, kind: str):
    """Load ZEA synthetic RF dataset (tissue or haze).

    Args:
        config: Configuration dict with data_root, batch_size, etc.
        kind: Either "tissue" or "haze"

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    data_root = Path(config.data_root)
    npz_key = config.get("npz_key", "rf")
    image_range = config.get("image_range", [0, 1])
    batch_size = config.get("batch_size")
    shuffle = config.get("shuffle", True)
    seed = config.get("seed", None)

    train_path = data_root / "zea_synth" / kind / "train.npz"
    val_path = data_root / "zea_synth" / kind / "val.npz"

    if not train_path.exists():
        raise FileNotFoundError(f"ZEA dataset not found: {train_path}")

    train_data = np.load(train_path)[npz_key].astype(np.float32) / 255.0
    val_data = np.load(val_path)[npz_key].astype(np.float32) / 255.0

    # Add channel dim: (N, H, W) -> (N, H, W, 1)
    train_data = train_data[..., np.newaxis]
    val_data = val_data[..., np.newaxis]

    train_ds = tf.data.Dataset.from_tensor_slices(train_data)
    val_ds = tf.data.Dataset.from_tensor_slices(val_data)

    if shuffle:
        train_ds = train_ds.shuffle(len(train_data), seed=seed)

    if config.get("limit_n_samples"):
        train_ds = train_ds.take(config.limit_n_samples)
        val_ds = val_ds.take(config.limit_n_samples)

    print(f"Using {len(train_data)} files for training.")
    print(f"Using {len(val_data)} files for validation.")

    if batch_size:
        train_ds = train_ds.batch(batch_size)
        val_ds = val_ds.batch(batch_size)

    # Normalize to image_range
    norm_layer = get_normalization_layer(*image_range)
    train_ds = train_ds.map(norm_layer, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(norm_layer, num_parallel_calls=AUTOTUNE)

    return train_ds, val_ds
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
        batch_size = tf.shape(images)[0]
        self.noise = tf.gather(next(self.gen), tf.range(batch_size))
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

## Phase 3: Generate Synthetic Data

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

## Phase 4: Train Diffusion Models

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
print(f'Value range: [{batch.numpy().min():.3f}, {batch.numpy().max():.3f}]')
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

## Phase 5: Run Joint Inference

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

## Phase 6: Evaluate Results

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
- Check ZEA version: `pip show zea`
- The API might differ between versions
- Consult ZEA documentation for your version

### CUDA/GPU errors
- Ensure TensorFlow sees GPU: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`
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
