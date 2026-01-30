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

## Phase 1: Environment Setup [done]

**Environments created:**
- `.venv_zea` — ZEA synthesis (JAX/CUDA)
- `.venv_joint` — Training/inference (PyTorch)

---

## Phase 2: Code Modifications (TF → PyTorch Port) [done]

**Key files:**
- `datasets.py` — `ZeaDataset` loads RF data as `(N, n_tx, H, W)`, adapts to any `n_tx`
- `utils/corruptors.py` — `HazeCorruptor` implements `y = x + γh`
- `configs/training/score_zea_{tissue,haze}.yaml` — Training configs
- `configs/inference/paper/zea_dehaze_pigdm.yaml` — PIGDM inference config

See **Port Status & Gap Analysis** section below for full details.

---

## Phase 3: Sanity Test the PyTorch Port [done]

**Scripts:**
- `joint_diffusion/test_sanity.py` — Tests imports, SDEs, layers, NCSNv2, loss+backward, sampling
- `joint_diffusion/test_training_loop.py` — Tests full training pipeline with synthetic data
- `joint_diffusion/slurm_test.sh` — SLURM wrapper to run both tests on A100

**To run:** `sbatch slurm_test.sh` → check `/scrfs/storage/tp030/home/f2f_ldm/slurm_logs/`

---

## Phase 4: Generate Synthetic Data [done]

**Scripts:**
- `reproduce_helpers/zea_synthesize_dataset.py` — Generates tissue and haze RF data using ZEA
- `reproduce_helpers/visualize_dataset.py` — B-mode visualization of generated data
- `reproduce_helpers/slurm_zea_synth.sh` — SLURM wrapper for full synthesis (1000 train, 100 val)

**To run:** `sbatch reproduce_helpers/slurm_zea_synth.sh`

**Output:**
- `data/zea_synth/tissue/{train,val}.npz` — Clean tissue RF data
- `data/zea_synth/haze/{train,val}.npz` — Haze RF data
- Shape: `(N, n_tx, n_ax, n_el)` = `(N, 3, 1024, 64)`, dtype: `float32`, key: `rf`

---

## Phase 5: Train Diffusion Models

> **Requires**: ZEA data from Phase 4

**Scripts:**
- `joint_diffusion/slurm_train_tissue.sh` — SLURM wrapper for tissue model
- `joint_diffusion/slurm_train_haze.sh` — SLURM wrapper for haze model
- `joint_diffusion/train.py` — Main training script

**Configs:**
- `configs/training/score_zea_tissue.yaml` — Tissue model (100 epochs, bs=8, lr=1e-4)
- `configs/training/score_zea_haze.yaml` — Haze model (same params)

**To train both models:**
```bash
cd /scrfs/storage/tp030/home/f2f_ldm/dehazing-diffusion/joint_diffusion

# Train tissue model
sbatch slurm_train_tissue.sh

# Train haze model (can run in parallel if GPU quota allows)
sbatch slurm_train_haze.sh
```

**Checkpoints:** Saved to `wandb/<run_id>/files/training_checkpoints/ckpt-<epoch>.pt`

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

## Port Status & Gap Analysis (2026-01-29)

> **⚠️ MAJOR UPDATE**: All TensorFlow dependencies have been removed. The codebase now runs on pure PyTorch.

---

### ✅ What Was Ported (Complete)

All files that previously imported TensorFlow have been rewritten. The following table shows what was changed:

| File | Status | What Changed |
|------|--------|--------------|
| `generators/SGM/sde_lib.py` | ✅ Ported (Session 1) | `VPSDE`, `VESDE`, `subVPSDE`, `simple` — pure `torch.*` |
| `generators/layers.py` | ✅ Ported (Session 1) | All layers are `nn.Module` subclasses |
| `generators/SGM/SGM.py` | ✅ Ported (Session 1) | `NCSNv2`, `ScoreNet` with `score_loss()` method |
| `generators/SGM/sampling.py` | ✅ Ported (Session 1) | `ScoreSampler`, predictors, correctors |
| `generators/SGM/guidance.py` | ✅ Ported (Session 1) | `PIGDM`, `DPS`, `Projection` using `torch.autograd.grad` |
| `utils/corruptors.py` | ✅ Ported (Session 1) | `GaussianCorruptor`, `CSCorruptor`, `HazeCorruptor` |
| `utils/utils.py` | ✅ Ported (Session 2) | Removed `tensorflow` import, kept PyTorch/numpy utilities |
| `utils/gpu_config.py` | ✅ Ported (Session 2) | Replaced `tf.config` with `torch.cuda` |
| `utils/signals.py` | ✅ Stubbed (Session 2) | Only used by legacy TF datasets, not ZEA |
| `datasets.py` | ✅ Ported (Session 2) | Removed TF datasets; `ZeaDataset` returns PyTorch DataLoader |
| `generators/models.py` | ✅ Ported (Session 2) | Removed TF/Keras; only `score` and `glow` paths remain |
| `utils/checkpoints.py` | ✅ Ported (Session 2) | Pure PyTorch `torch.save`/`torch.load` + `EMAHelper` class |
| `utils/callbacks.py` | ✅ Ported (Session 2) | Plain Python classes (not Keras Callbacks) |
| `train.py` | ✅ Rewritten (Session 2) | Standard PyTorch training loop with EMA, gradient clipping |
| `utils/inverse.py` | ✅ Ported (Session 2) | `SGMDenoiser`, `BM3DDenoiser`, `NLMDenoiser` — pure PyTorch |

**New files created:**
- `test_sanity.py` — Tests imports, SDEs, layers, NCSNv2, loss+backward, sampling (6 test groups)
- `test_training_loop.py` — Tests full training pipeline with synthetic data
- `slurm_train_tissue.sh` — SLURM job script for training tissue model
- `slurm_train_haze.sh` — SLURM job script for training haze model
- `slurm_test.sh` + `test_run.sh` — SLURM job scripts for testing

---

### 🧪 How to Verify the Port Works

Run these tests in order. Stop and debug if any fail.

**Test 1: Sanity tests (imports, shapes, loss)**
```bash
cd /scrfs/storage/tp030/home/f2f_ldm/dehazing-diffusion/joint_diffusion
source /scrfs/storage/tp030/home/f2f_ldm/.venv_joint/bin/activate
python test_sanity.py
```
Expected: All 6 test groups PASS.

**Test 2: Training loop (synthetic data, no real data needed)**
```bash
python test_training_loop.py
```
Expected: Completes 2 epochs, loss is finite, checkpoint saves/loads.

**Test 3: Full test via SLURM (on A100)**
```bash
sbatch slurm_test.sh
# Check output: /scrfs/storage/tp030/home/f2f_ldm/slurm_logs/<node>_<jobid>.out
```

---

### ⚠️ WARNINGS: Things That Could Break

#### 1. Dataset Shape Mismatch
**Symptom**: `RuntimeError: shape mismatch` or `Expected (B,C,H,W) got (B,H,W,C)`

**Cause**: PyTorch uses channel-first `(B, C, H, W)`, TensorFlow used channel-last `(B, H, W, C)`.

**Where to look**:
- `datasets.py` line 63-79: `ZeaDataset.__init__` — data should already be `(N, n_tx, H, W)`
- `generators/SGM/SGM.py` line 180-220: `ScoreNet.score_loss` — expects `(B, C, H, W)`
- `utils/corruptors.py`: All corruptors assume channel-first

**Fix**: If your NPZ data is `(N, H, W, n_tx)`, transpose it:
```python
data = data.transpose(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
```

---

#### 2. NaN/Inf Loss
**Symptom**: `loss = nan` or `loss = inf` during training

**Causes & where to look**:
1. **Sigma explosion** in VESDE — `sde_lib.py` line 85-100: Check `sigma_max` isn't too large
2. **Score explosion** — `SGM.py` line 200-210: `score = model(x) / std` — if `std ≈ 0`, division explodes
3. **Gradient explosion** — `train.py` uses `grad_clip=1.0` by default, increase if needed

**Fix**: Add gradient clipping or reduce learning rate:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

#### 3. EMA Not Working
**Symptom**: Model generates noise even after many epochs

**Cause**: EMA weights not being used for sampling/inference

**Where to look**:
- `utils/checkpoints.py` line 160-220: `EMAHelper` class
- `train.py` line 115: `ema.update(model)` must be called after each optimizer step
- Checkpoint must save both model and EMA: `ckpt_manager.save()` saves both

**Fix**: When loading for inference, apply EMA weights:
```python
ema.apply_shadow(model)  # Apply EMA weights before sampling
```

---

#### 4. PIGDM Guidance Not Working
**Symptom**: Guided sampling produces same result as unconditional

**Cause**: Gradient not flowing through guidance computation

**Where to look**:
- `generators/SGM/guidance.py` line 45-80: `PIGDM.guide()` uses `torch.autograd.grad`
- `generators/SGM/sampling.py` line 150-180: Predictor must have `requires_grad=True` on input

**Fix**: Ensure `x` has gradients enabled:
```python
x = x.requires_grad_(True)
```

---

#### 5. Checkpoint Loading Fails
**Symptom**: `KeyError` or `RuntimeError: Error(s) in loading state_dict`

**Cause**: Old TF checkpoints vs new PyTorch format

**Where to look**:
- `utils/checkpoints.py` line 85-110: `ModelCheckpoint.restore()`
- Old format: `torch.save(model.state_dict(), path)`
- New format: `torch.save({"model_state_dict": ..., "optimizer_state_dict": ..., "epoch": ...}, path)`

**Fix**: For old checkpoints:
```python
state = torch.load(path)
if "model_state_dict" in state:
    model.load_state_dict(state["model_state_dict"])
else:
    model.load_state_dict(state)  # Legacy format
```

---

#### 6. SLURM Job Fails Silently
**Symptom**: Job completes but output file is empty or has errors

**Where to look**:
- Check SLURM log: `/scrfs/storage/tp030/home/f2f_ldm/slurm_logs/<node>_<jobid>.out`
- Check if container exists: `ls -la $HOME/qwen3vl-cu128.sif`
- Check if venv exists: `ls -la /scrfs/storage/tp030/home/f2f_ldm/.venv_joint`

**Fix**: Run the inner script manually first:
```bash
source /scrfs/storage/tp030/home/f2f_ldm/.venv_joint/bin/activate
cd /scrfs/storage/tp030/home/f2f_ldm/dehazing-diffusion/joint_diffusion
python test_sanity.py  # or train.py, etc.
```

---

### 📁 Reference Files for Debugging

If results are wrong, compare against official PyTorch implementations:

| Ported File | Reference | Line-by-line check |
|---|---|---|
| `generators/SGM/sde_lib.py` | `reproduce_helpers/score_sde_pytorch/sde_lib.py` | `marginal_prob()`, `prior_sampling()`, `discretize()` |
| `generators/layers.py` | `reproduce_helpers/ncsnv2/models/layers.py` | `ResidualBlock`, `RefineBlock`, `ConvMeanPool` |
| `generators/SGM/SGM.py` | `reproduce_helpers/ncsnv2/models/ncsnv2.py` | Architecture, forward pass |
| `generators/SGM/SGM.py` (loss) | `reproduce_helpers/score_sde_pytorch/losses.py` | `get_sde_loss_fn()` |
| `generators/SGM/sampling.py` | `reproduce_helpers/score_sde_pytorch/sampling.py` | Predictor/corrector updates |
| `generators/SGM/guidance.py` | **No reference** — paper Algorithm 1 | TF version in original repo |

---

### 🔧 Key Design Decisions

These were intentional choices during porting:

1. **`NCSNv2(x)` takes only `x`, not `(x, t)`**
   - Time conditioning is in `ScoreNet.get_score(x, t)` which divides by `std(t)`
   - This matches the continuous-time SDE formulation from score_sde

2. **No `get_sde()` factory function**
   - Use SDE classes directly: `VESDE(sigma_min=..., sigma_max=..., N=...)`
   - `ScoreNet` auto-creates SDE from config

3. **EMA handled by `EMAHelper` class, not optimizer wrapper**
   - TF used `tfa.optimizers.MovingAverage`
   - PyTorch version uses explicit shadow parameter tracking
   - Call `ema.update(model)` after each optimizer step

4. **Callbacks are plain Python, not Keras Callbacks**
   - `on_epoch_end(epoch, logs)` interface preserved
   - But no automatic integration with `model.fit()`

5. **Legacy TF datasets removed**
   - MNIST, CelebA, TMNIST, SineNoise — all removed
   - Only ZEA datasets work now
   - If you need MNIST, use `torchvision.datasets.MNIST`

---

### 📋 Files That Were Intentionally Removed/Stubbed

| File/Function | Reason |
|---|---|
| `utils/signals.py` (most functions) | Only used by TF datasets, not ZEA |
| `generators/GAN.py` | GAN model not needed for diffusion |
| `generators/layers.py` (UNet, encoder, decoder) | Only NCSNv2 needed for score model |
| `datasets.py` (MNIST, CelebA, etc.) | Only ZEA datasets needed |
| `utils/utils.py` (`tf_expand_multiple_dims`, `random_augmentation`) | TF-specific utilities |
| `utils/utils.py` (`get_normalization_layer`) | TF Rescaling layer |

---

### 🚀 Next Steps

1. **Generate ZEA data** (if not done):
   ```bash
   sbatch /scrfs/storage/tp030/home/f2f_ldm/dehazing-diffusion/reproduce_helpers/slurm_zea_synth.sh
   ```

2. **Run training test** (synthetic data):
   ```bash
   sbatch slurm_test.sh
   ```

3. **Train tissue model** (requires ZEA data):
   ```bash
   sbatch slurm_train_tissue.sh
   ```

4. **Train haze model** (can run in parallel with tissue):
   ```bash
   sbatch slurm_train_haze.sh
   ```

---

## Troubleshooting

### GPU OOM during ZEA synthesis

The `simulate_rf` function memory scales with scatterer count. The fish phantom uses ~104 scatterers which is well within GPU limits. The haze phantom adds 30-60 more (total ~134-164), still safe.

If OOM occurs with larger phantoms:
- Reduce scatterer count
- Set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` to let JAX use more GPU memory
- Request a larger GPU in SLURM
