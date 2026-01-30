# Agent Prompt: Port Remaining TF Files to PyTorch

## Task Overview

Continue porting the TensorFlow-dependent files in `joint_diffusion/` to PyTorch so that `train.py` and `inference.py` can run without TensorFlow.

**Previous session** already ported the 6 core SGM files (all passing `test_sanity.py`). This session must port the remaining ~10 files that still import TensorFlow.

**Working directory**: `/scrfs/storage/tp030/home/f2f_ldm/dehazing-diffusion/joint_diffusion`

**Virtual environment**:
```bash
source /scrfs/storage/tp030/home/f2f_ldm/.venv_joint/bin/activate
```

**Full plan**: `/home/tp030/f2f_ldm/dehazing-diffusion/paper/plan.md`

---

## What's Already Done (DO NOT MODIFY)

These files were ported and pass all tests. Do not rewrite them:

| File | Status |
|------|--------|
| `generators/SGM/sde_lib.py` | Ported — `SDE`, `VPSDE`, `VESDE`, `subVPSDE`, `simple` |
| `generators/layers.py` | Ported — `ConvBlock`, `ResidualBlock`, `RCUBlock`, `MSFBlock`, `CRPBlock`, `RefineBlock`, `get_activation`, `get_normalization` |
| `generators/SGM/SGM.py` | Ported — `NCSNv2(nn.Module)`, `ScoreNet(nn.Module)` |
| `generators/SGM/sampling.py` | Ported — `ScoreSampler`, predictors, correctors |
| `generators/SGM/guidance.py` | Ported — `PIGDM`, `DPS`, `Projection` |
| `utils/corruptors.py` | Ported — `GaussianCorruptor`, `CSCorruptor`, `HazeCorruptor` |
| `generators/__init__.py` | Created |
| `generators/SGM/__init__.py` | Created |
| `utils/__init__.py` | Created |
| `test_sanity.py` | Created — 6 test groups, all PASS |
| `datasets.py` | Partially done — `ZeaDataset` class at bottom is PyTorch, but top-level imports are TF |

### Key Design Decisions Already Made

1. **Image format**: PyTorch `(B, C, H, W)` — channel first
2. **NCSNv2 signature**: `model(x)` takes only `x`, NOT `(x, t)`. Time conditioning is in `ScoreNet.get_score(x, t)` which divides by `std(t)`.
3. **Config format**: Uses attribute-style access (`config.channels`, `config.sde`). The codebase uses `easydict.EasyDict` or `wandb.config` objects.
4. **No `get_sde()` factory**: SDE classes are used directly. If you add a factory, put it in `sde_lib.py`.

---

## Files That Need Porting (Priority Order)

### Priority 1: Required for `train.py`

#### 1. `datasets.py`
**Current state**: Top-level imports `tensorflow`, `keras`. Bottom has PyTorch `ZeaDataset` (already working).
**Action**: Remove TF imports. Keep `ZeaDataset` and `_get_zea_dataset()`. The TF dataset functions (`_get_mnist`, `_get_celeba`, `_get_tmnist`, `_get_sine_noise_dataset`, `_get_sine_noise1D_dataset`) can be **removed or stubbed** — they are not needed for ZEA training. The `get_dataset()` dispatcher should still work for `zea_tissue` and `zea_haze`.
**Watch out**: `get_dataset()` currently calls `dataset.prefetch()` (TF) and checks `element_spec` (TF) for non-ZEA datasets. The ZEA path returns early before those lines, but the TF imports at the top will still fail.
**Imports to remove**: `tensorflow`, `keras.utils.to_categorical`, `sklearn.model_selection`, `utils.signals` (TF-dependent), `utils.utils.download_and_unpack`, `utils.utils.get_normalization_layer`

#### 2. `utils/utils.py`
**Current state**: Hybrid TF+PyTorch. Imports `tensorflow as tf` and `torch`.
**Action**: Remove all TF ops. Keep PyTorch and numpy utilities. Key functions needed downstream:
- `load_config_from_yaml(path)` → returns easydict. Currently uses `yaml.safe_load`. **No TF dependency** in this function itself.
- `save_dict_to_yaml(d, path)` → **No TF dependency**.
- `set_random_seed(seed)` → sets `np.random.seed`, `random.seed`, `torch.manual_seed`. Remove `tf.random.set_seed`.
- `check_model_library(model)` → returns `"pytorch"` if `torch.nn.Module`, else `"tensorflow"`. Remove TF branch or just always return `"pytorch"`.
- `add_args_to_config(args, config)` → merges argparse args into config dict. **No TF dependency**.
- `update_dict(base, update)` → recursive dict merge. **No TF dependency**.
- `tf_expand_multiple_dims` → **DELETE**, replaced by `while std.dim() < x.dim(): std = std.unsqueeze(-1)` in ported code.
- `tf_tensor_to_torch`, `convert_torch_tensor` → **DELETE** or simplify.
- `get_normalization_layer` → TF `Rescaling` layer. **DELETE** (only used by TF datasets).
- `download_and_unpack` → used for CelebA. Can keep or remove.
- `random_augmentation` → TF augmentation. **DELETE** (not needed for ZEA).
- `timefunc` decorator → keep (no TF).
- `save_animation`, `save_to_gif`, `save_to_video` → use matplotlib/numpy. Check for TF refs and remove.
- `get_latest_checkpoint` → keep (filesystem ops only).

#### 3. `generators/models.py`
**Current state**: Imports `tensorflow_addons`, `keras`, and all model classes.
**Action**: Remove TF imports. The `get_model()` function should handle `model_name == "score"` by returning `ScoreNet(config)`. For the "score" path, it currently calls `model.compile(optimizer=..., loss_fn="score_loss")` which is a Keras pattern. Replace with just returning the model — optimizer setup moves to the training script.
**Key change**: `get_model()` should NOT call `.compile()`. Just instantiate and return the model. The training loop handles optimizer.
**Remove**: GAN, UNet, NCSNv2-standalone (Keras versions), encoder/decoder models. Only keep `score` path using the already-ported `ScoreNet`.

#### 4. `utils/gpu_config.py`
**Current state**: Likely calls `tf.config.experimental.set_memory_growth` etc.
**Action**: Replace with PyTorch GPU config: `torch.cuda.set_device()`, or just make it a no-op. PyTorch handles GPU automatically.

#### 5. `utils/callbacks.py`
**Current state**: Keras `Callback` subclasses `EvalDataset` and `Monitor`.
**Action**: Rewrite as plain Python classes that get called from the training loop (not Keras callbacks). Key functionality:
- `EvalDataset`: computes evaluation loss periodically
- `Monitor`: generates sample images periodically, saves plots
Both need to work with PyTorch tensors and the ported `ScoreNet`.

#### 6. `utils/checkpoints.py`
**Current state**: Hybrid TF+PyTorch checkpointing.
**Action**: Keep PyTorch checkpointing (`torch.save`, `torch.load`). Remove TF checkpoint code. Key interface:
- `ModelCheckpoint.save(epoch)` → saves model state dict + optimizer state
- `ModelCheckpoint.restore(file=None)` → loads latest or specified checkpoint

#### 7. `train.py`
**Current state**: Uses `wandb.init`, `keras.callbacks.ReduceLROnPlateau`, `wandb.keras.WandbCallback`, `model.fit()`.
**Action**: Rewrite as a standard PyTorch training loop:
```python
for epoch in range(epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = model.score_loss(batch)
        loss.backward()
        optimizer.step()
        ema_update(...)  # if using EMA
    # eval, checkpoint, logging
```
Keep wandb logging. Keep YAML config loading. Keep argparse interface (`-c config.yaml`).

### Priority 2: Required for `inference.py`

#### 8. `utils/inverse.py`
**Current state**: Hybrid TF+PyTorch. Contains `SGMDenoiser`, `GANDenoiser`, `GlowDenoiser`, `BM3DDenoiser`, etc.
**Action**: Port `SGMDenoiser` to PyTorch. It's the one needed for PIGDM inference. It wraps `ScoreNet` + `ScoreSampler` + `Corruptor`. The other denoisers (GAN, Glow, BM3D, NLM) can be kept as-is or stubbed.
**Key method**: `SGMDenoiser.__call__(noisy, clean=None)` → runs `ScoreSampler` with guidance.

#### 9. `utils/runs.py`
**Current state**: No TF imports (uses yaml, wandb, easydict, utils.utils).
**Action**: Should work once `utils/utils.py` is ported. Verify.

#### 10. `utils/signals.py`
**Current state**: TF-only (`tensorflow`, `tensorflow_addons`).
**Action**: Only needed by TF datasets and `GaussianCorruptor` (old TF version). The ported `GaussianCorruptor` doesn't use it. Can stub or delete. If keeping: `add_gaussian_noise(x, sigma)` is just `x + torch.randn_like(x) * sigma`.

#### 11. `inference.py`
**Current state**: Uses `easydict`, chains through `get_model`, `get_denoiser`, `init_config`, etc.
**Action**: Once dependencies are ported, this should mostly work. May need minor adjustments.

### Not Needed (Skip)

- `generators/GAN.py` — TF Keras GAN, not needed for score-based dehazing
- `generators/glow/` — Already PyTorch, not needed for this task
- `utils/nlm.py` — Non-local means denoiser, classical method
- `utils/metrics.py` — Check if it uses TF; if pure numpy, leave as-is
- `utils/opt.py` — Check dependencies
- `utils/git_info.py` — Pure Python, no changes needed

---

## Reference Implementations

Use these as ground truth when unsure:

| Location | Contents |
|----------|----------|
| `reproduce_helpers/ncsnv2/` | Official PyTorch NCSNv2 (Yang Song) |
| `reproduce_helpers/ncsnv2/models/ncsnv2.py` | `NCSNv2` nn.Module |
| `reproduce_helpers/ncsnv2/models/layers.py` | `ResidualBlock`, `RefineBlock`, etc. |
| `reproduce_helpers/ncsnv2/models/normalization.py` | `InstanceNorm2dPlus`, etc. |
| `reproduce_helpers/ncsnv2/losses/dsm.py` | `anneal_dsm_score_estimation()` |
| `reproduce_helpers/ncsnv2/runners/ncsn_runner.py` | Training loop reference |
| `reproduce_helpers/score_sde_pytorch/` | Official PyTorch score_sde (Yang Song) |
| `reproduce_helpers/score_sde_pytorch/sde_lib.py` | `VPSDE`, `VESDE`, `subVPSDE` |
| `reproduce_helpers/score_sde_pytorch/sampling.py` | PC sampler, predictors, correctors |
| `reproduce_helpers/score_sde_pytorch/losses.py` | `get_sde_loss_fn()`, `get_smld_loss_fn()` |
| `reproduce_helpers/score_sde_pytorch/models/utils.py` | `get_score_fn()` |

**For training loop structure**, reference `reproduce_helpers/ncsnv2/runners/ncsn_runner.py` — it has a clean PyTorch training loop with EMA, checkpointing, and evaluation.

---

## Config Files (Already Created)

These YAML configs exist and should be loaded by the training/inference scripts:

| File | Purpose |
|------|---------|
| `configs/training/score_zea_tissue.yaml` | Train tissue model (100 epochs, bs=8, lr=1e-4, NCSNv2, VESDE) |
| `configs/training/score_zea_haze.yaml` | Train haze model (same params, different dataset) |
| `configs/inference/paper/zea_dehaze_pigdm.yaml` | PIGDM joint inference (lambda=0.5, kappa=0.5, T=200) |

---

## Existing Config Keys Used by Ported Code

`ScoreNet(config)` expects these config attributes:

```python
config.image_shape    # (C, H, W) e.g. (3, 1024, 64)
config.channels       # int, base feature channels, e.g. 32
config.activation     # str, e.g. 'elu'
config.normalization  # str, e.g. 'instance'
config.kernel_size    # int, e.g. 3
config.sde            # str: 'vesde', 'vpsde', 'subvpsde', 'simple'
config.sigma_min      # float (for VESDE)
config.sigma_max      # float (for VESDE)
config.beta_min       # float (for VPSDE)
config.beta_max       # float (for VPSDE)
config.num_scales     # int, number of discretization steps
config.score_backbone # str, default 'NCSNv2'
config.reduce_mean    # bool, default True
config.likelihood_weighting  # bool, default False
```

`ScoreSampler(...)` expects these init args (from inference config):

```python
sampling_method   # 'pc'
predictor         # 'euler_maruyama' or 'reverse_diffusion'
corrector         # 'langevin', 'ald', 'none'
corrector_snr     # float, e.g. 0.16
guidance          # 'pigdm', 'dps', 'projection', or None
lambda_coeff      # float, e.g. 0.5
kappa_coeff       # float, e.g. 0.5
noise_model       # ScoreNet instance (for joint inference) or None
```

---

## Success Criteria

### Must Pass

1. `python -c "from datasets import get_dataset"` — no TF import error
2. `python -c "from generators.models import get_model"` — no TF import error
3. `python -c "from utils.utils import load_config_from_yaml, set_random_seed"` — works
4. Training runs for at least 2 steps on ZEA data without error:
   ```bash
   cd /scrfs/storage/tp030/home/f2f_ldm/dehazing-diffusion/joint_diffusion
   source /scrfs/storage/tp030/home/f2f_ldm/.venv_joint/bin/activate
   python train.py -c configs/training/score_zea_tissue.yaml --data_root ../../data
   ```
   (or a new `train_pytorch.py` if `train.py` is rewritten)
5. Loss is finite (not NaN/Inf) and decreases over steps
6. Checkpoints save and load correctly
7. `test_sanity.py` still passes (don't break existing ported code)

### Bonus

8. `inference.py` runs PIGDM dehazing with two trained models
9. wandb logging works (offline mode)
10. EMA (exponential moving average) of model weights during training

---

## Important Constraints

- **DO NOT install TensorFlow**. The entire point is to run without it.
- **DO NOT modify the 6 already-ported SGM files** unless fixing a bug discovered during integration.
- **Keep the same import paths**: `from generators.SGM.SGM import ScoreNet`, `from generators.layers import ResidualBlock`, etc.
- **Packages available**: `torch`, `numpy`, `tqdm`, `pyyaml`, `easydict`, `wandb`, `matplotlib`, `scipy`. Check with `pip list`.
- **No `tensorflow_addons`** — this was used for `InstanceNormalization`, `GroupNormalization`, `SpectralNormalization`, `MovingAverage` optimizer. PyTorch equivalents: `nn.InstanceNorm2d`, `nn.GroupNorm`, `torch.nn.utils.spectral_norm`, custom EMA.

---

## EMA Reference

The original `train.py` used `tfa.optimizers.MovingAverage`. In PyTorch, EMA is typically done manually:

```python
# After each optimizer step:
for ema_param, param in zip(ema_model.parameters(), model.parameters()):
    ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)
```

Or use `torch.optim.swa_utils.AveragedModel` (PyTorch 1.8+).

Reference: `reproduce_helpers/ncsnv2/models/ema.py` has an `EMAHelper` class.

---

## File-by-File Porting Checklist

- [ ] `utils/utils.py` — Remove TF, keep PyTorch + numpy utilities
- [ ] `utils/gpu_config.py` — Replace TF GPU config with PyTorch or no-op
- [ ] `utils/signals.py` — Stub or delete (not needed for ZEA)
- [ ] `datasets.py` — Remove TF imports, keep ZeaDataset path working
- [ ] `generators/models.py` — Remove TF, keep `get_model()` for "score" path only
- [ ] `utils/checkpoints.py` — Remove TF checkpointing, keep PyTorch
- [ ] `utils/callbacks.py` — Rewrite as plain Python classes (not Keras Callbacks)
- [ ] `train.py` — Rewrite with PyTorch training loop
- [ ] `utils/inverse.py` — Port `SGMDenoiser` for inference
- [ ] `inference.py` — Adjust after dependencies are ported
- [ ] Verify `test_sanity.py` still passes
- [ ] Run training smoke test (2 steps on ZEA data)
