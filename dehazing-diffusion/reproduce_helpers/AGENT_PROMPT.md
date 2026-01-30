# Agent Prompt: Port TensorFlow Score-Based Diffusion to PyTorch & Reproduce Paper

## Task Overview

**Primary Goal**: Port the TensorFlow-based score-based diffusion model code in `joint_diffusion/` to PyTorch while maintaining the same import structure. The user only works with PyTorch and needs the code to be fully functional without TensorFlow dependencies.

**Secondary Goal**: Reproduce the results from the paper "Dehazing Ultrasound using Diffusion Models" (IEEE TMI 2024) using synthetic data generated with ZEA. /home/tp030/f2f_ldm/dehazing-diffusion/paper

DO NOT RUN any training or inference yet. Focus solely on porting the code and ensuring it can be imported and basic operations run without errors and sanity tests ready to verify correctness.

## Paper Summary

**Title**: Dehazing Ultrasound using Diffusion Models  
**Authors**: Tristan Stevens et al. (TU Eindhoven + Philips Research)  
**Published**: IEEE Transactions on Medical Imaging, October 2024

### Key Concepts

1. **Problem**: Cardiac ultrasound images suffer from "haze" - structured noise from multipath reflections through skin/fat/muscle layers

2. **Approach**: Joint posterior sampling with two diffusion models:
   - Model 1: `s_θ(x,t)` - learns clean tissue distribution `p(x)`
   - Model 2: `s_φ(h,t)` - learns haze distribution `p(h)`

3. **Measurement Model**: `y = x + h` (additive in RF domain)

4. **Inference**: PIGDM (Pseudo-Inverse Guidance Diffusion Model) for joint posterior sampling from `p(x,h|y)`

5. **Key Innovation**: Works in RF (radio-frequency) domain, not B-mode image domain

### Paper Figures to Reproduce

| Figure | Description | Data Required |
|--------|-------------|---------------|
| Fig. 7 | In-vitro phantom results with ground truth comparison | Synthetic tissue + haze |
| Fig. 9 | PSNR vs haze level comparison | Synthetic data with varying haze |
| Fig. 10-11 | gCNR scores grouped by subject/view | In-vivo or synthetic |
| Fig. 14 | Qualitative dehazing comparison | Any test data |

### Metrics to Implement

1. **PSNR** - Peak Signal-to-Noise Ratio (for in-vitro with ground truth)
2. **gCNR** - Generalized Contrast-to-Noise Ratio (unsupervised, for in-vivo)
3. **KS Test** - Kolmogorov-Smirnov for speckle statistics preservation
4. **FWHM** - Lateral speckle size (resolution metric)

## Current Codebase State

### Repository Structure
```
/scrfs/storage/tp030/home/f2f_ldm/dehazing-diffusion/
├── joint_diffusion/           # Main codebase (originally TensorFlow)
│   ├── generators/
│   │   ├── layers.py          # NEEDS PORTING (TF layers)
│   │   ├── SGM/
│   │   │   ├── SGM.py         # NEEDS PORTING (NCSNv2, ScoreNet)
│   │   │   ├── sde_lib.py     # NEEDS PORTING (VPSDE, VESDE, etc.)
│   │   │   ├── sampling.py    # NEEDS PORTING (PC sampler)
│   │   │   └── guidance.py    # NEEDS PORTING (PIGDM, DPS, Projection)
│   ├── utils/
│   │   ├── corruptors.py      # TF-dependent
│   │   ├── inverse.py         # TF-dependent
│   │   ├── signals.py         # TF-dependent
│   │   └── utils.py           # TF-dependent
│   └── datasets.py            # Mixed TF/PyTorch (ZeaDataset is PyTorch)
│
├── reproduce_helpers/
│   ├── ncsnv2/                # OFFICIAL PyTorch NCSNv2 from Yang Song
│   │   ├── models/
│   │   │   ├── ncsnv2.py      # PyTorch NCSNv2 model
│   │   │   ├── layers.py      # PyTorch layers (ResidualBlock, RefineBlock, etc.)
│   │   │   ├── normalization.py
│   │   │   └── __init__.py    # get_sigmas(), anneal_Langevin_dynamics()
│   │   ├── losses/dsm.py      # anneal_dsm_score_estimation()
│   │   ├── runners/ncsn_runner.py  # Training/sampling loop
│   │   └── configs/           # Example configs (celeba.yml, etc.)
│   │
│   ├── score_sde_pytorch/     # OFFICIAL PyTorch score_sde from Yang Song
│   │   ├── sde_lib.py         # VPSDE, VESDE, subVPSDE (PyTorch)
│   │   ├── sampling.py        # PC sampler, predictors, correctors
│   │   ├── losses.py          # get_sde_loss_fn, get_smld_loss_fn
│   │   ├── models/
│   │   │   ├── ncsnv2.py      # PyTorch NCSNv2
│   │   │   ├── ncsnpp.py      # PyTorch NCSN++
│   │   │   ├── layers.py      # PyTorch layers
│   │   │   └── utils.py       # get_score_fn(), get_model_fn()
│   │   └── configs/           # Example configs
│   │
│   └── sanity_test_pytorch.sh # Sanity test script
```

## Reference Implementations (USE THESE)

### 1. Official NCSNv2 (PyTorch)
Location: `reproduce_helpers/ncsnv2/`
- **Author**: Yang Song (paper author)
- **Paper**: "Improved Techniques for Training Score-Based Generative Models"
- **Key files**:
  - `models/ncsnv2.py`: `NCSNv2`, `NCSNv2Deeper`, `NCSNv2Deepest` nn.Modules
  - `models/layers.py`: `ResidualBlock`, `RefineBlock`, `RCUBlock`, `MSFBlock`, `CRPBlock`
  - `models/__init__.py`: `get_sigmas()`, `anneal_Langevin_dynamics()`
  - `losses/dsm.py`: `anneal_dsm_score_estimation()` - the loss function

### 2. Official Score SDE (PyTorch)
Location: `reproduce_helpers/score_sde_pytorch/`
- **Author**: Yang Song (paper author)
- **Paper**: "Score-Based Generative Modeling through Stochastic Differential Equations"
- **Key files**:
  - `sde_lib.py`: `VPSDE`, `VESDE`, `subVPSDE` classes with `marginal_prob()`, `prior_sampling()`, `reverse()`
  - `sampling.py`: `EulerMaruyamaPredictor`, `ReverseDiffusionPredictor`, `LangevinCorrector`, `get_pc_sampler()`
  - `losses.py`: `get_sde_loss_fn()`, `get_smld_loss_fn()`, `get_ddpm_loss_fn()`
  - `models/utils.py`: `get_score_fn()` - wraps model to return proper score

## Porting Requirements

### 1. Keep Same Import Structure
The user wants to keep imports like:
```python
from generators.SGM.SGM import NCSNv2, ScoreNet
from generators.SGM.sde_lib import VPSDE, VESDE, get_sde
from generators.SGM.sampling import ScoreSampler, get_predictor, get_corrector
from generators.layers import ConvBlock, ResidualBlock, RefineBlock
```

### 2. PyTorch-Only
- NO TensorFlow imports
- NO Keras imports
- Use `torch.nn.Module` for all models
- Use `torch.Tensor` throughout

### 3. Match TF Interface Where Possible
The TF code uses configs with attributes like:
```python
config.image_shape = (H, W, C)  # TF format
config.channels = 128
config.activation = 'elu'
config.normalization = 'batch'
config.sde = 'vesde'
config.num_scales = 1000
```

The PyTorch code should accept the same config format.

## Files to Port (Priority Order)

### Priority 1: Core Model Components
1. **`generators/SGM/sde_lib.py`**
   - Port from `score_sde_pytorch/sde_lib.py`
   - Classes: `SDE`, `VPSDE`, `VESDE`, `subVPSDE`, `simple`
   - Keep `get_sde(config)` factory function

2. **`generators/layers.py`**
   - Port from `ncsnv2/models/layers.py` and `score_sde_pytorch/models/layers.py`
   - Classes: `ConvBlock`, `ResidualBlock`, `RefineBlock`, `RCUBlock`, `MSFBlock`, `CRPBlock`
   - Functions: `get_activation()`, `get_normalization()`

3. **`generators/SGM/SGM.py`**
   - Port `NCSNv2` from `ncsnv2/models/ncsnv2.py`
   - Port `ScoreNet` training wrapper (handles SDE, loss, sampling)
   - Ensure `model(x, t)` returns score

### Priority 2: Sampling & Training
4. **`generators/SGM/sampling.py`**
   - Port from `score_sde_pytorch/sampling.py`
   - Classes: `Predictor`, `Corrector`, `EulerMaruyamaPredictor`, `ReverseDiffusionPredictor`, `LangevinCorrector`
   - Functions: `get_pc_sampler()`, `anneal_Langevin_dynamics()`

5. **`generators/SGM/guidance.py`**
   - Port guidance classes: `PIGDM`, `DPS`, `Projection`
   - Used for conditional sampling (denoising, compressed sensing)

### Priority 3: Utils & Dataset
6. **`utils/corruptors.py`** - Port noise corruptors
7. **`utils/inverse.py`** - Port denoising utilities
8. **`datasets.py`** - `ZeaDataset` is already PyTorch, keep TF loaders for compatibility

## Key Differences TF vs PyTorch

| Aspect | TensorFlow | PyTorch |
|--------|------------|---------|
| Image format | (B, H, W, C) | (B, C, H, W) |
| Model class | `keras.Model` | `nn.Module` |
| Forward | `model(x, training=True)` | `model(x)` |
| Gradients | `tf.GradientTape()` | `torch.autograd.grad()` |
| Random | `tf.random.normal()` | `torch.randn()` |
| Device | Auto | `.to(device)` required |

## Sanity Test Criteria

Create a test script that verifies:

1. **Imports work** without TensorFlow
   ```python
   from generators.SGM.SGM import NCSNv2, ScoreNet
   from generators.SGM.sde_lib import VPSDE, VESDE
   from generators.layers import ResidualBlock, RefineBlock
   ```

2. **SDE classes work**
   ```python
   sde = VESDE(sigma_min=0.01, sigma_max=50, N=100)
   x = torch.randn(2, 1, 64, 64)
   t = torch.rand(2)
   mean, std = sde.marginal_prob(x, t)
   assert mean.shape == x.shape
   ```

3. **Model forward pass works**
   ```python
   model = NCSNv2(config)
   x = torch.randn(2, 1, 64, 64)
   score = model(x)
   assert score.shape == x.shape
   ```

4. **Loss computation works**
   ```python
   score_net = ScoreNet(config)
   loss = score_net.score_loss(batch)
   loss.backward()
   ```

5. **Sampling works** (at least a few steps)
   ```python
   samples = score_net.sample(batch_size=2)
   assert samples.shape == (2, C, H, W)
   ```

6. **ZeaDataset works**
   ```python
   dataset = ZeaDataset(root=data_path, split='train')
   batch = next(iter(DataLoader(dataset, batch_size=4)))
   assert isinstance(batch, torch.Tensor)
   ```

## Config Example

```yaml
# Example config for ZEA dehazing
dataset_name: zea_haze
data_root: /path/to/data
image_size: [128, 64]  # H, W
channels: 32           # Base feature channels
in_channels: 1         # Input image channels

# Model
score_backbone: NCSNv2
activation: elu
normalization: instance
kernel_size: 3

# SDE
sde: vesde
sigma_min: 0.01
sigma_max: 50
num_scales: 1000

# Training
batch_size: 16
lr: 0.0001
n_iters: 100000

# Sampling
predictor: reverse_diffusion
corrector: langevin
snr: 0.16
```

## Important Notes

1. **Don't guess implementations** - Always reference the official PyTorch code in `ncsnv2/` and `score_sde_pytorch/`

2. **Test incrementally** - After porting each file, run the sanity test to verify

3. **Preserve TF config format** - The user has existing YAML configs that use TF conventions

4. **ZeaDataset is special** - It's for ultrasound RF data with shape (N, n_tx, n_ax, n_el), already PyTorch

5. **The guidance module** is for conditional/inverse problems (denoising, dehazing) - port carefully

## Success Criteria

1. ✅ All imports work without TensorFlow
2. ✅ `python -c "from generators.SGM.SGM import NCSNv2"` succeeds
3. ✅ Sanity test script passes all layers
4. ✅ Can instantiate model, compute loss, run backward
5. ✅ ZeaDataset loads and batches correctly
6. ✅ (Bonus) Full training loop runs for a few iterations

## Virtual Environment

```bash
source /scrfs/storage/tp030/home/f2f_ldm/.venv_joint/bin/activate
cd /scrfs/storage/tp030/home/f2f_ldm/dehazing-diffusion/joint_diffusion
```

Packages needed: `torch`, `numpy`, `tqdm`, `pyyaml`

---

## Paper Reproduction Plan

The full reproduction plan is in `/home/tp030/f2f_ldm/dehazing-diffusion/paper/plan.md`. Here's the summary:

### Phase 1: Environment Setup ✅
- ZEA environment: `.venv_zea` with JAX backend
- Joint diffusion environment: `.venv_joint` with PyTorch

### Phase 2: Code Porting (THIS TASK)
Port TF → PyTorch:
- `generators/layers.py`
- `generators/SGM/SGM.py` (NCSNv2, ScoreNet)
- `generators/SGM/sde_lib.py` (VPSDE, VESDE)
- `generators/SGM/sampling.py` (PC sampler)
- `generators/SGM/guidance.py` (PIGDM, DPS, Projection)
- `utils/corruptors.py` (HazeCorruptor)

### Phase 3: Sanity Tests
Run incremental tests after each file is ported:
1. Import smoke test
2. Component shape tests
3. Dataset loading
4. Training smoke test (2 steps)
5. Corruptor test

### Phase 4: Data Synthesis with ZEA
```bash
cd /home/tp030/f2f_ldm
source .venv_zea/bin/activate
KERAS_BACKEND=jax python dehazing-diffusion/reproduce_helpers/zea_synthesize_dataset.py \
  --output-root data/zea_synth \
  --n-train 1000 --n-val 100
```

Output structure:
```
data/zea_synth/
├── tissue/
│   ├── train.npz  # shape: (N, n_tx, n_ax, n_el)
│   └── val.npz
└── haze/
    ├── train.npz
    └── val.npz
```

### Phase 5: Train Diffusion Models

**Tissue model** (clean ultrasound prior):
```bash
python train.py -c configs/training/score_zea_tissue.yaml --data_root ../../data
```

**Haze model** (haze prior):
```bash
python train.py -c configs/training/score_zea_haze.yaml --data_root ../../data
```

Training parameters (from paper Section 3.2.4):
- Batch size: 8
- Learning rate: 1e-4
- Epochs: 100
- Network: NCSNv2 (channels=32, kernel=3)
- Image size: [1024, 64] or configured

### Phase 6: Joint Inference (Dehazing)

**Inference config**: `configs/inference/paper/zea_dehaze_pigdm.yaml`

Key parameters:
- `guidance: pigdm` (Pseudo-Inverse Guidance)
- `lambda_coeff: 0.5` (tissue data consistency weight)
- `kappa_coeff: 0.5` (haze data consistency weight)
- `num_scales: 200` (T=200 diffusion steps)

```bash
python inference.py -e paper/zea_dehaze_pigdm -t denoise -m sgm --data_root ../../data
```

### Phase 7: Evaluation

Compute metrics from `processing.py`:

```python
from processing import gcnr, psnr

# gCNR: Generalized Contrast-to-Noise Ratio
# Requires two ROIs: chamber (hazy region) and wall (tissue region)
score = gcnr(chamber_roi, wall_roi)

# PSNR: Peak Signal-to-Noise Ratio (when ground truth available)
score = psnr(dehazed, ground_truth)
```

---

## Key Files for Paper Reproduction

### Configs
| File | Purpose |
|------|---------|
| `configs/training/score_zea_tissue.yaml` | Train tissue model |
| `configs/training/score_zea_haze.yaml` | Train haze model |
| `configs/inference/paper/zea_dehaze_pigdm.yaml` | Joint dehazing inference |

### Scripts
| File | Purpose |
|------|---------|
| `reproduce_helpers/zea_synthesize_dataset.py` | Generate synthetic data |
| `reproduce_helpers/visualize_dataset.py` | Visualize tissue vs haze |
| `reproduce_helpers/visualize_cyst.py` | Visualize cyst phantom |
| `joint_diffusion/train.py` | Train diffusion models |
| `joint_diffusion/inference.py` | Run dehazing inference |

### SLURM Scripts
| File | Purpose |
|------|---------|
| `reproduce_helpers/slurm_zea_synth.sh` | Data synthesis job |
| `reproduce_helpers/slurm_sanity_test.sh` | Sanity test job |

---

## Algorithm: Joint Posterior Sampling (PIGDM)

From the paper (Algorithm 1):

```
Input: measurement y, models s_θ (tissue), s_φ (haze)
Output: dehazed x_0, haze estimate h_0

1. Initialize x_T ~ N(0, σ_max²I), h_T ~ N(0, σ_max²I)
2. for t = T, T-1, ..., 1 do:
   # Predictor step (reverse diffusion)
   3. x_{t-1} = x_t + s_θ(x_t, t) * step_size + noise
   4. h_{t-1} = h_t + s_φ(h_t, t) * step_size + noise
   
   # Corrector step (Langevin dynamics)
   5. x_{t-1} = x_{t-1} + s_θ(x_{t-1}, t) * step_size + noise
   6. h_{t-1} = h_{t-1} + s_φ(h_{t-1}, t) * step_size + noise
   
   # Data consistency step (PIGDM guidance)
   7. r_t² = σ_t² / (σ_t² + 1)
   8. Σ_t = r_t² * I + r_t² * I  # Joint covariance
   9. μ_t = x_{0|t} + h_{0|t}     # Joint mean (Tweedie estimates)
   10. ∇_x log p(y|x_t,h_t) = Σ_t⁻¹(y - μ_t) * ∂x_{0|t}/∂x_t
   11. ∇_h log p(y|x_t,h_t) = Σ_t⁻¹(y - μ_t) * ∂h_{0|t}/∂h_t
   12. x_{t-1} = x_{t-1} + λ * r_t² * ∇_x log p(y|x_t,h_t)
   13. h_{t-1} = h_{t-1} + κ * r_t² * ∇_h log p(y|x_t,h_t)
   
3. return x_0, h_0
```

This is implemented in `generators/SGM/guidance.py` class `PIGDM`.

---

## Troubleshooting

### TensorFlow Import Errors
The goal is to eliminate ALL TensorFlow imports. If you see:
```
ModuleNotFoundError: No module named 'tensorflow'
```
Find the file with TF imports and port it to PyTorch.

### Shape Mismatches
- TensorFlow: `(B, H, W, C)` - channels last
- PyTorch: `(B, C, H, W)` - channels first
- ZEA data: `(N, n_tx, n_ax, n_el)` = `(N, C, H, W)` already PyTorch format

### GPU OOM
- Reduce batch size
- Use gradient checkpointing
- For ZEA: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`

### NaN in Training
- Check data normalization (should be in `image_range`, typically [0,1] or [-1,1])
- Reduce learning rate
- Check SDE sigma values

---

## Expected Outputs for Paper Figures

### Figure 7: In-Vitro Comparison
Generate with synthetic data:
- Ground truth tissue `x`
- Ground truth haze `h`
- Measurement `y = x + h`
- Dehazed output `x̂`
- Haze estimate `ĥ`
- Error maps `|x - x̂|`

### Figure 9: PSNR vs Haze Level
Test at different haze strengths:
```python
for gamma in [0.1, 0.2, 0.3, 0.4, 0.5]:
    y = x + gamma * h
    x_hat = dehaze(y)
    psnr_score = psnr(x_hat, x)
```

### Figure 10-11: gCNR Box Plots
Compute gCNR for each frame:
```python
for frame in dataset:
    score = gcnr(frame, chamber_mask, wall_mask)
```
Group by subject (Fig 10) or view (Fig 11).

### Figure 14: Qualitative Results
Show B-mode images:
- Original hazy `y`
- Dehazed `x̂`
- Haze estimate `ĥ`
- Compare: Diffusion vs NCSNv2 supervised vs BM3D

---

## Checklist: Full Paper Reproduction with sanity test

### Code Porting
- [ ] `generators/layers.py` - PyTorch layers
- [ ] `generators/SGM/sde_lib.py` - SDE classes
- [ ] `generators/SGM/SGM.py` - NCSNv2, ScoreNet
- [ ] `generators/SGM/sampling.py` - PC sampler
- [ ] `generators/SGM/guidance.py` - PIGDM, DPS, Projection
- [ ] `utils/corruptors.py` - HazeCorruptor
- [ ] `datasets.py` - ZeaDataset (already done)

### Sanity Tests
- [ ] All imports work without TensorFlow
- [ ] SDE marginal_prob shapes correct
- [ ] NCSNv2 forward pass works
- [ ] Loss computation and backward work
- [ ] ZeaDataset loads correctly

### Data & Training
- [ ] Generate synthetic tissue data with ZEA
- [ ] Generate synthetic haze data with ZEA
- [ ] Train tissue diffusion model
- [ ] Train haze diffusion model
- [ ] Validate models on held-out data

### Inference & Evaluation
- [ ] Run PIGDM joint inference
- [ ] Compute PSNR (synthetic data)
- [ ] Compute gCNR
- [ ] Generate comparison figures
- [ ] Compare with baselines (BM3D, supervised NCSNv2)
