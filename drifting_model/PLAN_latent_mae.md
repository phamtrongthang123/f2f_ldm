# Replace MoCo v2 with Latent-MAE Feature Encoder

## Context

The current implementation uses MoCo v2 (pixel-space ResNet-50) as the feature encoder, which requires decoding latents to 256×256 pixels via the VAE decoder during training. This is wrong — the paper's ablation default config actually uses a **latent-MAE** encoder that operates directly on 32×32×4 latents. Switching to latent-MAE:

- Matches what the paper actually uses for all latent-space configs
- Eliminates VAE decoder from the training loop (~49M params + expensive 32→256 upsample)
- Makes training feasible on 1× A100 40GB (user's hardware)

**Target**: ResNet basic blocks, base width 256, GroupNorm, 192 epochs MAE pre-training.

---

## Files to Create

### 1. `models/latent_mae.py` — MAE architecture

**BasicBlock**: Two 3×3 convs with GroupNorm(32) and ReLU. Optional stride-2 downsample + 1×1 shortcut projection.

**ResNetEncoder** (per `appendix_impl.tex:90-103`):
- `conv1`: Conv2d(4→C, 3×3, stride=1, pad=1) + GN + ReLU — no downsampling, no maxpool
- Stage 1: 3 basic blocks at C=256 channels, 32×32
- Stage 2: 4 basic blocks at 2C=512, 16×16 (first block stride 2)
- Stage 3: 6 basic blocks at 4C=1024, 8×8
- Stage 4: 3 basic blocks at 8C=2048, 4×4
- Returns: `(x0, f1, f2, f3, f4)` for decoder skip connections
- ~340M params at C=256

**UNetDecoder** (per `appendix_impl.tex:105-113`):
- 3×3 conv on f4 (4×4)
- 3 UpsampleBlocks: bilinear 2× → cat skip → GN → two 3×3 convs + GN + ReLU
  - Block 1: (8C + 4C) → 4C, 4→8
  - Block 2: (4C + 2C) → 2C, 8→16
  - Block 3: (2C + C) → C, 16→32
- FinalBlock: cat with x0 (no upsample, already 32×32) → (C + C) → C
- 1×1 conv → 4 output channels

**LatentMAE**:
- `_create_mask()`: 16×16 grid of 2×2 patches, 50% independent Bernoulli, upsample to 32×32 pixel mask
- `forward(latents)`: zero masked patches → encoder → decoder → L2 loss on masked regions only
- `forward_encoder(latents)`: encoder only (for feature extraction, no masking)

### 2. `train_mae.py` — MAE pre-training script

Single-GPU, gradient accumulation. Per `appendix_impl.tex:121-124`:
- AdamW, lr 4e-3, betas (0.9, 0.95), weight_decay 0.05
- Effective batch 8192 via grad accum (micro-batch 64 × 128 accum steps)
- EMA decay 0.9995
- 192 epochs (~30K effective steps)
- bf16 autocast
- Cosine LR schedule with 10-epoch warmup
- Data: pre-computed latents (recommended) or on-the-fly with RandomResizedCrop + VAE encode
- Saves `{encoder, ema_encoder, decoder, optimizer, epoch}` checkpoint

---

## Files to Modify

### 3. `models/feature_encoder.py` — Replace MoCo with LatentMAE extractor

Replace `MoCoV2FeatureExtractor` with `LatentMAEFeatureExtractor`:
- Input: `[B, 4, 32, 32]` latents (not pixels) — no ImageNet normalization needed
- Import `ResNetEncoder` from `latent_mae.py`
- Load EMA encoder weights from MAE checkpoint
- Freeze all params (`requires_grad_(False)`)
- **Reuse** existing `_get_stage_features()` and `_extract_multiscale()` verbatim — they are architecture-agnostic
- `extract_features(latents)`: run encoder stages manually (conv1 → stage1-4), collect intermediate features, apply `_extract_multiscale` per feature map
- Output interface unchanged: list of `[B, num_vecs, C]` tensors

Extraction points (same logic as MoCo, same block counts [3,4,6,3]):
- Stage 1: after block 2, after block 3 (final) → 2 maps at 32×32×256
- Stage 2: after block 2, after block 4 (final) → 2 maps at 16×16×512
- Stage 3: after block 2, after block 4, after block 6 (final) → 3 maps at 8×8×1024
- Stage 4: after block 2, after block 3 (final) → 2 maps at 4×4×2048
- Plus input layer x² mean → 1 tensor

### 4. `train_imagenet.py` — Remove VAE decode, add single-GPU support

**Remove VAE decode from training loop** (lines 305-316):
```python
# OLD: gen_latent → vae.decode → pixel → normalize → feature_encoder
# NEW: gen_latent → feature_encoder (directly)
gen_feats = feature_encoder(gen_latent)
with torch.no_grad():
    pos_feats = feature_encoder(pos_latent)
    unc_feats = feature_encoder(unc_latent)
```

**Add single-GPU support** (conditional DDP):
- Check `WORLD_SIZE` env var; skip `dist.init_process_group` if single GPU
- Use plain model instead of DDP wrapper when single GPU
- Use regular DataLoader (no DistributedSampler) when single GPU
- Skip `no_sync()` context when single GPU
- Access model directly (not `.module`) when single GPU

**Load feature encoder** — change from MoCo to LatentMAE:
```python
feature_encoder = LatentMAEFeatureExtractor(
    checkpoint_path=feat_cfg["checkpoint_path"],
    base_width=feat_cfg.get("base_width", 256),
).to(device)
```

**VAE only needed conditionally**: load only if `dataset.mode == "onthefly"` (for encoding images to latents when pushing to queue). Not needed for feature extraction.

**Add bf16 autocast** around forward + loss computation (optional flag).

### 5. `configs/ablation_default.yaml` — Update feature encoder section

```yaml
feature_encoder:
  type: latent_mae
  checkpoint_path: ./checkpoints/mae/mae_final.pt
  base_width: 256
```

### 6. `models/__init__.py` — Update exports

```python
from .dit import DiTGenerator
from .feature_encoder import LatentMAEFeatureExtractor
from .latent_mae import LatentMAE, ResNetEncoder
```

### 7. `scripts/download_weights.sh` — Remove MoCo download

Remove MoCo v2 wget. Note that MAE encoder must be pre-trained via `train_mae.py`.

---

### 8. `drifting_loss.py` — Vectorize per-location loop (critical for performance)

The current `_compute_single_feature_loss` has a Python `for loc in range(num_vecs)` loop. For stage 1 at 32×32, this is 1024 serial `compute_V` calls × 3 temperatures = 3072 serial GPU kernel launches per feature map, per class. This is a severe bottleneck.

**Fix**: Rewrite `compute_V` to accept a batched location dimension. Instead of looping over locations in Python, reshape data to `[num_locs, N, C]` and use `torch.cdist` on the batched dim (cdist supports batched inputs). Then `_compute_single_feature_loss` does a single batched call per temperature instead of `num_vecs` separate calls.

```python
# Batched compute_V: x [L, N, D], y_pos [L, N_pos, D], y_neg [L, N_neg, D]
# cdist supports batch dim: torch.cdist([L,N,D], [L,M,D]) → [L,N,M]
```

This turns 3072 serial kernel launches into 3 batched calls (one per temperature) per feature map. Major speedup.

---

## Files NOT modified

- `data/sample_queue.py` — stores latents, no change
- `data/imagenet.py` — loads latents, no change
- `eval_imagenet.py` — still needs VAE decoder for latent→pixel at eval time, no change
- `drifting_model_demo.py` — existing toy demo, untouched

---

## Memory Budget (1× A100 40GB, bf16)

| Component | Params | Memory (bf16) |
|-----------|--------|---------------|
| DiT-B/2 | 130M | 260MB |
| DiT-B/2 EMA | 130M | 260MB |
| AdamW states | — | 1.04GB (fp32) |
| MAE encoder (frozen) | 340M | 680MB |
| **Total fixed** | | **~2.2GB** |

Per-class activations (64 generated + 64 pos + 16 unc):
- Generator forward (64 samples): ~100MB bf16
- MAE encoder forward with grad (64 gen): ~60MB bf16
- MAE encoder forward no grad (80 pos+unc): freed after forward
- Drifting loss cdist: small (N=64 matrices)
- **Peak per class**: ~200-300MB

**Headroom**: ~37GB free. Gradient accumulation across 64 classes is comfortable.

---

## Compute Estimates (1× A100 40GB)

### MAE Pre-training
- 192 epochs × ~157 effective steps/epoch = ~30K steps
- Per step: forward (340M ResNet encoder + decoder) + backward on micro-batch 64
- ~128 grad accum steps per effective step
- Estimate: **~5-7 days** on 1× A100

### Generator Training
- 30K steps, per step: 64 classes × (DiT forward + MAE encoder forward + drifting loss)
- Estimate: **~2-4 days** on 1× A100

---

## Implementation Order

1. `models/latent_mae.py` — BasicBlock, ResNetEncoder, UNetDecoder, LatentMAE
2. `train_mae.py` — MAE pre-training with grad accum
3. `models/feature_encoder.py` — LatentMAEFeatureExtractor (reuse `_extract_multiscale`)
4. `drifting_loss.py` — Vectorize compute_V and per-location loop
5. `train_imagenet.py` — remove VAE decode, add single-GPU support
6. `configs/ablation_default.yaml`, `models/__init__.py`, `scripts/download_weights.sh`

---

## Verification

1. **MAE smoke test**: `LatentMAE(base_width=64)` with random `[2, 4, 32, 32]` → loss computes, gradients flow
2. **Feature extractor shapes**: `LatentMAEFeatureExtractor` with random latents → list of `[B, num_vecs, C]` tensors with expected shapes
3. **Training integration**: 10 generator training steps with random MAE encoder → loss computes without error
4. **MAE pre-training**: run 100 steps, verify reconstruction loss decreases
5. **Full pipeline**: pre-train MAE 192 epochs → train generator 30K steps → eval FID ~8.46
