# Vertical Bar Artifact Removal Designs

This document describes two high-level options to target the vertical bar artifact seen in m3 images.

## Option 1: Stripe Loss During SDXL LoRA Training

### Goal
Explicitly penalize vertical stripe energy while training the SDXL LoRA so the denoiser learns to suppress bars.

### Core Idea
Add a lightweight "stripe loss" on reconstructed samples for m3 inputs only. The loss measures vertical periodic energy by comparing column-wise statistics to a smoothed version.

### Flow
- **Input**: Training batch of m1 + m3 images.
- **Forward**: Standard diffusion loss (MSE on predicted noise).
- **Stripe Loss**:
  - Compute a column-mean signal for each image (mean over height and channels).
  - Smooth the signal with a 1D blur kernel.
  - Stripe energy = L1(column_mean - smoothed_column_mean).
  - Apply only to m3 samples (where bars exist).
- **Total Loss**: `loss = diffusion_loss + stripe_weight * stripe_loss`.

### Parameters
- `stripe_weight`: 0.1 to 0.5 (start low, increase if bars persist).
- `blur_kernel`: 9 to 21 pixels (controls bar frequency emphasis).

### Expected Effect
The model learns that vertical periodic structure is "bad" for m3 and should be removed during denoising.

### Risks
- Over-penalizing can wash out true vertical structures.
- Needs careful weight tuning and monitoring for oversmoothing.

---

## Option 2: Synthetic Bar Removal Pre-Cleaner (Paired Denoising)

### Goal
Train a dedicated bar-removal model using synthetic bars, then apply it before diffusion.

### Core Idea
Create paired data: `m1 + synthetic bars -> m1`, and train a small U-Net to remove bars. Use this model as a pre-cleaner for m3 inputs.

### Data Generation
- Start with m1 images (clean domain).
- Add synthetic vertical bars:
  - Random bar width (1-6 px).
  - Random gap (8-24 px).
  - Random intensity (bright or dark bars).
  - Optional slight blur to mimic sensor spread.

### Model
- Small U-Net (3-4 levels, base channels 32/64).
- Input: 3x512x512 image with synthetic bars.
- Output: Clean image.
- Loss: L1 (optionally add edge-preserving or TV loss).

### Integration
- Train and save `bar_removal.pt`.
- At inference:
  - Apply bar-removal model to m3 input image.
  - Feed the cleaned result into the SDXL + ET pipeline.

### Expected Effect
Bars are removed explicitly before diffusion, reducing the burden on SDXL/ET and improving translation quality.

### Risks
- Synthetic bars may not fully match real artifact distribution.
- Overfitting to synthetic patterns can leave residual real bars.

---

## Recommendation
Start with Option 2 to directly target the bar artifact. If bars remain, combine Option 2 with a small stripe loss (Option 1) for extra suppression.
