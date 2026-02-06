# Implementation Status: Current Code vs Paper Requirements

This document compares the current reproduction code against the paper's requirements.

---

## Quick Summary

### Correctly implemented (matching Algorithm 1):
| Aspect | Code | Paper | Status |
|--------|------|-------|--------|
| Measurement model | Elevation row mask (every r-th row) | Binary matrix A selecting elevation planes | Correct |
| DPS guidance | Manual loop with `model.guidance_fn()` + `model.reverse_diffusion_step()` | Eq. 9, Algorithm 1 lines 27-32 | Correct |
| TV regularization | Per-step inside diffusion loop, azimuth axis | Algorithm 1 lines 35-36 | Correct |
| SeqDiff warm-start | Forward-diffuse previous recon to tau' | Algorithm 1 lines 16-19 | Correct |
| Normalization | [-1, 1] | [-1, 1] | Correct |
| Dynamic range | -50 dB | -50 dB | Correct |
| Hyperparameters (T, tau', gamma, zeta) | 200, 50, 35, 0.001 | 200, 50, 35, 0.001 | Correct |
| Volume shape | (112, 112, 112, 1) = (N_el, N_az, N_ax, C) | (N_el, N_az, N_ax) | Correct (C=1 channel dim added for Keras) |

### Limitations due to lacking 3D data:
| Aspect | Current Code | Paper Requirement | Impact |
|--------|--------------|-------------------|--------|
| Data source | CAMUS 2D images (2 val images cycled) | 3D volumetric cardiac ultrasound | No real elevation coherence |
| Pretrained model | EchoNet-Dynamic (A4C views) | Custom (trained on B-planes) | Domain gap |
| B-plane content | Stacked unrelated 2D images | Real elevation cross-sections | Artificial volume |

---

## What's Correct

### 1. Algorithm 1 — Fully Implemented

The reconstruction loop in `03_reconstruct_volume.py` faithfully implements Algorithm 1 (verified line-by-line, see `algo1_code_map.md`):

```
for step in range(start_step, N_STEPS):          # Algo 1 line 25
    for batch in batched_bplanes:                 # Algo 1 line 26
        guidance_fn() → reverse_diffusion_step()  # Algo 1 lines 27-32
    transpose to volume                           # Algo 1 line 34
    TV along azimuth (axis 1)                     # Algo 1 lines 35-36
    transpose back to bplanes
```

### 2. Measurement Model — Elevation Row Mask

The code uses the paper's actual measurement model: a binary elevation mask that selects every r-th row within each B-plane. Same mask for all B-planes.

```python
elevation_mask = np.zeros((N_el, N_ax, C), dtype=np.float32)
elevation_mask[observed_rows] = 1.0  # observed_rows = range(0, N_el, ACCEL_RATE)
measurements_all = bplanes_gt * elevation_mask[np.newaxis]
```

### 3. DPS Guidance — Manual Loop

The code implements the DPS loop manually (not via `model.posterior_sample()`), calling ZEA's low-level APIs directly:

```python
gradients, (error, (pred_noises, pred_images)) = model.guidance_fn(...)  # lines 27-30
next_noisy_images = model.reverse_diffusion_step(...)                     # line 32
next_noisy_images = next_noisy_images - gradients                         # line 31
```

This matches ZEA's own `posterior_sample` implementation ordering.

### 4. TV Regularization — Per-Step Inside Loop

TV smoothness is applied at every diffusion step (inside the outer loop), matching Algorithm 1 lines 35-36:

```python
tv_grad = compute_tv_gradient_azimuth(volume_noisy)  # axis 1
volume_noisy = volume_noisy - alpha_step * ZETA * tv_grad
```

### 5. SeqDiff Warm-Start

Both cold start and SeqDiff paths are implemented via the `USE_SEQDIFF` flag:

- Cold start: `noisy_bplanes = randn(...)`, `start_step = 0`
- SeqDiff: `noisy_bplanes = signal_rates * prev + noise_rates * noise`, `start_step = N_STEPS - SEQDIFF_TAU`

### 6. Hyperparameters

| Parameter | Value | Paper |
|-----------|-------|-------|
| N_STEPS (T) | 200 | 200 |
| SEQDIFF_TAU (tau') | 50 | 50 |
| OMEGA (gamma) | 35.0 | 35.0 |
| ZETA (zeta) | 0.001 | 0.001 |
| ACCEL_RATE (r) | 4 | 2, 3, 6, 10 |
| DYNAMIC_RANGE | -50 dB | -50 dB |

---

## What's Different (Data Limitations)

### 1. Pseudo-Volume from 2 Cycled Images

The CAMUS sample dataset only has 2 validation images. These get cycled to fill 112 elevation planes:

```
plane 0: img0, plane 1: img1, plane 2: img0, plane 3: img1, ...
```

With ACCEL_RATE=4, all observed planes (0, 4, 8, ...) are img0. Half the missing planes are img1 (a different patient), which the DPS guidance has zero information about.

**Impact**: Bimodal metrics. See README.md "Expected Metrics Behavior" for details.

### 2. Pretrained Model Domain Gap

The `diffusion-echonet-dynamic` model was trained on EchoNet-Dynamic (apical 4-chamber views). The paper trains on B-plane cross-sections from 3D volumes, which show different anatomical views.

### 3. No Real 3D Structure

The pseudo-volume has no spatial coherence along elevation — consecutive planes are from different patients. TV regularization can't meaningfully smooth between unrelated images.

---

## What Would Be Needed for True Reproduction

### Training Data
1. Acquire fully-sampled 3D volumetric ultrasound data (or find a public dataset)
2. Extract B-planes from each volume (all elevation cross-sections)
3. Clip to 50 dB, normalize to [-1, 1]
4. Train a 2D diffusion model on this B-plane data (~25 epochs, ~3.9M param U-Net)

### Inference Data
1. Acquire sparse 3D volumes (every r-th elevation plane)
2. Same preprocessing (50 dB, [-1, 1])
3. Apply interlocking acquisition pattern for temporal sequences

### Code Changes
1. Replace CAMUS loading with 3D volume loading and real B-plane extraction
2. Train a domain-matched diffusion model prior
