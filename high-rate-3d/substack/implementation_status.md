# Implementation Status: Current Code vs Paper Requirements

This document compares the current reproduction code against the paper's requirements.

---

## Quick Summary

### Issues due to lacking 3D data:
| Aspect | Current Code | Paper Requirement | Status |
|--------|--------------|-------------------|--------|
| Data source | CAMUS 2D images | 3D volumetric cardiac ultrasound | Different |
| B-plane extraction | None (artificial stacking) | Real extraction from 3D | Missing |
| Pretrained model | EchoNet | Custom (trained on B-planes) | Domain gap |
| Measurement model | Scanline inpainting | Elevation plane subsampling | Workaround* |

*Scanline inpainting is used as a workaround to demonstrate DPS works. Without real 3D data, elevation plane subsampling is meaningless — the stacked images are unrelated, so there's nothing to reconstruct.

### Issues independent of data (code differences):
| Aspect | Current Code | Paper Requirement | Status |
|--------|--------------|-------------------|--------|
| Dynamic range | -40 dB | -50 dB | Easy fix |
| TV regularization | Post-hoc | Per-step inside diffusion loop | Algorithmic difference |

### Correct:
| Aspect | Current Code | Paper Requirement | Status |
|--------|--------------|-------------------|--------|
| Normalization | [-1, 1] | [-1, 1] | Correct |
| DPS posterior sampling | ✓ | ✓ | Correct |
| SeqDiff warm-starting | ✓ | ✓ | Correct |
| Hyperparameters (T, τ', γ, ζ) | 200, 50, 35, 0.001 | 200, 50, 35, 0.001 | Correct |

---

## What's Correct

### 1. Normalization to [-1, 1]

The code correctly normalizes images to the [-1, 1] range expected by the diffusion model:

```python
img = translate(img, DYNAMIC_RANGE, (-1, 1))
```

### 2. DPS Posterior Sampling

The reconstruction uses the correct DPS algorithm via zea:

```python
recon = model.posterior_sample(
    measurements=measurements,
    mask=mask,
    n_samples=1,
    n_steps=N_STEPS,
    omega=OMEGA,  # γ = 35
)
```

### 3. SeqDiff Temporal Warm-Starting

The temporal acceleration is correctly implemented:

```python
recon_t2 = model.posterior_sample(
    initial_step=N_STEPS - SEQDIFF_TAU,  # Start at step 150, not 200
    initial_samples=recon_t1,             # Warm-start from previous frame
    ...
)
```

### 4. TV Smoothness (Concept)

TV regularization is applied for cross-slice consistency. The strength `ζ = 0.001` matches the paper.

### 5. Hyperparameters

| Parameter | Current | Paper |
|-----------|---------|-------|
| N_STEPS | 200 | 200 |
| SEQDIFF_TAU (τ') | 50 | 50 |
| OMEGA (γ) | 35.0 | 35.0 |
| ZETA (ζ) | 0.001 | 0.001 |

---

## What's Wrong or Missing

### 1. Data Source: CAMUS Instead of 3D Volumetric

**Current**: Uses CAMUS dataset — standard 2D echocardiography images (apical 4-chamber view).

```python
dataset = Dataset("hf://zeahub/camus-sample/val", key="image")
```

**Paper**: Uses 100 in-vivo 3D volumetric cardiac cine-loops from a Philips EPIQ scanner with X5-1C matrix probe.

**Impact**: CAMUS provides standalone 2D images. There's no 3D volume to extract B-plane cross-sections from.

---

### 2. Viewing Angle Diversity and Anatomical Consistency

**Current**: The code uses CAMUS images (apical 4-chamber view) and artificially stacks them as "elevation planes":

```python
volume_t1 = processed[:N_ELEVATION]  # Stack 2D images as fake 3D
```

Two problems:
1. **Single viewing angle**: All images are apical 4-chamber views — the same orientation. The diffusion model only learns what one viewing angle looks like.
2. **No anatomical consistency**: The stacked images are from different patients/acquisitions. There's no spatial relationship between consecutive "planes."

**Paper**: B-planes are extracted from actual 3D volumes:

```python
# For a 3D volume of shape (N_el=48, N_az=64, N_ax=400)
B_plane_j = volume[:, j, :]  # Shape: (N_el, N_ax) = (48, 400) for azimuth index j
```

The paper's setup has:
1. **64 different azimuth positions**: Each azimuth index j gives a different B-plane cross-section. The diffusion model learns what these elevation slices look like across the full volume.
2. **Anatomical consistency**: Consecutive B-planes (j=0, j=1, j=2...) are neighboring slices through the same heart at the same moment. They're spatially coherent.

**Impact**: The EchoNet prior was trained on images with a specific anatomical appearance (apical 4-chamber views). The paper's B-plane cross-sections show the heart from different orientations and may look different. Additionally, the pseudo-volume has no real spatial structure to reconstruct — the stacked images are unrelated.

---

### 3. Dynamic Range: -40 dB Instead of -50 dB

**Current**:
```python
DYNAMIC_RANGE = (-40, 0)  # 40 dB range
```

**Paper**: Uses 50 dB dynamic range.

**Impact**: Minor, but could affect model behavior at low intensities.

---

### 4. Pretrained Model: Different Anatomical Appearance

**Current**: Uses `"diffusion-echonet-dynamic"` — trained on EchoNet-Dynamic (apical 4-chamber echocardiography videos).

**Paper**: Trained their own model on B-planes extracted from 3D volumes — elevation cross-sections of shape (48, 400).

**Impact**: The anatomical appearance in EchoNet images differs from the paper's B-plane cross-sections. EchoNet shows the heart from apical 4-chamber views. The paper's B-planes are elevation slices through 3D volumes, showing the heart from different orientations. The diffusion model learns the distribution of what it sees — if trained on one type of appearance, it may not generalize well to another.

---

### 5. Measurement Model: Scanline vs Elevation Planes

**Current**: Uses `EquispacedLines` — subsamples scanlines within a single 2D plane.

```python
agent = EquispacedLines(n_actions=..., n_possible_actions=..., ...)
```

This is a **workaround** because there's no real 3D data. Without spatially related planes, elevation subsampling is meaningless — you'd just be asking the model to hallucinate unrelated images. Scanline inpainting within a single coherent 2D image lets you demonstrate that DPS posterior sampling works.

**Paper**: The measurement matrix A selects **which elevation planes** are acquired, not which scanlines within a plane.

```
A ∈ {0,1}^{(N_el/r) × N_el}  — selects elevation indices
```

At acceleration rate r=3, you acquire planes 0, 3, 6, 9, ... and reconstruct the missing planes 1, 2, 4, 5, 7, 8, ... This only makes sense when consecutive planes are spatially adjacent slices through the same anatomy.

**Impact**: The reconstruction task is different. Current code does "inpainting within a plane." Paper does "interpolation across planes." Switching to the paper's measurement model requires real 3D volumetric data.

---

### 6. TV Regularization: Post-hoc Instead of Per-Step

**Current**: TV smoothness is applied once after all DPS steps complete:

```python
# After DPS loop finishes
volume_denoised = tv_denoise_elevation(reconstructed_volume, ZETA, TV_ITERATIONS)
```

**Paper**: TV is applied inside the diffusion loop at each step (Algorithm 1, line after guidance):

```
For τ = τ' to 0:
    ... DPS steps ...
    Stack planes, apply TV smoothness along azimuth  ← inside loop
```

**Impact**: Per-step TV integrates smoothness into the diffusion trajectory. Post-hoc TV is a separate denoising pass that may not respect the diffusion prior.

**Note**: This is purely a code difference, not a data workaround. Moving TV inside the loop could be done regardless of the data source. However, TV smoothness across planes only makes sense when planes are spatially adjacent (real 3D data) — with the current fake volume of unrelated images, TV smoothness is meaningless anyway.

---

### 7. Volume Shape

**Current**: `(16, 112, 112, 1)` — 16 "planes" of 112×112 images

This is **not a real volume**. CAMUS is a dataset of standard 2D echocardiography videos (apical views). The code artificially stacks 16 separate 2D images to create a fake "volume":

```python
volume_t1 = processed[:N_ELEVATION]  # Stack 16 unrelated 2D images
```

These 16 images are either different time frames or different patients — they have no spatial relationship to each other. The number 16 is arbitrary and has no connection to the paper's dimensions.

**Paper**: `(N_el, N_az, N_ax)` = `(48, 64, 400)` — 48 elevation samples, 64 azimuth samples, 400 axial (depth) samples

This is a real 3D volume where consecutive elevation indices are spatially adjacent slices through the same heart at the same moment.

The B-plane (what the diffusion model is trained on) has shape `(N_el, N_ax)` = `(48, 400)`.

---

## What Would Be Needed for True Reproduction

### Training Data
1. Acquire fully-sampled 3D volumetric ultrasound data (or find a public dataset)
2. Extract B-planes from each volume (all elevation cross-sections)
3. Clip to 50 dB, normalize to [-1, 1]
4. Train a 2D diffusion model on this multi-angle data (~25 epochs, ~3.9M param U-Net)

### Inference Data
1. Acquire sparse 3D volumes (every r-th elevation plane)
2. Same preprocessing (50 dB, [-1, 1])
3. Apply interlocking acquisition pattern for temporal sequences

### Code Changes
1. Replace CAMUS loading with 3D volume loading
2. Implement proper B-plane extraction
3. Change measurement model from scanlines to elevation planes
4. Move TV regularization inside the diffusion loop

---

## Current Status: Proof of Concept

The current code is a **proof-of-concept demonstration** that shows:
- The zea API works
- DPS posterior sampling runs
- SeqDiff warm-starting works
- The pipeline structure is correct

### What's missing for full reproduction:

**Data-related (need 3D volumetric data):**
1. Real 3D volumetric ultrasound data
2. Proper B-plane extraction from volumes
3. A diffusion model trained on B-plane data
4. Switch from scanline inpainting to elevation plane subsampling (only meaningful with real 3D data)

**Code-related (can be fixed without new data):**
1. Change `DYNAMIC_RANGE` from -40 dB to -50 dB
2. Move TV regularization inside the diffusion loop (per-step instead of post-hoc)
