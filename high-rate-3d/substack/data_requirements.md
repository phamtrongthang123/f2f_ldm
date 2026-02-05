# What Data Do You Need?

This page covers data requirements for both **training** the diffusion model prior and **inference** (reconstruction).

## Quick Summary

| Stage | Data Needed |
|-------|-------------|
| **Training** | 2D B-plane slices extracted from fully-sampled 3D volumes |
| **Inference** | Sparsely-acquired 3D volumes (every r-th elevation plane) |

---

# Part 1: Training Data

## The Short Answer

For training the diffusion model prior, you need **2D B-plane images** — individual elevation slices extracted from 3D volumes. You do not need full 3D volumes at training time.

## What is a B-Plane?

Think of the 3D volume as a loaf of bread with shape `(N_el, N_az, N_ax)` = `(48, 64, 400)`.

A **B plane** is one **slice** of that loaf — you cut at a fixed azimuth position. Each slice is a 2D image of size `(N_el, N_ax)` = `(48, 400)`.

"Cross-section" just means you're cutting through the volume to reveal a 2D surface inside it. It's called "cross" because you're cutting **across** one dimension (azimuth in this case). Like slicing a cucumber — each round slice is a cross-section.

### CT Analogy

If you're familiar with CT imaging, a B-plane is like taking **one axial slice** from a CT stack — a single 2D image out of the full 3D volume.

The diffusion model is trained on a large collection of these individual 2D slices, just like you'd train a 2D model on axial CT slices rather than building a full 3D generative model.

## The Three Orthogonal Planes

The paper defines three orthogonal viewing planes:

- **A plane**: slice at a fixed elevation index → shows azimuth × axial (like a standard 2D ultrasound image)
- **B plane**: slice at a fixed azimuth index → shows elevation × axial
- **C plane**: slice at a fixed depth (axial) index → shows elevation × azimuth (a horizontal cut parallel to the probe face)

For training the prior, they use **B planes** — the slices that span the elevation direction, which is exactly the direction being sparsely sampled and needs reconstruction.

## How Much Training Data?

From each 3D volume of shape `(48, 64, 400)`, you get **64 independent 2D training images** (one B-plane per azimuth index).

The paper used:
- ~100 cine-loops
- ~40 frames per cine-loop
- 64 B-planes per volume

That's roughly `100 × 40 × 64 ≈ 256,000` 2D training images.

## Data Format Requirements

The training images must be:
- B-mode images in **polar coordinates** (before scan conversion to Cartesian)
- Clipped to **50 dB dynamic range**
- Normalized to **[-1, 1]**

## Why 2D Instead of 3D?

The paper deliberately uses a 2D diffusion model rather than a full 3D model. Three reasons:

1. **Curse of dimensionality** — 3D generative models need vastly more data and compute
2. **Computational cost** — a 2D U-Net with ~3.9M parameters is lightweight and fast
3. **Simpler training** — you just need a collection of 2D slices, not complex 3D data loaders

The 3D consistency between slices is handled at **inference time**, not training time — the posterior sampling algorithm applies TV (total variation) regularization across the azimuth direction to enforce smoothness between neighboring slices.

## Critical: You Must Train Your Own Prior

The zea toolbox provides a pretrained model `"diffusion-echonet-dynamic"` trained on **Echonet-Dynamic** — a large dataset of apical 4-chamber echocardiography videos.

**This is NOT a drop-in replacement for the paper's model.**

Why? The anatomical appearance in EchoNet images may differ from the paper's B-plane cross-sections. EchoNet shows the heart from apical 4-chamber views. The paper's B-planes are elevation slices through 3D volumes, showing the heart from different orientations. A diffusion model learns the distribution of what it sees — if the training images look different from the target domain, the prior may not transfer well.

To properly reproduce the paper's method, you must:
1. Acquire fully-sampled 3D volumetric ultrasound data
2. Extract all B-plane slices (64 per volume)
3. Train the 2D diffusion model on this B-plane data

The pretrained Echonet model is useful for understanding the API and running quick experiments, but the actual reconstruction quality depends on having a prior trained on data that matches your target domain.

---

# Part 2: Inference Data

At inference time, you're solving the actual reconstruction problem: filling in missing elevation planes from a sparsely-acquired 3D volume.

## Input: Sparse 3D Volume

The input is a **sparsely-acquired 3D volume** where only every r-th elevation index was measured. For example:

- Full volume shape: `(48, 64, 400)` — 48 elevation samples per B-plane
- At acceleration rate r=3: only 16 elevation indices acquired (indices 0, 3, 6, 9, ...)
- At acceleration rate r=6: only 8 elevation indices acquired

The acquired planes are real measurements. The missing planes are initially zero (zero-filled reconstruction) or can be linearly interpolated as a starting point.

## The Measurement Model

Mathematically, the sparse acquisition is modeled as:

```
y = A * x
```

Where:
- `x` is the full volume (what you want to reconstruct)
- `y` is the sparse measurement (what you actually acquired)
- `A` is a binary matrix that selects which elevation planes were acquired

The zero-filled observation is `y_zf = A^T * y` — you put the measured planes back in their correct positions, with zeros elsewhere.

## Interlocking Acquisition Scheme

For temporal sequences (like cardiac cine-loops), the paper uses an **interlocking acquisition pattern**:

- Frame 1: acquire planes 0, 3, 6, 9, ...
- Frame 2: acquire planes 1, 4, 7, 10, ...
- Frame 3: acquire planes 2, 5, 8, 11, ...
- Frame 4: acquire planes 0, 3, 6, 9, ... (cycle repeats)

This way, consecutive frames capture **complementary** slices. The temporal consistency (SeqDiff) can leverage this — neighboring frames have information about different parts of the volume.

## Output: Reconstructed Full Volume

The diffusion model reconstructs the full `(48, 64, 400)` volume by:
1. Running the 2D diffusion model on each B-plane independently (parallelizable)
2. Using posterior sampling (DPS) to ensure consistency with the measured planes
3. Applying TV regularization across azimuth for cross-slice smoothness

## Data Format (Same as Training)

Inference data should be in the same format as training:
- **Polar coordinates** (before scan conversion)
- **50 dB dynamic range**
- **Normalized to [-1, 1]**

After reconstruction, you can scan-convert to Cartesian coordinates for visualization.

## Temporal Warm-Starting (SeqDiff)

For video sequences, you don't start each frame from scratch. Instead:
- Take the previous frame's reconstruction
- Forward-diffuse it to an intermediate noise level (τ' = 50 out of T = 200 steps)
- Use that as the starting point for the current frame

This exploits temporal continuity — the heart doesn't change much between consecutive frames — and gives:
- Faster inference (only 50 steps instead of 200)
- Better temporal consistency (no flickering between frames)

---

# Summary Table

| Aspect | Training | Inference |
|--------|----------|-----------|
| **Data type** | 2D B-plane slices | Sparse 3D volumes |
| **Source** | Fully-sampled 3D acquisitions | Sparsely-acquired 3D acquisitions |
| **Volume needed?** | No (just slices) | Yes (sparse volume as input) |
| **Format** | Polar, 50 dB, [-1, 1] | Polar, 50 dB, [-1, 1] |
| **Elevation coverage** | All 48 elevation indices | Subset (every r-th index) |
| **Ground truth needed?** | Yes (for training loss) | Optional (for evaluation only) |
