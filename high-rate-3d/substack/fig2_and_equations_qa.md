# Q&A: Figure 2 Acquisition Assumption & Key Equations

## Q1: Figure 2 assumes we can capture multiple times — is this true?

Figure 2 shows the **sparse interlocking acquisition scheme**. The assumption is that the probe captures **multiple consecutive time steps** (t-1, t, t+1), each with a **different subset** of elevation planes in a staggered pattern.

This is physically true for **cardiac cine-loop acquisitions** — the probe sits on the patient's chest and continuously acquires frames over at least one cardiac cycle (~40 frames per volume in this dataset). The scheme works as follows (example with r=3):

- At time `t`: acquire elevation planes {1, 4, 7, 10, ...}
- At time `t+1`: acquire elevation planes {2, 5, 8, 11, ...}
- At time `t+2`: acquire elevation planes {3, 6, 9, 12, ...}

Across 3 consecutive frames, you collectively observe **all** elevation planes, but each individual frame is sparse. The **SeqDiff** mechanism (Section III-D) exploits this by warm-starting diffusion from the previous frame's reconstruction. It does not literally combine measurements across frames — instead it uses the temporal redundancy as a better initialization point (starting at diffusion step tau' instead of pure noise).

**When does this assumption hold?**
- Continuous acquisition scenarios: cardiac imaging, monitoring — yes.
- Single-shot acquisition — no, the interlocking scheme requires multiple captures over time.

---

## Q2: What is A in the equations, and which equations matter for reproduction?

### Definition of A

`A` is a **binary selection (subsampling) matrix**: `A ∈ {0,1}^{(N_el/r) × N_el}`.

With `N_el = 48` elevation planes and acceleration rate `r = 3`:
- A is `16 × 48`
- Each row has exactly one `1`, selecting one elevation plane
- `y = A x` gives only the 16 acquired planes (the compressed measurement)
- `A^T y = M ⊙ x` gives the zero-filled version (acquired planes in their correct positions, zeros elsewhere)

Concretely, if you acquire planes {0, 3, 6, ...}:

```
A = [[1,0,0,0,0,0,...],   # selects plane 0
     [0,0,0,1,0,0,...],   # selects plane 3
     [0,0,0,0,0,0,1,...], # selects plane 6
     ...]
```

The related masking operator is `M = diag(A^T A)`, which is simply a diagonal of 1s where planes are acquired and 0s where they are missing.

In practice, A operates **per B-plane** (per azimuth index). Each B-plane is a 2D image of size `(N_el, N_ax)`, and A selects rows from it.

### Which equations are essential to reproduce?

| Equation | Content | Role |
|----------|---------|------|
| **Eq 3** (forward diffusion) | `x_τ = α_τ x_0 + σ_τ ε` | Used in both training and every inference step |
| **Eq 4** (Tweedie's formula) | `x_{0|τ} = (1/α_τ)(x_τ + σ²_τ ∇log p)` | The denoising step |
| **Eq 5** (DSM training loss) | `L(θ) = E[‖ε_θ(x_τ, τ) − ε‖²]` | How to train the U-Net |
| **Eq 9** (DPS guidance) | `−γ (I − σ_τ ∇ε_θ)^T A^T (y − A x_{0|τ})` | The measurement-consistency step — this is the core contribution |
| **Algorithm 1** | Full inference loop | The actual step-by-step reconstruction procedure |

**Eq 9 is the most important equation beyond standard diffusion.** It combines the denoising prior with measurement consistency via DPS (Diffusion Posterior Sampling). Breaking it down:

1. `y − A x_{0|τ}` — **measurement error**: how far is the current denoised estimate from matching the actual acquired data?
2. `A^T (...)` — **backproject** the error into the full image space.
3. `(I − σ_τ ∇ε_θ)^T` — **projection** through the score network Jacobian (accounts for the diffusion model's learned structure).
4. `γ` — **guidance strength** hyperparameter (set to 35 in the paper).

### Practical note

If using the `zea` toolbox, you don't implement Eq 9 manually:

```python
model.posterior_sample(measurements, mask, n_samples, n_steps, omega)
```

handles the entire DPS loop internally. But if implementing from scratch, **Algorithm 1** (algo.tex) is the definitive reference — it specifies the exact order of operations including the TV smoothness step across azimuth planes.

### Algorithm 1 key steps (per diffusion step τ):

1. For each B-plane (parallelizable):
   - Predict noise: `ε = ε_θ(x_τ, τ)`
   - Denoise (prior): `x_{0|τ} = (1/α_τ)(x_τ − σ_τ ε)`
   - Measurement error: `M = y − A x_{0|τ}`
   - Projection: `P = (I − σ_τ ∇ε_θ)^T A^T`
   - Guidance: `x_{0|τ} -= γ · P · M`
   - Forward diffuse to next step: `x_{τ−1} = α_{τ−1} x_{0|τ} + σ_{τ−1} ε`
2. Stack all B-planes back into a volume.
3. Apply TV smoothness along azimuth: `X_{τ−1} -= α_{τ−1} ζ ∇TV_az(X_{τ−1})`

### Key hyperparameters for inference

| Parameter | Value |
|-----------|-------|
| Total diffusion steps T | 200 |
| SeqDiff start τ' | 50 |
| Guidance strength γ | 35 |
| TV smoothness ζ | 0.001 |
| Sampler | DDIM |
| Noise schedule | Cosine |

---

## Q3: Eq 1 shows the 3D volume as a matrix, but each slice is a cone — how can it be a matrix?

### Short answer

The matrix representation works because the data is stored in **polar coordinates** (angle × depth), not Cartesian coordinates (x × z). In polar coordinates, a cone/sector-shaped slice is just a regular rectangle.

### Concrete example

> *Source: the paper states data is in "native polar coordinate system" (Section III-C). The geometric explanation below is general ultrasound knowledge, not from the paper.*

Consider a single 2D slice (one A-plane at a fixed elevation angle). The probe fires a diverging wave and the beamformer resolves echoes at 64 azimuth angles, each with 400 depth samples along that direction:

```
Azimuth angle index j:    1      2      3     ...    64
                          |      |      |             |
                          v      v      v             v
                         /      |      \              \
                        /       |       \              \
                       /        |        \              \
                      /         |         \              \

Each beam direction gives you N_ax=400 depth samples along that ray.
```

**Physically**, these beams fan out in a sector. But **in the data**, each beam is indexed by its angle number `j`, and the samples along that beam are indexed by range sample number `k`. So a single slice is a regular matrix:

```
         azimuth angle j →
         j=1    j=2    j=3   ...  j=64
depth    ┌──────┬──────┬──────┬──────┐
k=1      │  ·   │  ·   │  ·   │  ·   │
k=2      │  ·   │  ·   │  ·   │  ·   │
k=3      │  ·   │  ·   │  ·   │  ·   │
  ↓      │  :   │  :   │  :   │  :   │
k=400    │  ·   │  ·   │  ·   │  ·   │
         └──────┴──────┴──────┴──────┘
```

Each element `x_{i,j} ∈ R^{N_ax}` in Eq 1 is the 400-sample depth vector for elevation angle `i`, azimuth angle `j`. Stack 48 elevation angles and you get the full tensor `X ∈ R^{48 × 64 × 400}`.

The cone/fan shape only appears after **scan conversion** — mapping from `(angle, angle, range)` to `(x, y, z)` Cartesian coordinates for display.

### What does this mean for near-field vs. far-field?

> *Source: general ultrasound knowledge — the paper does not discuss this.*

Because the beams all originate from roughly the same small probe aperture:

```
        Probe surface
        ============
         |||||||         ← at k=1 (shallow), all beams nearly overlap spatially
          /||||\
         / |||  \
        /  | |   \
       /   |  |   \      ← at k=200, beams have spread apart
      /    |   |    \
     /     |    |    \    ← at k=400 (deep), maximum spatial separation
```

- **Small k (shallow depth)**: adjacent columns in the matrix correspond to beams that are physically very close together — high spatial overlap
- **Large k (deep)**: adjacent columns correspond to well-separated spatial locations

However, **in the polar data itself there is no duplication** — each `(i, j, k)` is a distinct measurement at a unique `(elevation angle, azimuth angle, range)` coordinate. The beamformer separates signals by angle even if the physical resolution cells overlap at shallow depth.

After scan conversion to Cartesian, this manifests as the near-field looking "dense" (many polar samples compressed into a small area) and the far-field looking "sparse" (samples spread over a larger area, requiring interpolation).

### What the paper actually says vs. general knowledge

| Claim | Source |
|-------|--------|
| Data is in native polar coordinates, dimensions 48×64×400 | **Paper** (Section III-C) |
| Scan conversion to Cartesian is for visualization only | **Paper** (Section III-C) |
| Metrics computed in polar domain to avoid scan conversion artifacts | **Paper** (Section IV) |
| Beams overlap spatially near the probe and diverge at depth | **General ultrasound knowledge** |
| Near-field is spatially dense, far-field is sparse in Cartesian | **General ultrasound knowledge** |
| Working in polar is better for the neural network (uniform grid) | **General ultrasound knowledge / inference** |

---

## Q4: What is the acceleration rate r?

> *Source: Paper (Section III-A, Section IV).*

The acceleration rate `r ≥ 1` is simply **how many times fewer elevation planes you acquire** compared to the full volume. With `N_el = 48` total elevation planes:

| r | Planes acquired | Fraction kept | Effect |
|---|----------------|---------------|--------|
| 1 | 48 | 100% | No acceleration — full volume |
| 2 | 24 | 50% | Every 2nd plane |
| 3 | 16 | 33% | Every 3rd plane |
| 6 | 8 | 17% | Every 6th plane |
| 10 | ~5 | 10% | Every 10th plane |

It directly determines the size of the measurement matrix `A ∈ {0,1}^{(N_el/r) × N_el}` and how underdetermined the inverse problem is. Higher `r` = faster acquisition (fewer focused elevation transmissions needed → higher volume rate) but more missing data to reconstruct.

The paper tests r = 2, 3, 6, and 10:
- At **r = 2**, all methods (interpolation, U-Net, diffusion) perform comparably — there's not much missing data.
- At **r = 3 and above**, the diffusion model's advantage becomes clear. Baselines show visible interpolation artifacts, while the DM maintains strong perceptual and distortion metrics.

The term "acceleration rate" is borrowed from **MRI compressed sensing** literature, where the same concept describes the ratio of full k-space lines to acquired lines.

---

## Q5: The 3D volume is a cone/pyramid, not a cube — and what does "polar coordinates" actually mean?

### The physical shape

> *Source: general ultrasound knowledge. The paper does not discuss the volume geometry explicitly.*

Unlike CT/MRI which produces a rectangular cuboid with uniform spatial sampling, a 3D phased-array ultrasound volume is a **truncated pyramid** (or cone). Beams fan out from the probe in both azimuth and elevation. Near the probe surface, all beams converge to a small region (spatial overlap). At depth, they spread apart.

### What "polar coordinates" means concretely

> *Source: the paper states "native polar coordinate system" (Section III-C). The angle/range mapping below is general ultrasound knowledge — the paper does not report specific FOV or angle values.*

Each integer index in the `48 × 64 × 400` tensor maps to a physical coordinate:

**Azimuth (j = 1 to 64)** — maps to evenly spaced angles across the azimuth FOV:

```
Index j:    1        2        3       ...     64
Angle:    -45°    -43.6°   -42.1°   ...    +45°

Spacing = azimuth_FOV / (N_az - 1)
```

**Elevation (i = 1 to 48)** — maps to evenly spaced angles across the elevation FOV:

```
Index i:    1        2        3       ...     48
Angle:    -30°    -28.7°   -27.4°   ...    +30°

Spacing = elevation_FOV / (N_el - 1)
```

**Depth/range (k = 1 to 400)** — maps to evenly spaced range values:

```
Index k:    1        2        3       ...     400
Range:    0.0cm   0.0375cm  0.075cm  ...    15.0cm

Spacing = max_depth / (N_ax - 1)
```

(The exact FOV and depth values depend on the X5-1C probe and acquisition settings; these numbers are illustrative.)

So `x_{i,j} ∈ R^{400}` is the echo signal at elevation angle `i`, azimuth angle `j`, sampled at 400 evenly spaced depth points along that beam direction. The indices 1, 2, 3... are just integers, but they implicitly encode `(elevation_angle, azimuth_angle, range)`.

### Why this makes the data rectangular

In polar coordinates `(angle, angle, range)`, the grid is perfectly regular — equal spacing in each dimension. That's why it forms a clean `48 × 64 × 400` tensor, even though the physical volume is a cone/pyramid.

The cone shape only appears when you do **scan conversion**: mapping each `(θ_el, θ_az, range)` to Cartesian `(x, y, z)` via trigonometry:

```
x = range × sin(θ_az) × cos(θ_el)
y = range × sin(θ_el)
z = range × cos(θ_az) × cos(θ_el)
```

This is a non-uniform mapping — points near the probe (small range) get compressed together, points at depth get spread apart. The paper avoids this entirely by working in polar coordinates throughout training, inference, and evaluation.

### What the paper says vs. general knowledge

| Claim | Source |
|-------|--------|
| Data is in native polar coordinates, 48×64×400 | **Paper** (Section III-C) |
| Physical volume is a cone/pyramid, not a cube | **General ultrasound knowledge** |
| Integer indices map to evenly spaced angles and range values | **General ultrasound knowledge** |
| Scan conversion formula (angle,range) → (x,y,z) | **General ultrasound knowledge** |
| Actual FOV and angle values for the X5-1C probe | **Not reported in the paper** |
