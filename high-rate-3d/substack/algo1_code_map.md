# Algorithm 1 → Code Map

Maps each line of Algorithm 1 (algo.tex) to `03_reconstruct_volume.py`.

---

## Inputs (line 10)

| Algo 1 | Symbol | Code | Location |
|--------|--------|------|----------|
| Partial volume | `Y ∈ R^{N_el/r × N_az × N_ax}` | `bplanes_gt * elevation_mask` → `measurements_all` | line 190 |
| Subsampling rate | `r ≥ 1` | `ACCEL_RATE = 4` | line 46 |
| Measurement matrix | `A ∈ {0,1}^{(N_el/r) × N_el}` | `elevation_mask` (1s at observed rows, 0s elsewhere) | line 178-179 |
| Score model | `ε_θ(·)` | `DiffusionModel.from_preset("diffusion-echonet-dynamic")` | line 60 |
| Guidance strength | `γ` | `OMEGA = 35.0` | line 48 |
| Diffusion steps | `T` | `N_STEPS = 200` | line 47 |
| Noise schedule | `α_τ, σ_τ` | `model.diffusion_schedule(diffusion_times)` | line 128, 132 |
| Smoothness strength | `ζ` | `ZETA = 0.001` | line 49 |

## Optional inputs (line 11)

| Algo 1 | Symbol | Code | Location |
|--------|--------|------|----------|
| Previous reconstruction | `X^prev` | `prev_recon` loaded from `reconstructed_volume.npy` | line 201 |
| Accelerated step | `τ'` | `SEQDIFF_TAU = 50`, `start_step = N_STEPS - SEQDIFF_TAU` | lines 52, 208 |

---

## Initialization (lines 16-23)

| Line | Algo 1 | Code | Location |
|------|--------|------|----------|
| 16 | `if X^prev available` | `if USE_SEQDIFF and os.path.exists(prev_recon_path)` | line 198 |
| 17 | `X_0 ← X^prev` (SeqDiff) | `prev_bplanes = np.transpose(prev_recon, (1,0,2,3))` | line 205 |
| 18 | `E ~ N(0, I)` | `noise = np.random.randn(...)` | line 216 |
| 19 | `X_τ ← α_τ' X_0 + σ_τ' E` (forward diffuse prev) | `noisy_bplanes = signal_rates * prev_bplanes + noise_rates * noise` | line 217 |
| 20 | `else` | `else:` (cold start branch) | line 221 |
| 21 | `X_τ ~ N(0, σ²_T I)` (cold start) | `noisy_bplanes = np.random.randn(N_az, N_el, N_ax, C)` | line 224 |
| 22 | `τ' ← T` | `start_step = 0` → loop starts at step 0 (= τ=T) | line 223 |

---

## Reverse diffusion loop (lines 25-37)

| Line | Algo 1 | Code | Location |
|------|--------|------|----------|
| 25 | `for τ = τ' to 0` | `for step in range(start_step, N_STEPS)` | line 243 |
| 26 | `for all B planes x_τ, y in X_τ, Y (parallel)` | `for batch_start in range(0, N_az, BATCH_SIZE)` (batched) | line 248 |
| 27 | `ε ← ε_θ(x_τ, τ)` (predict noise) | Inside `model.guidance_fn()` → returns `pred_noises` | line 140-147 |
| 28 | `x_{0|τ} ← (1/α_τ)(x_τ - σ_τ ε)` (denoise/prior) | Inside `model.guidance_fn()` → returns `pred_images` | line 140-147 |
| 29 | `M ← y - A x_{0|τ}` (measurement error) | Inside `model.guidance_fn()` → returns `error` | line 140-147 |
| 30 | `P ← (I - σ_τ ∇ε_θ)^T A^T` (projection) | Inside `model.guidance_fn()` → returns `gradients` | line 140-147 |
| 31 | `x_{0|τ} -= γ P M` (guidance step) | `next_noisy_images = next_noisy_images - gradients` | line 162 |
| 32 | `x_{τ-1} ← α_{τ-1} x_{0|τ} + σ_{τ-1} ε` (forward diffuse) | `model.reverse_diffusion_step(...)` | line 150-159 |
| 33 | `end for` (B planes) | End of batch loop | line 264 |
| 34 | Stack planes into `X_{τ-1}` | `volume_noisy = np.transpose(updated_bplanes, (1,0,2,3))` | line 268 |
| 35 | `V ← ∇ TV_az(X_{τ-1})` | `tv_grad = compute_tv_gradient_azimuth(volume_noisy)` | line 273 |
| 36 | `X_{τ-1} -= α_{τ-1} ζ V` (smoothness step) | `volume_noisy = volume_noisy - alphas[step+1] * ZETA * tv_grad` | line 275 |
| 37 | `end for` (diffusion steps) | End of `for step in range(start_step, N_STEPS)` | line 243 |

## Return (line 38)

| Line | Algo 1 | Code | Location |
|------|--------|------|----------|
| 38 | `return X_0` | `reconstructed = np.transpose(noisy_bplanes, ...)` → `np.save(save_path, reconstructed)` (TV-regularized volume) | line 287, 292 |

---

## Notes

### What's inside `model.guidance_fn()` (lines 27-30)
Lines 27-30 of Algorithm 1 are **all wrapped** inside a single `model.guidance_fn()` call (line 140-147). This ZEA function internally:
- Predicts noise (`pred_noises`)
- Computes Tweedie denoised estimate (`pred_images`)
- Computes measurement error
- Computes DPS projection and gradients

The code doesn't implement these steps manually — it delegates to ZEA.

### B-plane orientation (faithful to paper)
The code now processes **B-planes** as in the paper:
- B-plane j = `volume[:, j, :, :]` = (N_el, N_ax, C) at fixed azimuth j
- Missing data = missing **elevation rows** within each B-plane (every r-th row observed)
- Same elevation mask for all B-planes
- N_az B-planes processed in batches of `BATCH_SIZE`

### Volume axis convention
```
Volume: (N_el, N_az, N_ax, C) = (112, 112, 112, 1)
  axis 0 = elevation (subsampled by r)
  axis 1 = azimuth   (TV regularization applied here)
  axis 2 = axial/depth
  axis 3 = channel
```

### SeqDiff is now integrated
`03_reconstruct_volume.py` handles both cold start and SeqDiff warm-start via the `USE_SEQDIFF` flag (lines 198-227). When enabled, it loads a previous reconstruction, forward-diffuses it to τ', and starts the loop from `start_step = N_STEPS - SEQDIFF_TAU`. `04_seqdiff_temporal.py` remains as a standalone single-plane demo.

### Guidance ordering is correct
The code applies guidance AFTER `reverse_diffusion_step` (line 162). This matches ZEA's own `posterior_sample` implementation (diffusion.py lines 678-699) exactly.
