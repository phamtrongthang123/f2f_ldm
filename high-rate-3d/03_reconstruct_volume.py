"""
03_reconstruct_volume.py — Core Algorithm 1: DPS reconstruction of missing planes.

Implements the paper's volume reconstruction pipeline (Algorithm 1, algo.tex):
1. Load pseudo-volume (N_el, N_az, N_ax, C), extract B-planes along azimuth axis
2. Create elevation row mask: observed every r-th row, zeros elsewhere
3. For each diffusion step τ:
   - Batch B-planes through one diffusion step + DPS guidance
   - Transpose to volume, apply TV regularization along azimuth (axis 1)
   - Transpose back to B-planes for next step
4. Save reconstructed volume

Paper Algorithm 1 → ZEA API mapping:
  ε_θ(x_τ, τ)           → DiffusionModel (Eq. 4, eq:dsm)
  x_τ = α_τ x_0 + σ_τ ε → built-in forward diffusion (Eq. 2, eq:forward-diffusion)
  x_{0|τ} Tweedie       → built-in reverse diffusion (Eq. 3, eq:tweedie)
  y = Ax (measurement)   → inpainting operator (Eq. 5, eq:inverse-problem)
  M = diag(A^T A) (mask) → elevation row mask (Eq. 7, eq:observation_zf)
  DPS guidance (γ=35)    → manual gradient computation (Eq. 9-12, eq:dps-linear-*)
  TV smoothness (ζ)      → per-step TV gradient (Algo 1 line 35-36)
  SeqDiff warm-start     → forward-diffuse previous reconstruction (Algo 1 line 16-19)

Structure matches Algorithm 1 exactly:
  Volume: (N_el, N_az, N_ax, C)
  B-plane j: volume[:, j, :, :] = (N_el, N_ax, C)

  For τ = τ' to 1:                        # per-step (outer loop, line 25)
      For all B-planes j (batched):        # inner loop (line 26)
          One diffusion step + DPS guidance # lines 27-32
      EndFor                               # line 33
      Stack B-planes into X_{τ-1}          # line 34
      Apply TV_az to volume                # lines 35-36
  EndFor                                   # line 37
"""

import env_setup  # noqa: F401 — must be first

import os
import numpy as np
import keras
from keras import ops
from zea import init_device
from zea.models.diffusion import DiffusionModel

# --- Config ---
ACCEL_RATE = 4       # r ≥ 1, acceleration rate (Eq. 5, eq:inverse-problem)
N_STEPS = 200        # T, diffusion steps (Algo 1 line 25)
OMEGA = 35.0         # γ, guidance strength (Eq. 12, eq:dps-linear-4)
ZETA = 0.001         # ζ, smoothness strength (Algo 1 line 36)
BATCH_SIZE = 16      # Batch B-planes through model for memory
USE_SEQDIFF = False  # Enable SeqDiff warm-start from previous reconstruction
SEQDIFF_TAU = 50     # τ', warm-start diffusion step (Algo 1 line 11, 19)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

# --- Init device ---
init_device(verbose=False)

# --- Load model ---
print("Loading diffusion model...")
model = DiffusionModel.from_preset("diffusion-echonet-dynamic")
img_shape = model.input_shape  # (H, W, 1)
H, W = img_shape[0], img_shape[1]
print(f"Model loaded. Input shape: {img_shape}")

# --- Load pseudo-volume ---
volume_path = os.path.join(OUTPUT_DIR, "pseudo_volume.npy")
volume_gt = np.load(volume_path)
N_el, N_az, N_ax, C = volume_gt.shape
print(f"Loaded ground truth volume: {volume_gt.shape}")
print(f"  N_el={N_el}, N_az={N_az}, N_ax={N_ax}, C={C}")
assert (N_el, N_ax, C) == (H, W, 1), (
    f"B-plane shape (N_el, N_ax, C) = ({N_el}, {N_ax}, {C}) "
    f"does not match model input shape ({H}, {W}, 1)"
)


def compute_tv_gradient_azimuth(volume):
    """Compute TV gradient along the azimuth dimension (axis 1).

    Implements lines 35-36 of Algorithm 1:
      V ← ∇_{X_τ} TV_az(X_{τ-1})
      X_{τ-1} ← X_{τ-1} - α_{τ-1} ζ V

    Volume shape: (N_el, N_az, N_ax, C).
    TV_az computes total variation along azimuth (axis 1).

    Args:
        volume: Volume of shape (N_el, N_az, N_ax, C).

    Returns:
        TV gradient of same shape as volume.
    """
    diff = np.diff(volume, axis=1)  # (N_el, N_az-1, N_ax, C)
    eps = 1e-8
    norm = np.sqrt(diff**2 + eps)
    normalized = diff / norm

    # Divergence (adjoint of gradient)
    div = np.zeros_like(volume)
    div[:, :-1] += normalized
    div[:, 1:] -= normalized

    return -div  # Negative divergence as TV gradient


def one_diffusion_step(model, noisy_images, measurements, mask, step, n_steps, omega):
    """Perform one diffusion step with DPS guidance for a batch of B-planes.

    Args:
        model: DiffusionModel instance.
        noisy_images: Current noisy images, shape (B, H, W, C).
        measurements: Measurement data, shape (B, H, W, C).
        mask: Measurement mask, shape (B, H, W, C) or (1, H, W, C).
        step: Current diffusion step (0 to n_steps-1).
        n_steps: Total number of diffusion steps.
        omega: DPS guidance weight.

    Returns:
        Updated noisy images after one diffusion step.
    """
    num_images = noisy_images.shape[0]
    n_dims = len(model.input_shape)
    step_size = model.max_t / n_steps

    # Compute diffusion times for current step
    base_diffusion_times = ops.ones((num_images, *[1] * n_dims)) * model.max_t
    diffusion_times = base_diffusion_times - step * step_size
    noise_rates, signal_rates = model.diffusion_schedule(diffusion_times)

    # Compute next diffusion times
    next_diffusion_times = diffusion_times - step_size
    next_noise_rates, next_signal_rates = model.diffusion_schedule(next_diffusion_times)

    # Convert to tensors
    noisy_images_t = ops.convert_to_tensor(noisy_images)
    measurements_t = ops.convert_to_tensor(measurements)
    mask_t = ops.convert_to_tensor(mask)

    # DPS guidance: compute gradients and predictions (Algo 1 lines 27-31)
    gradients, (error, (pred_noises, pred_images)) = model.guidance_fn(
        noisy_images_t,
        measurements=measurements_t,
        noise_rates=noise_rates,
        signal_rates=signal_rates,
        omega=omega,
        mask=mask_t,
    )

    # Reverse diffusion step — DDIM (Algo 1 line 32)
    next_noisy_images = model.reverse_diffusion_step(
        shape=(num_images, *model.input_shape),
        pred_images=pred_images,
        pred_noises=pred_noises,
        signal_rates=signal_rates,
        next_signal_rates=next_signal_rates,
        next_noise_rates=next_noise_rates,
        seed=None,
        stochastic_sampling=False,
    )

    # Apply DPS guidance correction
    next_noisy_images = next_noisy_images - gradients
    pred_images = pred_images - gradients

    return np.array(next_noisy_images), np.array(pred_images)


# --- Elevation subsampling: observed rows in each B-plane ---
observed_rows = list(range(0, N_el, ACCEL_RATE))
missing_rows = [i for i in range(N_el) if i not in observed_rows]
print(f"\nAcceleration rate: {ACCEL_RATE}x")
print(f"Observed elevation rows ({len(observed_rows)}): {observed_rows[:8]}...")
print(f"Missing elevation rows ({len(missing_rows)}): {missing_rows[:8]}...")

# --- Create elevation row mask (same for ALL B-planes) ---
# Shape: (N_el, N_ax, C) = (H, W, 1) — matches model input
# 1s at observed elevation rows, 0s elsewhere
elevation_mask = np.zeros((N_el, N_ax, C), dtype=np.float32)
elevation_mask[observed_rows] = 1.0
print(f"Elevation mask shape: {elevation_mask.shape}, "
      f"observed fraction: {elevation_mask.mean():.2f}")

# --- Extract B-planes and create measurements (Algo 1 input: Y = AX) ---
# B-plane j = volume[:, j, :, :] → shape (N_el, N_ax, C)
# Transpose to (N_az, N_el, N_ax, C) for batch processing
bplanes_gt = np.transpose(volume_gt, (1, 0, 2, 3))  # (N_az, N_el, N_ax, C)
print(f"B-planes shape: {bplanes_gt.shape}")

# Measurements: observed elevation rows from GT, zeros elsewhere
measurements_all = bplanes_gt * elevation_mask[np.newaxis]  # (N_az, N_el, N_ax, C)

# Broadcast mask to batch: (1, N_el, N_ax, C)
mask_batch = elevation_mask[np.newaxis]  # (1, H, W, C) — broadcasts over batch

# --- SeqDiff initialization (Algo 1 lines 16-23) ---
prev_recon_path = os.path.join(OUTPUT_DIR, "reconstructed_volume.npy")

if USE_SEQDIFF and os.path.exists(prev_recon_path):
    # SeqDiff warm-start (Algo 1 lines 17-19)
    print(f"\nSeqDiff: loading previous reconstruction from {prev_recon_path}")
    prev_recon = np.load(prev_recon_path)
    print(f"Previous reconstruction shape: {prev_recon.shape}")

    # X_0 ← X^prev, extract B-planes (Algo 1 line 17)
    prev_bplanes = np.transpose(prev_recon, (1, 0, 2, 3))  # (N_az, N_el, N_ax, C)

    # Forward diffuse to τ': X_τ' ← α_τ' X_0 + σ_τ' ε (Algo 1 line 19)
    start_step = N_STEPS - SEQDIFF_TAU
    step_size = model.max_t / N_STEPS
    n_dims = len(model.input_shape)
    diffusion_times = np.ones((1, *[1] * n_dims)) * model.max_t - start_step * step_size
    noise_rates, signal_rates = model.diffusion_schedule(diffusion_times)
    noise_rates = np.array(noise_rates)
    signal_rates = np.array(signal_rates)

    noise = np.random.randn(*prev_bplanes.shape).astype(np.float32)
    noisy_bplanes = signal_rates * prev_bplanes + noise_rates * noise

    print(f"SeqDiff: forward-diffused to step {start_step}, "
          f"running {SEQDIFF_TAU} steps (τ'={SEQDIFF_TAU})")
else:
    # Cold start (Algo 1 lines 21-22)
    start_step = 0
    noisy_bplanes = np.random.randn(N_az, N_el, N_ax, C).astype(np.float32)
    if USE_SEQDIFF:
        print("\nSeqDiff enabled but no previous reconstruction found. Cold start.")
    print("Initializing all B-planes with noise (cold start)")

# --- Compute alpha schedule for TV regularization ---
alphas = []
for step in range(N_STEPS):
    diffusion_times = np.ones((1, 1, 1, 1)) * model.max_t - step * (model.max_t / N_STEPS)
    _, signal_rates = model.diffusion_schedule(diffusion_times)
    alphas.append(float(np.array(signal_rates)[0, 0, 0, 0]))
alphas = np.array(alphas)

# --- Main reconstruction loop (Algorithm 1 lines 25-37) ---
print(f"\nReconstructing volume: {N_az} B-planes over {N_STEPS} diffusion steps "
      f"(starting at step {start_step})...")
print(f"B-plane shape: ({N_el}, {N_ax}, {C}), batch size: {BATCH_SIZE}")
print("Structure: for τ → for all B-planes (batched) → DPS → stack → TV_az\n")

for step in range(start_step, N_STEPS):
    # --- Process ALL B-planes for one diffusion step (Algo 1 lines 26-33) ---
    updated_bplanes = np.empty_like(noisy_bplanes)
    pred_bplanes = np.empty_like(noisy_bplanes)

    for batch_start in range(0, N_az, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, N_az)
        batch_noisy = noisy_bplanes[batch_start:batch_end]
        batch_meas = measurements_all[batch_start:batch_end]

        batch_updated, batch_pred = one_diffusion_step(
            model,
            batch_noisy,
            batch_meas,
            mask_batch,
            step,
            N_STEPS,
            OMEGA,
        )

        updated_bplanes[batch_start:batch_end] = batch_updated
        pred_bplanes[batch_start:batch_end] = batch_pred

    # --- Stack B-planes into volume (Algo 1 line 34) ---
    # Transpose from (N_az, N_el, N_ax, C) → (N_el, N_az, N_ax, C)
    volume_noisy = np.transpose(updated_bplanes, (1, 0, 2, 3))

    # --- Apply TV regularization to volume (Algo 1 lines 35-36) ---
    # V ← ∇_{X_τ} TV_az(X_{τ-1})
    # X_{τ-1} ← X_{τ-1} - α_{τ-1} ζ V
    tv_grad = compute_tv_gradient_azimuth(volume_noisy)
    alpha_step = alphas[step] if step < len(alphas) else alphas[-1]
    volume_noisy = volume_noisy - alpha_step * ZETA * tv_grad

    # --- Transpose back to B-planes for next step ---
    noisy_bplanes = np.transpose(volume_noisy, (1, 0, 2, 3))

    # Keep denoised volume for final output
    reconstructed = np.transpose(pred_bplanes, (1, 0, 2, 3))

    # Progress logging
    if (step + 1) % 20 == 0 or step == start_step:
        tv_val = np.sum(np.abs(np.diff(reconstructed, axis=1)))
        print(f"Step {step+1}/{N_STEPS}: TV={tv_val:.4f}, alpha={alpha_step:.4f}")

print("\nReconstruction complete.")

# --- Save ---
save_path = os.path.join(OUTPUT_DIR, "reconstructed_volume.npy")
np.save(save_path, reconstructed)
print(f"\nSaved reconstructed volume to {save_path}")
print(f"Reconstructed volume shape: {reconstructed.shape}")
print("Volume reconstruction complete.")
