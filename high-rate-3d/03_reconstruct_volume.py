"""
03_reconstruct_volume.py — Core Algorithm 1: DPS reconstruction of missing planes.

Implements the paper's volume reconstruction pipeline (Algorithm 1, algo.tex):
1. Load pseudo-volume, subsample along elevation axis (keep every r-th plane)
2. For each diffusion step τ:
   - Process ALL N_el planes in parallel (one diffusion step + DPS guidance)
   - Stack planes into volume
   - Apply TV regularization across azimuth dimension (TV_az)
3. Save reconstructed volume

Paper Algorithm 1 → ZEA API mapping:
  ε_θ(x_τ, τ)           → DiffusionModel (Eq. 4, eq:dsm)
  x_τ = α_τ x_0 + σ_τ ε → built-in forward diffusion (Eq. 2, eq:forward-diffusion)
  x_{0|τ} Tweedie       → built-in reverse diffusion (Eq. 3, eq:tweedie)
  y = Ax (measurement)   → inpainting operator (Eq. 5, eq:inverse-problem)
  M = diag(A^T A) (mask) → EquispacedLines scanline mask (Eq. 7, eq:observation_zf)
  DPS guidance (γ=35)    → manual gradient computation (Eq. 9-12, eq:dps-linear-*)
  TV smoothness (ζ)      → per-step TV gradient (Algo 1 line 35-36)
  SeqDiff warm-start     → initial_step, initial_samples (Algo 1 line 16-19)

Structure matches Algorithm 1:
  For τ = T to 1:                      # per-step (outer loop)
      For all N_el planes (parallel):  # ALL planes (inner loop)
          One diffusion step + DPS guidance
      EndFor
      Stack planes
      Apply TV to volume               # per-step TV regularization
  EndFor
"""

import env_setup  # noqa: F401 — must be first

import os
import numpy as np
import keras
from keras import ops
from zea import init_device
from zea.models.diffusion import DiffusionModel
from zea.agent.selection import EquispacedLines

# --- Config ---
ACCEL_RATE = 4       # r ≥ 1, acceleration rate (Eq. 5, eq:inverse-problem)
N_STEPS = 200        # T, diffusion steps (Algo 1 line 25)
OMEGA = 35.0         # γ, guidance strength (Eq. 12, eq:dps-linear-4)
ZETA = 0.001         # ζ, smoothness strength (Algo 1 line 36)
SCANLINE_FACTOR = 2  # Scanline subsampling factor for inpainting mask
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
N_elev = volume_gt.shape[0]
print(f"Loaded ground truth volume: {volume_gt.shape}")

# --- Elevation subsampling ---
observed_indices = list(range(0, N_elev, ACCEL_RATE))
missing_indices = [i for i in range(N_elev) if i not in observed_indices]
print(f"Acceleration rate: {ACCEL_RATE}x")
print(f"Observed planes ({len(observed_indices)}): {observed_indices}")
print(f"Missing planes ({len(missing_indices)}): {missing_indices}")

# --- Create scanline subsampling mask ---
line_thickness = 2
agent = EquispacedLines(
    n_actions=W // line_thickness // SCANLINE_FACTOR,
    n_possible_actions=W // line_thickness,
    img_width=W,
    img_height=H,
)


def interpolate_from_neighbors(plane_idx, volume, observed_idx):
    """Linearly interpolate a missing plane from its two nearest observed neighbors."""
    below = max([i for i in observed_idx if i <= plane_idx], default=None)
    above = min([i for i in observed_idx if i >= plane_idx], default=None)

    if below is None:
        return volume[above]
    if above is None:
        return volume[below]
    if below == above:
        return volume[below]

    # Linear interpolation weight
    t = (plane_idx - below) / (above - below)
    return (1 - t) * volume[below] + t * volume[above]


def compute_tv_gradient_azimuth(volume):
    """Compute TV gradient along the azimuth dimension (axis 1).

    Implements lines 35-36 of Algorithm 1:
      V ← ∇_{X_τ} TV_az(X_{τ-1})
      X_{τ-1} ← X_{τ-1} - α_{τ-1} ζ V

    Volume shape: (N_el, N_az, N_ax, C) = (N_elev, H, W, C)
    TV_az computes total variation along azimuth (axis 1).

    Args:
        volume: Volume of shape (N_elev, H, W, C).

    Returns:
        TV gradient of same shape as volume.
    """
    # TV gradient along azimuth (axis 1)
    diff = np.diff(volume, axis=1)  # (N_el, N_az-1, N_ax, C)
    eps = 1e-8
    norm = np.sqrt(diff**2 + eps)
    normalized = diff / norm

    # Divergence (adjoint of gradient)
    div = np.zeros_like(volume)
    div[:, :-1] += normalized
    div[:, 1:] -= normalized

    return -div  # Return negative divergence as TV gradient


def one_diffusion_step(model, noisy_images, measurements, mask, step, n_steps, omega):
    """Perform one diffusion step with DPS guidance for a batch of planes.

    Args:
        model: DiffusionModel instance.
        noisy_images: Current noisy images, shape (B, H, W, C).
        measurements: Measurement data, shape (B, H, W, C).
        mask: Measurement mask, shape (B, H, W, C).
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

    # DPS guidance: compute gradients and predictions
    gradients, (error, (pred_noises, pred_images)) = model.guidance_fn(
        noisy_images_t,
        measurements=measurements_t,
        noise_rates=noise_rates,
        signal_rates=signal_rates,
        omega=omega,
        mask=mask_t,
    )

    # Reverse diffusion step (DDIM)
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


# --- Prepare measurements and masks for ALL planes (Algorithm 1 line 26) ---
print(f"\nPreparing measurements and masks for all {N_elev} planes...")
measurements_all = []
masks_all = []

for plane_idx in range(N_elev):
    if plane_idx in observed_indices:
        # Observed planes: use ground truth with full mask (all 1s)
        # This provides strong guidance to keep observed planes at their values
        measurement = volume_gt[plane_idx]  # (H, W, C)
        mask = np.ones((H, W, 1), dtype=np.float32)  # Full mask
    else:
        # Missing planes: use interpolated estimate with partial scanline mask
        initial_estimate = interpolate_from_neighbors(
            plane_idx, volume_gt, observed_indices
        )
        _, scanline_mask = agent.sample(batch_size=1)
        scanline_mask = keras.ops.expand_dims(scanline_mask, axis=-1)  # (1, H, W, 1)
        mask = np.array(scanline_mask)[0]  # (H, W, 1)
        measurement = np.where(mask, initial_estimate, -1.0)

    measurements_all.append(measurement)
    masks_all.append(mask)

measurements_all = np.stack(measurements_all, axis=0)  # (N_elev, H, W, C)
masks_all = np.stack(masks_all, axis=0)  # (N_elev, H, W, 1)

# --- Initialize ALL planes with noise (Algorithm 1 line 21: cold start) ---
print("Initializing all planes with noise (cold start)...")
noisy_planes = np.random.randn(N_elev, H, W, 1).astype(np.float32)

# --- Main reconstruction loop (Algorithm 1 structure) ---
print(f"\nReconstructing volume: {N_elev} planes over {N_STEPS} diffusion steps...")
print("Structure: per-step outer loop, ALL planes processed, TV applied each step\n")

# Compute alpha schedule for TV regularization (decreasing with tau)
# alpha[tau] corresponds to signal_rate at step tau
alphas = []
for step in range(N_STEPS):
    diffusion_times = np.ones((1, 1, 1, 1)) * model.max_t - step * (model.max_t / N_STEPS)
    _, signal_rates = model.diffusion_schedule(diffusion_times)
    alphas.append(float(np.array(signal_rates)[0, 0, 0, 0]))
alphas = np.array(alphas)

for step in range(N_STEPS):
    # --- Process ALL planes for one diffusion step (Algorithm 1 line 26-33) ---
    noisy_planes, pred_planes = one_diffusion_step(
        model,
        noisy_planes,
        measurements_all,
        masks_all,
        step,
        N_STEPS,
        OMEGA,
    )

    # --- Stack planes into volume (Algorithm 1 line 34) ---
    # X_{τ-1} is the NOISY states (noisy_planes), not denoised estimates (pred_planes)
    # pred_planes = x_{0|τ} (denoised), noisy_planes = x_{τ-1} (noisy)

    # --- Apply TV regularization to volume (Algorithm 1 lines 35-36) ---
    # V ← ∇_{X_τ} TV_az(X_{τ-1})
    # X_{τ-1} ← X_{τ-1} - α_{τ-1} ζ V
    tv_grad = compute_tv_gradient_azimuth(noisy_planes)  # TV on X_{τ-1} (noisy states)
    alpha_step = alphas[step] if step < len(alphas) else alphas[-1]

    # Apply TV correction to X_{τ-1} (line 36)
    noisy_planes = noisy_planes - alpha_step * ZETA * tv_grad

    # Keep denoised estimates for final output (paper returns X_0)
    reconstructed = pred_planes

    # Progress logging
    if (step + 1) % 20 == 0 or step == 0:
        tv_val = np.sum(np.abs(np.diff(reconstructed, axis=1)))  # TV along azimuth
        print(f"Step {step+1}/{N_STEPS}: TV={tv_val:.4f}, alpha={alpha_step:.4f}")

print("\nReconstruction complete.")

# --- Save ---
save_path = os.path.join(OUTPUT_DIR, "reconstructed_volume.npy")
np.save(save_path, reconstructed)
print(f"\nSaved reconstructed volume to {save_path}")
print(f"Reconstructed volume shape: {reconstructed.shape}")
print("Volume reconstruction complete.")
