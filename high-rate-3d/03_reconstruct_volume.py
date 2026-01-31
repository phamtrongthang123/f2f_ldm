"""
03_reconstruct_volume.py — Core Algorithm 1: DPS reconstruction of missing planes.

Implements the paper's volume reconstruction pipeline:
1. Load pseudo-volume, subsample along elevation axis (keep every r-th plane)
2. For each missing plane: use DPS posterior sampling with inpainting mask
3. Apply TV smoothness across elevation dimension (post-hoc)
4. Save reconstructed volume

Paper Algorithm 1 → ZEA API mapping:
  Score model ε_θ        → DiffusionModel.from_preset("diffusion-echonet-dynamic")
  DPS guidance (γ=35)    → posterior_sample(..., omega=35.0)
  Inpainting operator A  → built-in (mask kwarg)
  Cosine schedule        → built-in diffusion_schedule()
  DDIM, T=200 steps      → n_steps=200
  TV smoothness (ζ)      → post-hoc TV denoising across elevation axis
"""

import os
os.environ["KERAS_BACKEND"] = "jax"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import keras
from zea import init_device
from zea.models.diffusion import DiffusionModel
from zea.agent.selection import EquispacedLines

# --- Config ---
ACCEL_RATE = 4       # Keep every r-th plane (use 2 for quick test)
N_STEPS = 200        # Diffusion steps (use 50 for quick test)
OMEGA = 35.0         # DPS guidance weight (paper: γ=35)
ZETA = 0.001         # TV smoothness weight
TV_ITERATIONS = 50   # TV denoising iterations
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

# --- Init device ---
init_device(verbose=False)

# --- Load model ---
print("Loading diffusion model...")
model = DiffusionModel.from_preset("diffusion-echonet-dynamic")
img_shape = model.input_shape  # (H, W, 1)
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

# --- Create inpainting mask ---
# For each missing plane, we use the nearest observed plane as the measurement.
# The mask is all ones (fully observed measurement), and DPS generates a new
# image conditioned on that measurement.


def reconstruct_plane(model, measurement, n_steps, omega):
    """Reconstruct a missing plane using DPS posterior sampling.

    Args:
        model: DiffusionModel instance.
        measurement: Observed plane of shape (1, H, W, C).
        n_steps: Number of diffusion steps.
        omega: DPS guidance weight.

    Returns:
        Reconstructed plane of shape (H, W, C).
    """
    # Create a full mask (all pixels observed in the measurement)
    mask = np.ones_like(measurement)

    # DPS posterior sampling
    recon = model.posterior_sample(
        measurements=measurement,
        mask=mask,
        n_samples=1,
        n_steps=n_steps,
        omega=omega,
        verbose=False,
    )
    # recon shape: (batch=1, n_samples=1, H, W, C)
    recon = np.array(recon)
    return recon[0, 0]  # (H, W, C)


# --- Reconstruct missing planes ---
reconstructed = np.copy(volume_gt)

print(f"\nReconstructing {len(missing_indices)} missing planes...")
for i, plane_idx in enumerate(missing_indices):
    # Find nearest observed plane
    nearest = min(observed_indices, key=lambda x: abs(x - plane_idx))
    print(f"\n--- Plane {plane_idx} ({i+1}/{len(missing_indices)}) "
          f"[nearest observed: {nearest}] ---")

    measurement = volume_gt[nearest:nearest+1]  # (1, H, W, C)

    recon = reconstruct_plane(model, measurement, N_STEPS, OMEGA)
    print(f"  Reconstructed range: [{recon.min():.3f}, {recon.max():.3f}]")

    reconstructed[plane_idx] = recon


# --- TV smoothness across elevation ---
def tv_denoise_elevation(volume, zeta, n_iter):
    """Apply total variation denoising across the elevation dimension.

    Minimizes: ||volume - volume_input||^2 + zeta * TV(volume along elevation)
    Uses iterative gradient descent on the TV penalty.
    """
    print(f"\nApplying TV smoothness (zeta={zeta}, iterations={n_iter})...")
    vol = volume.copy().astype(np.float64)
    vol_input = vol.copy()

    for iteration in range(n_iter):
        # TV gradient along elevation (axis 0)
        diff = np.diff(vol, axis=0)  # (N-1, H, W, C)
        eps = 1e-8
        norm = np.sqrt(diff**2 + eps)
        normalized = diff / norm

        # Divergence (adjoint of gradient)
        div = np.zeros_like(vol)
        div[:-1] += normalized
        div[1:] -= normalized

        # Gradient step: data fidelity + TV
        grad = 2.0 * (vol - vol_input) - zeta * div
        step_size = 0.01
        vol -= step_size * grad

        if (iteration + 1) % 10 == 0:
            tv_val = np.sum(np.abs(diff))
            fidelity = np.sum((vol - vol_input) ** 2)
            print(f"  Iter {iteration+1}: TV={tv_val:.4f}, "
                  f"Fidelity={fidelity:.6f}")

    return vol.astype(np.float32)


reconstructed = tv_denoise_elevation(reconstructed, ZETA, TV_ITERATIONS)

# --- Save ---
save_path = os.path.join(OUTPUT_DIR, "reconstructed_volume.npy")
np.save(save_path, reconstructed)
print(f"\nSaved reconstructed volume to {save_path}")
print(f"Reconstructed volume shape: {reconstructed.shape}")
print("Volume reconstruction complete.")
