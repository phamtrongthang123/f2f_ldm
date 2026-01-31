"""
03_reconstruct_volume.py — Core Algorithm 1: DPS reconstruction of missing planes.

Implements the paper's volume reconstruction pipeline:
1. Load pseudo-volume, subsample along elevation axis (keep every r-th plane)
2. For each missing plane: interpolate from neighbors, then refine with DPS
   using a partial scanline mask (EquispacedLines) — the model inpaints
   the unobserved scanlines guided by the diffusion prior.
3. Apply TV smoothness across elevation dimension (post-hoc)
4. Save reconstructed volume

Paper Algorithm 1 → ZEA API mapping:
  Score model ε_θ        → DiffusionModel.from_preset("diffusion-echonet-dynamic")
  DPS guidance (γ=35)    → posterior_sample(..., omega=35.0)
  Inpainting operator A  → built-in (mask kwarg) with EquispacedLines
  Cosine schedule        → built-in diffusion_schedule()
  DDIM, T=200 steps      → n_steps=200
  TV smoothness (ζ)      → post-hoc TV denoising across elevation axis
"""

import env_setup  # noqa: F401 — must be first

import os
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
# The paper uses a sparse measurement model. For per-plane DPS inpainting,
# we create a scanline mask via EquispacedLines (as in the zea example).
# The measurement is: observed scanlines from an interpolated estimate,
# and DPS fills in the missing scanlines guided by the learned prior.
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


def reconstruct_plane(model, initial_estimate, agent, n_steps, omega):
    """Reconstruct a missing plane using DPS posterior sampling.

    Uses partial scanline masking: the initial estimate provides observed
    scanlines, and DPS inpaints the missing ones using the diffusion prior.

    Args:
        model: DiffusionModel instance.
        initial_estimate: Interpolated plane of shape (H, W, C).
        agent: EquispacedLines agent for creating scanline masks.
        n_steps: Number of diffusion steps.
        omega: DPS guidance weight.

    Returns:
        Reconstructed plane of shape (H, W, C).
    """
    # Create measurement: keep only masked scanlines from interpolated estimate
    estimate_batch = initial_estimate[np.newaxis]  # (1, H, W, C)
    _, mask = agent.sample(batch_size=1)
    mask = keras.ops.expand_dims(mask, axis=-1)  # (1, H, W, 1)
    mask = np.array(mask)

    measurements = np.where(mask, estimate_batch, -1.0)

    # DPS posterior sampling — inpaint missing scanlines
    recon = model.posterior_sample(
        measurements=measurements,
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
    # Interpolate initial estimate from neighboring observed planes
    initial_estimate = interpolate_from_neighbors(
        plane_idx, volume_gt, observed_indices
    )
    print(f"\n--- Plane {plane_idx} ({i+1}/{len(missing_indices)}) ---")

    recon = reconstruct_plane(model, initial_estimate, agent, N_STEPS, OMEGA)
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
