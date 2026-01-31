"""
04_seqdiff_temporal.py — SeqDiff warm-start demo.

Demonstrates temporal acceleration: instead of running full 200-step
diffusion for every frame, use the previous frame's reconstruction
as a warm start and run only ~50 steps.

Paper mapping:
  Cold start (Frame 1):  200 steps, no warm-start
  SeqDiff (Frame 2):     initial_step=150, initial_samples=recon_t1 → only 50 steps
  Expected speedup:      ~4x
"""

import env_setup  # noqa: F401 — must be first

import os
import time
import numpy as np
import keras
import matplotlib.pyplot as plt
from zea import init_device
from zea.models.diffusion import DiffusionModel
from zea.ops import Pipeline, ScanConvert

# --- Config ---
N_STEPS = 200        # Full diffusion steps
SEQDIFF_TAU = 50     # SeqDiff: run only last tau steps
OMEGA = 35.0         # DPS guidance weight
PLANE_IDX = 0        # Index into the missing_indices list
ACCEL_RATE = 4       # Elevation acceleration rate
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

# --- Init device ---
init_device(verbose=False)

# --- Load model ---
print("Loading diffusion model...")
model = DiffusionModel.from_preset("diffusion-echonet-dynamic")

# --- Load both pseudo-volumes ---
volume_t1 = np.load(os.path.join(OUTPUT_DIR, "pseudo_volume.npy"))
volume_t2 = np.load(os.path.join(OUTPUT_DIR, "pseudo_volume_t2.npy"))
print(f"Volume t1 shape: {volume_t1.shape}")
print(f"Volume t2 shape: {volume_t2.shape}")

# --- Select a missing plane to reconstruct ---
observed_indices = list(range(0, volume_t1.shape[0], ACCEL_RATE))
missing_indices = [i for i in range(volume_t1.shape[0]) if i not in observed_indices]
if PLANE_IDX >= len(missing_indices):
    PLANE_IDX = 0
target_plane = missing_indices[PLANE_IDX]
nearest_obs = min(observed_indices, key=lambda x: abs(x - target_plane))
print(f"Target missing plane: {target_plane}, nearest observed: {nearest_obs}")

# --- Frame 1: Cold start (full 200 steps) ---
print(f"\n=== Frame 1: Cold start ({N_STEPS} steps) ===")
measurement_t1 = volume_t1[nearest_obs:nearest_obs+1]  # (1, H, W, C)
mask = np.ones_like(measurement_t1)

t_start = time.time()
recon_t1 = model.posterior_sample(
    measurements=measurement_t1,
    mask=mask,
    n_samples=1,
    n_steps=N_STEPS,
    omega=OMEGA,
    verbose=True,
)
recon_t1 = np.array(recon_t1)  # (1, 1, H, W, C)
time_cold = time.time() - t_start
print(f"Cold start time: {time_cold:.2f}s")
print(f"Recon t1 shape: {recon_t1.shape}, "
      f"range: [{recon_t1.min():.3f}, {recon_t1.max():.3f}]")

# --- Frame 2: SeqDiff warm-start (only last tau steps) ---
initial_step = N_STEPS - SEQDIFF_TAU  # Skip first 150 steps
print(f"\n=== Frame 2: SeqDiff warm-start "
      f"(initial_step={initial_step}, running {SEQDIFF_TAU} steps) ===")
measurement_t2 = volume_t2[nearest_obs:nearest_obs+1]

t_start = time.time()
recon_t2 = model.posterior_sample(
    measurements=measurement_t2,
    mask=mask,
    n_samples=1,
    n_steps=N_STEPS,
    initial_step=initial_step,
    initial_samples=recon_t1,  # warm-start from previous frame
    omega=OMEGA,
    verbose=True,
)
recon_t2 = np.array(recon_t2)  # (1, 1, H, W, C)
time_warm = time.time() - t_start
print(f"SeqDiff warm-start time: {time_warm:.2f}s")
print(f"Recon t2 shape: {recon_t2.shape}, "
      f"range: [{recon_t2.min():.3f}, {recon_t2.max():.3f}]")

# --- Compare ---
speedup = time_cold / time_warm if time_warm > 0 else float("inf")
print(f"\n=== Results ===")
print(f"Cold start:   {time_cold:.2f}s ({N_STEPS} steps)")
print(f"SeqDiff:      {time_warm:.2f}s ({SEQDIFF_TAU} steps)")
print(f"Speedup:      {speedup:.1f}x")

# --- Save results ---
np.save(os.path.join(OUTPUT_DIR, "seqdiff_recon_t1.npy"), recon_t1)
np.save(os.path.join(OUTPUT_DIR, "seqdiff_recon_t2.npy"), recon_t2)

# --- Visualization ---
pipeline = Pipeline([ScanConvert(order=2, jit_compile=False)])
parameters = pipeline.prepare_parameters(
    theta_range=[-0.78, 0.78],
    rho_range=[0, 1],
)


def scan_convert_img(img_2d):
    """Scan convert a single 2D image."""
    img = img_2d[np.newaxis]  # add batch dim
    out = pipeline(data=img, **parameters)["data"]
    return np.array(out[0])


gt_t1_cart = scan_convert_img(volume_t1[target_plane, ..., 0])
gt_t2_cart = scan_convert_img(volume_t2[target_plane, ..., 0])
recon_t1_cart = scan_convert_img(recon_t1[0, 0, ..., 0])
recon_t2_cart = scan_convert_img(recon_t2[0, 0, ..., 0])

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

axes[0].imshow(gt_t1_cart, cmap="gray", vmin=-1, vmax=1)
axes[0].set_title(f"GT Frame 1 (plane {target_plane})")
axes[1].imshow(recon_t1_cart, cmap="gray", vmin=-1, vmax=1)
axes[1].set_title(f"Cold Start ({N_STEPS} steps)\n{time_cold:.1f}s")
axes[2].imshow(gt_t2_cart, cmap="gray", vmin=-1, vmax=1)
axes[2].set_title(f"GT Frame 2 (plane {target_plane})")
axes[3].imshow(recon_t2_cart, cmap="gray", vmin=-1, vmax=1)
axes[3].set_title(f"SeqDiff ({SEQDIFF_TAU} steps)\n{time_warm:.1f}s, {speedup:.1f}x faster")

for ax in axes:
    ax.axis("off")

plt.suptitle("SeqDiff Temporal Acceleration Demo")
plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, "04_seqdiff_comparison.png")
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved visualization to {save_path}")
print("SeqDiff demo complete.")
