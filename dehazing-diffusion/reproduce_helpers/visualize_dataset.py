#!/usr/bin/env python3
"""Visualize synthesized ZEA dataset (tissue + haze) as B-mode images."""

import os
os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("ZEA_DISABLE_CACHE", "1")

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import zea
from zea import init_device
from zea.probes import Probe
from zea.scan import Scan
from zea.beamform.delays import compute_t0_delays_planewave
from zea.visualize import set_mpl_style

init_device(verbose=False)
set_mpl_style()

# Probe parameters (must match synthesis script)
n_el = 64
aperture = 20e-3
probe_geometry = np.stack(
    [np.linspace(-aperture / 2, aperture / 2, n_el), np.zeros(n_el), np.zeros(n_el)], axis=1
)
probe = Probe(probe_geometry=probe_geometry, center_frequency=5e6, sampling_frequency=20e6)

# Scan parameters
n_tx = 3
angles = np.linspace(-5, 5, n_tx) * np.pi / 180
sound_speed = 1540.0
xlims = (-20e-3, 20e-3)
zlims = (10e-3, 35e-3)
wavelength = sound_speed / probe.center_frequency
grid_size_x = int((xlims[1] - xlims[0]) / (0.5 * wavelength)) + 1
grid_size_z = int((zlims[1] - zlims[0]) / (0.5 * wavelength)) + 1

t0_delays = compute_t0_delays_planewave(probe_geometry, angles, sound_speed)
tx_apodizations = np.ones((n_tx, n_el)) * np.hanning(n_el)[None]

scan = Scan(
    n_tx=n_tx, n_el=n_el,
    center_frequency=probe.center_frequency, sampling_frequency=probe.sampling_frequency,
    probe_geometry=probe_geometry, t0_delays=t0_delays, tx_apodizations=tx_apodizations,
    element_width=np.linalg.norm(probe_geometry[1] - probe_geometry[0]),
    focus_distances=np.ones(n_tx) * np.inf, polar_angles=angles,
    initial_times=np.ones(n_tx) * 1e-6, n_ax=1024,
    xlims=xlims, zlims=zlims, grid_size_x=grid_size_x, grid_size_z=grid_size_z,
    lens_sound_speed=1000, lens_thickness=1e-3, n_ch=1,
    selected_transmits="all", sound_speed=sound_speed,
    apply_lens_correction=False, attenuation_coef=0.0,
)


def frame_to_bmode(rf_frame):
    """Convert stored RF frame (n_tx, n_ax, n_el) to B-mode image via ZEA pipeline."""
    # Add channel dim: (n_tx, n_ax, n_el) -> (n_tx, n_ax, n_el, 1)
    rf_data = rf_frame[:, :, :, np.newaxis]

    pipeline = zea.Pipeline.from_default(enable_pfield=False, with_batch_dim=False, baseband=False)
    parameters = pipeline.prepare_parameters(probe, scan, dynamic_range=(-50, 0))
    inputs = {pipeline.key: rf_data}

    outputs = pipeline(**inputs, **parameters)
    image = outputs[pipeline.output_key]
    image = zea.display.to_8bit(image, dynamic_range=(-50, 0))
    return image


def main():
    parser = argparse.ArgumentParser(description="Visualize ZEA synthesized dataset")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--n-samples", type=int, default=4)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    n_show = args.n_samples

    tissue_data = np.load(data_root / "tissue" / "train.npz")["rf"]
    haze_data = np.load(data_root / "haze" / "train.npz")["rf"]

    n_show = min(n_show, len(tissue_data), len(haze_data))
    fig, axes = plt.subplots(2, n_show, figsize=(4 * n_show, 8))
    if n_show == 1:
        axes = axes[:, np.newaxis]

    extent = [xlims[0] * 1e3, xlims[1] * 1e3, zlims[1] * 1e3, zlims[0] * 1e3]

    for i in range(n_show):
        tissue_img = frame_to_bmode(tissue_data[i])
        haze_img = frame_to_bmode(haze_data[i])

        axes[0, i].imshow(tissue_img, cmap="gray", extent=extent)
        axes[0, i].set_title(f"Tissue #{i}")
        axes[0, i].set_xlabel("X (mm)")
        axes[0, i].set_ylabel("Z (mm)")

        axes[1, i].imshow(haze_img, cmap="gray", extent=extent)
        axes[1, i].set_title(f"Haze #{i}")
        axes[1, i].set_xlabel("X (mm)")
        axes[1, i].set_ylabel("Z (mm)")

    plt.tight_layout()
    out_path = data_root / "visualization.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved visualization to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
