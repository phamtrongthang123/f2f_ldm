#!/usr/bin/env python3
"""Synthesize ultrasound RF datasets (tissue + haze) with ZEA."""

import os
os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("ZEA_DISABLE_CACHE", "1")

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from zea import init_device
from zea.simulator import simulate_rf
from zea.probes import Probe
from zea.scan import Scan
from zea.beamform.delays import compute_t0_delays_planewave
from zea.beamform import phantoms

init_device(verbose=False)

# Probe parameters
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


def create_tissue_phantom():
    """Create tissue phantom using the fish phantom from ZEA."""
    positions = phantoms.fish()
    magnitudes = np.ones(len(positions), dtype=np.float32)
    return positions, magnitudes


def create_haze_phantom(rng, tissue_positions, tissue_magnitudes):
    """Add extra random scatterers to tissue phantom to simulate haze."""
    n_haze = rng.integers(30, 60)
    hx = rng.uniform(xlims[0], xlims[1], n_haze)
    hz = rng.uniform(zlims[0], zlims[1], n_haze)
    hy = np.zeros(n_haze)
    haze_pos = np.stack([hx, hy, hz], axis=1).astype(np.float32)
    haze_mag = rng.uniform(0.3, 0.8, n_haze).astype(np.float32)

    positions = np.concatenate([tissue_positions, haze_pos], axis=0)
    magnitudes = np.concatenate([tissue_magnitudes, haze_mag], axis=0)
    return positions, magnitudes


def run_simulation(positions, magnitudes):
    """Run ZEA RF simulation and return normalized frame."""
    rf_data = simulate_rf(
        scatterer_positions=positions,
        scatterer_magnitudes=magnitudes,
        probe_geometry=probe.probe_geometry,
        apply_lens_correction=scan.apply_lens_correction,
        lens_thickness=scan.lens_thickness,
        lens_sound_speed=scan.lens_sound_speed,
        sound_speed=scan.sound_speed,
        n_ax=scan.n_ax,
        center_frequency=probe.center_frequency,
        sampling_frequency=probe.sampling_frequency,
        t0_delays=scan.t0_delays,
        initial_times=scan.initial_times,
        element_width=scan.element_width,
        attenuation_coef=scan.attenuation_coef,
        tx_apodizations=scan.tx_apodizations,
    )
    # Extract first batch, all transmits, squeeze channel: (n_tx, n_ax, n_el)
    frame = np.array(rf_data[0, :, :, :, 0], dtype=np.float32)
    return frame


def synthesize_split(n_samples, rng):
    """Generate n_samples of paired tissue/haze RF frames."""
    tissue_frames = []
    haze_frames = []
    for _ in tqdm(range(n_samples)):
        # Create tissue phantom and simulate
        tissue_pos, tissue_mag = create_tissue_phantom()
        tissue_frame = run_simulation(tissue_pos, tissue_mag)
        tissue_frames.append(tissue_frame)

        # Create haze phantom from same tissue and simulate
        haze_pos, haze_mag = create_haze_phantom(rng, tissue_pos, tissue_mag)
        haze_frame = run_simulation(haze_pos, haze_mag)
        haze_frames.append(haze_frame)

    tissue_arr = np.stack(tissue_frames, axis=0)  # (N, n_tx, n_ax, n_el)
    haze_arr = np.stack(haze_frames, axis=0)
    return tissue_arr, haze_arr


def main():
    parser = argparse.ArgumentParser(description="Synthesize ZEA ultrasound RF dataset")
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-val", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.output_root)

    # Create output directories
    (out / "tissue").mkdir(parents=True, exist_ok=True)
    (out / "haze").mkdir(parents=True, exist_ok=True)

    # Generate training split
    print(f"Generating {args.n_train} training samples...")
    tissue_train, haze_train = synthesize_split(args.n_train, rng)
    np.savez(out / "tissue" / "train.npz", rf=tissue_train)
    np.savez(out / "haze" / "train.npz", rf=haze_train)
    print(f"  tissue train: {tissue_train.shape}, dtype={tissue_train.dtype}")
    print(f"  haze train:   {haze_train.shape}, dtype={haze_train.dtype}")

    # Generate validation split
    print(f"Generating {args.n_val} validation samples...")
    tissue_val, haze_val = synthesize_split(args.n_val, rng)
    np.savez(out / "tissue" / "val.npz", rf=tissue_val)
    np.savez(out / "haze" / "val.npz", rf=haze_val)
    print(f"  tissue val: {tissue_val.shape}, dtype={tissue_val.dtype}")
    print(f"  haze val:   {haze_val.shape}, dtype={haze_val.dtype}")

    print(f"Dataset saved to {out}")


if __name__ == "__main__":
    main()
