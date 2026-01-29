#!/usr/bin/env python3
"""Synthesize ultrasound RF datasets (tissue + haze) with ZEA.

Outputs raw beamformed RF data (float32) for training diffusion models,
following the methodology in "Dehazing Ultrasound using Diffusion Models".
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("ZEA_DISABLE_CACHE", "1")

import zea  # noqa: E402
from zea.beamform.delays import compute_t0_delays_planewave  # noqa: E402
from zea.ops import Beamform  # noqa: E402
from zea.probes import Probe  # noqa: E402
from zea.scan import Scan  # noqa: E402
from zea.simulator import simulate_rf  # noqa: E402


def _make_probe(n_el: int, aperture: float, center_frequency: float, sampling_frequency: float):
    """Build a linear-array probe model for ZEA simulations."""
    probe_geometry = np.stack(
        [
            np.linspace(-aperture / 2, aperture / 2, n_el),
            np.zeros(n_el),
            np.zeros(n_el),
        ],
        axis=1,
    )
    return Probe(
        probe_geometry=probe_geometry,
        center_frequency=center_frequency,
        sampling_frequency=sampling_frequency,
    )


def _compute_grid_sizes(xlims, zlims, wavelength):
    """Compute beamforming grid sizes from physical dimensions and wavelength.

    Uses half-wavelength sampling (Nyquist) as in the ZEA example notebook.
    """
    width = xlims[1] - xlims[0]
    height = zlims[1] - zlims[0]
    grid_size_x = int(width / (0.5 * wavelength)) + 1
    grid_size_z = int(height / (0.5 * wavelength)) + 1
    return grid_size_x, grid_size_z


def _make_scan(
    probe: Probe,
    n_tx: int,
    angles_rad: np.ndarray,
    sound_speed: float,
    xlims: tuple[float, float],
    zlims: tuple[float, float],
    grid_size_x: int,
    grid_size_z: int,
    n_ax: int,
):
    """Create a ZEA Scan that defines beamforming geometry and timing."""
    t0_delays = compute_t0_delays_planewave(
        probe_geometry=probe.probe_geometry,
        polar_angles=angles_rad,
        sound_speed=sound_speed,
    )
    tx_apodizations = np.ones((n_tx, probe.n_el)) * np.hanning(probe.n_el)[None]

    scan = Scan(
        n_tx=n_tx,
        n_el=probe.n_el,
        center_frequency=probe.center_frequency,
        sampling_frequency=probe.sampling_frequency,
        probe_geometry=probe.probe_geometry,
        t0_delays=t0_delays,
        tx_apodizations=tx_apodizations,
        element_width=np.linalg.norm(probe.probe_geometry[1] - probe.probe_geometry[0]),
        focus_distances=np.ones(n_tx) * np.inf,
        polar_angles=angles_rad,
        initial_times=np.ones(n_tx) * 1e-6,
        n_ax=n_ax,
        xlims=xlims,
        zlims=zlims,
        grid_size_x=grid_size_x,
        grid_size_z=grid_size_z,
        lens_sound_speed=1000,
        lens_thickness=1e-3,
        n_ch=1,
        selected_transmits="all",
        sound_speed=sound_speed,
        apply_lens_correction=False,
        attenuation_coef=0.0,
    )
    return scan


def _random_scatterers(
    rng: np.random.Generator,
    n_scat: int,
    xlims: tuple[float, float],
    zlims: tuple[float, float],
    z_bias: tuple[float, float] | None = None,
    magnitude_scale: float = 1.0,
):
    """Sample scatterer positions and magnitudes in 2D (x, z)."""
    x = rng.uniform(xlims[0], xlims[1], n_scat)
    if z_bias is None:
        z = rng.uniform(zlims[0], zlims[1], n_scat)
    else:
        z_unit = rng.beta(z_bias[0], z_bias[1], n_scat)
        z = zlims[0] + z_unit * (zlims[1] - zlims[0])
    y = np.zeros_like(x)
    positions = np.stack([x, y, z], axis=1).astype(np.float32)

    magnitudes = rng.rayleigh(scale=magnitude_scale, size=n_scat).astype(np.float32)
    magnitudes *= rng.choice([-1.0, 1.0], size=n_scat).astype(np.float32)
    return positions, magnitudes


def _simulate_frame(
    positions: np.ndarray,
    magnitudes: np.ndarray,
    probe: Probe,
    scan: Scan,
    beamform_op: Beamform,
    beamform_params: dict,
):
    """Simulate and beamform RF data.

    Returns beamformed RF as float32 array (grid_size_z, grid_size_x).
    This is pre-envelope-detection data suitable for diffusion model training.
    """
    _SCAT_BATCH = 200
    n_scat = positions.shape[0]
    rf_data = None
    for start in range(0, n_scat, _SCAT_BATCH):
        end = min(start + _SCAT_BATCH, n_scat)
        rf_batch = simulate_rf(
            scatterer_positions=positions[start:end],
            scatterer_magnitudes=magnitudes[start:end],
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
        rf_batch = np.array(rf_batch)
        if rf_data is None:
            rf_data = rf_batch
        else:
            rf_data += rf_batch

    # rf_data shape: (1, n_tx, n_ax, n_el, n_ch) — use first batch
    # Beamform only (no envelope detection, no log compression)
    beamformed = beamform_op(rf_data[0], **beamform_params)
    beamformed = np.array(beamformed)

    # Output may be (grid_z, grid_x) or (grid_z, grid_x, 1)
    if beamformed.ndim == 3:
        beamformed = beamformed[:, :, 0]

    return beamformed.astype(np.float32)


def _resize_rf(rf: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize RF data using bilinear interpolation (preserves float values)."""
    from scipy.ndimage import zoom
    zoom_factors = (target_h / rf.shape[0], target_w / rf.shape[1])
    return zoom(rf, zoom_factors, order=1).astype(np.float32)


def _write_dataset(path: Path, data: np.ndarray, key: str):
    """Write a compressed NPZ dataset to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{key: data})


def _generate_split(
    rng: np.random.Generator,
    n_samples: int,
    kind: str,
    probe: Probe,
    scan: Scan,
    beamform_op: Beamform,
    beamform_params: dict,
    xlims: tuple[float, float],
    tissue_zlims: tuple[float, float],
    haze_zlims: tuple[float, float],
    n_scat_tissue: int,
    n_scat_haze: int,
    output_size: tuple[int, int] | None = None,
):
    """Generate a split of beamformed RF data.

    Returns a float32 array with shape (n_samples, H, W).
    """
    frames = []
    for i in range(n_samples):
        if kind == "tissue":
            positions, magnitudes = _random_scatterers(
                rng,
                n_scat_tissue,
                xlims,
                tissue_zlims,
                z_bias=(2.0, 2.0),
                magnitude_scale=1.0,
            )
        elif kind == "haze":
            positions, magnitudes = _random_scatterers(
                rng,
                n_scat_haze,
                xlims,
                haze_zlims,
                z_bias=(0.8, 3.0),
                magnitude_scale=1.5,
            )
        else:
            raise ValueError(f"Unknown dataset kind: {kind}")

        frame = _simulate_frame(
            positions, magnitudes, probe, scan, beamform_op, beamform_params
        )

        if output_size is not None and (frame.shape[0] != output_size[0] or frame.shape[1] != output_size[1]):
            frame = _resize_rf(frame, output_size[0], output_size[1])

        frames.append(frame)
        print(f"  [{kind}] {i + 1}/{n_samples}", flush=True)

    return np.stack(frames, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="data/zea_synth")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--n-train", type=int, default=150)
    parser.add_argument("--n-val", type=int, default=38)
    # Output image size (after resizing from full-resolution beamformed image)
    parser.add_argument("--output-height", type=int, default=128,
                        help="Output image height (axial). Set to 0 to keep native resolution.")
    parser.add_argument("--output-width", type=int, default=64,
                        help="Output image width (lateral). Set to 0 to keep native resolution.")
    parser.add_argument("--n-ax", type=int, default=1024)
    parser.add_argument("--n-el", type=int, default=64)
    parser.add_argument("--aperture", type=float, default=20e-3)
    parser.add_argument("--center-frequency", type=float, default=5e6)
    parser.add_argument("--sampling-frequency", type=float, default=20e6)
    parser.add_argument("--sound-speed", type=float, default=1540.0)
    parser.add_argument("--n-tx", type=int, default=3)
    parser.add_argument("--tx-angle-deg", type=float, default=5.0)
    parser.add_argument("--xlims", type=float, nargs=2, default=[-20e-3, 20e-3])
    parser.add_argument("--tissue-zlims", type=float, nargs=2, default=[15e-3, 50e-3])
    parser.add_argument("--haze-zlims", type=float, nargs=2, default=[5e-3, 20e-3])
    parser.add_argument("--n-scat-tissue", type=int, default=2000)
    parser.add_argument("--n-scat-haze", type=int, default=3500)
    parser.add_argument("--npz-key", default="rf")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    probe = _make_probe(
        n_el=args.n_el,
        aperture=args.aperture,
        center_frequency=args.center_frequency,
        sampling_frequency=args.sampling_frequency,
    )

    # Compute proper grid sizes from wavelength (half-wavelength sampling)
    wavelength = args.sound_speed / args.center_frequency
    # Scan zlims = union of tissue and haze regions
    scan_zlims = (min(args.tissue_zlims[0], args.haze_zlims[0]),
                  max(args.tissue_zlims[1], args.haze_zlims[1]))
    grid_size_x, grid_size_z = _compute_grid_sizes(
        tuple(args.xlims), scan_zlims, wavelength
    )
    print(f"Wavelength: {wavelength * 1e3:.2f} mm")
    print(f"Beamforming grid: {grid_size_z} (axial) x {grid_size_x} (lateral)")

    angles_rad = np.linspace(-args.tx_angle_deg, args.tx_angle_deg, args.n_tx) * np.pi / 180
    scan = _make_scan(
        probe=probe,
        n_tx=args.n_tx,
        angles_rad=angles_rad,
        sound_speed=args.sound_speed,
        xlims=tuple(args.xlims),
        zlims=scan_zlims,
        grid_size_x=grid_size_x,
        grid_size_z=grid_size_z,
        n_ax=args.n_ax,
    )

    # Use Beamform only (no envelope detection, no log compression)
    # This outputs raw RF data for diffusion model training per the paper
    beamform_op = Beamform()
    beamform_params = beamform_op.prepare_parameters(probe, scan)

    # Determine output size
    output_size = None
    if args.output_height > 0 and args.output_width > 0:
        output_size = (args.output_height, args.output_width)
        print(f"Output size: {output_size[0]} x {output_size[1]} (resized from {grid_size_z} x {grid_size_x})")
    else:
        print(f"Output size: {grid_size_z} x {grid_size_x} (native resolution)")

    tissue_train = _generate_split(
        rng, args.n_train, "tissue", probe, scan, beamform_op, beamform_params,
        tuple(args.xlims), tuple(args.tissue_zlims),
        tuple(args.haze_zlims), args.n_scat_tissue, args.n_scat_haze, output_size,
    )
    tissue_val = _generate_split(
        rng, args.n_val, "tissue", probe, scan, beamform_op, beamform_params,
        tuple(args.xlims), tuple(args.tissue_zlims),
        tuple(args.haze_zlims), args.n_scat_tissue, args.n_scat_haze, output_size,
    )
    haze_train = _generate_split(
        rng, args.n_train, "haze", probe, scan, beamform_op, beamform_params,
        tuple(args.xlims), tuple(args.tissue_zlims),
        tuple(args.haze_zlims), args.n_scat_tissue, args.n_scat_haze, output_size,
    )
    haze_val = _generate_split(
        rng, args.n_val, "haze", probe, scan, beamform_op, beamform_params,
        tuple(args.xlims), tuple(args.tissue_zlims),
        tuple(args.haze_zlims), args.n_scat_tissue, args.n_scat_haze, output_size,
    )

    output_root = Path(args.output_root)
    _write_dataset(output_root / "tissue" / "train.npz", tissue_train, args.npz_key)
    _write_dataset(output_root / "tissue" / "val.npz", tissue_val, args.npz_key)
    _write_dataset(output_root / "haze" / "train.npz", haze_train, args.npz_key)
    _write_dataset(output_root / "haze" / "val.npz", haze_val, args.npz_key)

    actual_shape = tissue_train.shape
    metadata = {
        "seed": args.seed,
        "n_train": args.n_train,
        "n_val": args.n_val,
        "beamform_grid_size_z": grid_size_z,
        "beamform_grid_size_x": grid_size_x,
        "output_height": actual_shape[1],
        "output_width": actual_shape[2],
        "n_ax": args.n_ax,
        "n_el": args.n_el,
        "aperture": args.aperture,
        "center_frequency": args.center_frequency,
        "sampling_frequency": args.sampling_frequency,
        "sound_speed": args.sound_speed,
        "wavelength": wavelength,
        "n_tx": args.n_tx,
        "tx_angle_deg": args.tx_angle_deg,
        "xlims": args.xlims,
        "tissue_zlims": args.tissue_zlims,
        "haze_zlims": args.haze_zlims,
        "n_scat_tissue": args.n_scat_tissue,
        "n_scat_haze": args.n_scat_haze,
        "data_type": "rf",  # beamformed RF, pre-envelope detection
        "npz_key": args.npz_key,
        "dtype": "float32",
    }
    metadata_path = output_root / "metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nWrote datasets to: {output_root}")
    print(f"  tissue train: {tissue_train.shape}, dtype={tissue_train.dtype}")
    print(f"  haze train:   {haze_train.shape}, dtype={haze_train.dtype}")


if __name__ == "__main__":
    main()
