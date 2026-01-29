#!/usr/bin/env python3
"""Synthesize ultrasound RF datasets (tissue + haze) with ZEA."""
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

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
import processing  # noqa: E402


def _make_probe(n_el: int, aperture: float, center_frequency: float, sampling_frequency: float):
    """Build a linear-array probe model for ZEA simulations.

    Role:
        Defines the probe element positions and sampling parameters used by ZEA.

    Example:
        >>> probe = _make_probe(64, 20e-3, 5e6, 20e6)
        >>> probe.n_el
        64

    Output / Expectation:
        Returns a `zea.probes.Probe` with a (n_el, 3) geometry array. No randomness.
    """
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
    """Create a ZEA Scan that defines beamforming geometry and timing.

    Role:
        Encapsulates transmit angles, sampling grid, and physical parameters for
        RF simulation and beamforming.

    Example:
        >>> scan = _make_scan(probe, 3, np.array([0.0]), 1540.0, (-0.02, 0.02), (0.01, 0.05), 64, 128, 1024)
        >>> scan.grid_size_z
        128

    Output / Expectation:
        Returns a `zea.scan.Scan` configured for planar wave transmissions.
    """
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
    """Sample scatterer positions and magnitudes in 2D (x, z).

    Role:
        Generates random scatterers within x/z bounds, optionally biased in z
        to model tissue or haze distributions.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> pos, mag = _random_scatterers(rng, 10, (-1, 1), (0, 1))
        >>> pos.shape, mag.shape
        ((10, 3), (10,))

    Output / Expectation:
        Returns (positions, magnitudes) as float32 arrays. Positions are in meters,
        magnitudes are signed Rayleigh samples.
    """
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
    beamformer: Beamform,
    beamformer_params: dict,
):
    """Simulate and beamform one 2D RF frame from scatterers.

    Role:
        Runs ZEA RF simulation followed by beamforming to produce a 2D image.

    Example:
        >>> frame = _simulate_frame(positions, magnitudes, probe, scan, beamformer, params)
        >>> frame.shape
        (128, 64)

    Output / Expectation:
        Returns a float32 2D array (axial x lateral). Values are unnormalized RF.
    """
    # Batch scatterers to avoid GPU OOM. Inside simulate_rf, the peak tensor
    # has shape (n_scat, n_el, n_el, n_freq) in complex64. With n_el=64 and
    # n_ax=1024 (n_freq=513), each scatterer costs ~64*64*513*8 ≈ 16.8 MB.
    # 2000 scatterers → ~31 GB which exceeds A100-40GB.
    # Batching to 500 scatterers keeps peak usage under ~10 GB.
    _SCAT_BATCH = 500
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

    inputs = {beamformer.key: rf_data[0]}
    outputs = beamformer(**inputs, **beamformer_params)
    beamformed = outputs[beamformer.output_key]

    beamformed = np.array(beamformed)
    if beamformed.ndim == 3:
        beamformed = beamformed[:, :, 0]
    return beamformed.astype(np.float32)


def _normalize_and_compand(data: np.ndarray, max_abs: float, mu: float):
    """Normalize RF values and apply mu-law companding.

    Role:
        Scales RF values to [-1, 1], maps to [0, 1], then applies mu-law to
        compress dynamic range for learning.

    Example:
        >>> x = np.array([-2.0, 0.0, 2.0], dtype=np.float32)
        >>> y = _normalize_and_compand(x, max_abs=2.0, mu=255.0)
        >>> y.min() >= 0.0 and y.max() <= 1.0
        True

    Output / Expectation:
        Returns float32 array in [0, 1] with same shape as input.
    """
    if max_abs <= 0:
        return data
    data_norm = np.clip(data / max_abs, -1.0, 1.0)
    data_01 = (data_norm + 1.0) / 2.0
    data_companded = processing.companding(data_01, image_range=(0, 1), mu=mu)
    if hasattr(data_companded, "numpy"):
        data_companded = data_companded.numpy()
    return np.array(data_companded, dtype=np.float32)


def _write_dataset(path: Path, data: np.ndarray, key: str):
    """Write a compressed NPZ dataset to disk.

    Role:
        Ensures parent folders exist and saves data under a given key.

    Example:
        >>> _write_dataset(Path("out/train.npz"), np.zeros((1, 2, 3)), "rf")
        >>> Path("out/train.npz").is_file()
        True

    Output / Expectation:
        Writes `path` on disk. No return value.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{key: data})


def _generate_split(
    rng: np.random.Generator,
    n_samples: int,
    kind: str,
    probe: Probe,
    scan: Scan,
    beamformer: Beamform,
    beamformer_params: dict,
    xlims: tuple[float, float],
    tissue_zlims: tuple[float, float],
    haze_zlims: tuple[float, float],
    n_scat_tissue: int,
    n_scat_haze: int,
):
    """Generate a split of multiple 2D RF frames.

    Role:
        Repeats scatterer sampling and beamforming to build train/val datasets.

    Example:
        >>> frames = _generate_split(rng, 4, "haze", probe, scan, beamformer, params, (-0.02, 0.02), (0.015, 0.05), (0.005, 0.02), 2000, 3500)
        >>> frames.shape
        (4, 128, 64)

    Output / Expectation:
        Returns a float32 array with shape (n_samples, Z, X).
    """
    frames = []
    for _ in range(n_samples):
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
            positions, magnitudes, probe, scan, beamformer, beamformer_params
        )
        frames.append(frame)

    return np.stack(frames, axis=0)


def main():
    """CLI entry point for 2D ZEA synthesis.

    Role:
        Parses CLI args, generates tissue/haze frames, compands values, writes
        NPZ datasets and metadata.

    Example:
        $ python zea_synthesize_dataset.py --output-root data/zea_synth --n-train 10 --n-val 2

    Output / Expectation:
        Writes `tissue/` and `haze/` train/val NPZ files and a metadata JSON file.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="data/zea_synth")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--n-train", type=int, default=150)
    parser.add_argument("--n-val", type=int, default=38)
    parser.add_argument("--grid-size-z", type=int, default=128)
    parser.add_argument("--grid-size-x", type=int, default=64)
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
    parser.add_argument("--mu", type=float, default=255.0)
    parser.add_argument("--npz-key", default="rf")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    probe = _make_probe(
        n_el=args.n_el,
        aperture=args.aperture,
        center_frequency=args.center_frequency,
        sampling_frequency=args.sampling_frequency,
    )

    angles_rad = np.linspace(-args.tx_angle_deg, args.tx_angle_deg, args.n_tx) * np.pi / 180
    scan = _make_scan(
        probe=probe,
        n_tx=args.n_tx,
        angles_rad=angles_rad,
        sound_speed=args.sound_speed,
        xlims=tuple(args.xlims),
        zlims=(min(args.tissue_zlims[0], args.haze_zlims[0]), max(args.tissue_zlims[1], args.haze_zlims[1])),
        grid_size_x=args.grid_size_x,
        grid_size_z=args.grid_size_z,
        n_ax=args.n_ax,
    )

    beamformer = Beamform(num_patches=1, enable_pfield=False, with_batch_dim=False)
    beamformer_params = beamformer.prepare_parameters(probe, scan, dynamic_range=(-60, 0))

    tissue_train = _generate_split(
        rng,
        args.n_train,
        "tissue",
        probe,
        scan,
        beamformer,
        beamformer_params,
        tuple(args.xlims),
        tuple(args.tissue_zlims),
        tuple(args.haze_zlims),
        args.n_scat_tissue,
        args.n_scat_haze,
    )
    tissue_val = _generate_split(
        rng,
        args.n_val,
        "tissue",
        probe,
        scan,
        beamformer,
        beamformer_params,
        tuple(args.xlims),
        tuple(args.tissue_zlims),
        tuple(args.haze_zlims),
        args.n_scat_tissue,
        args.n_scat_haze,
    )

    haze_train = _generate_split(
        rng,
        args.n_train,
        "haze",
        probe,
        scan,
        beamformer,
        beamformer_params,
        tuple(args.xlims),
        tuple(args.tissue_zlims),
        tuple(args.haze_zlims),
        args.n_scat_tissue,
        args.n_scat_haze,
    )
    haze_val = _generate_split(
        rng,
        args.n_val,
        "haze",
        probe,
        scan,
        beamformer,
        beamformer_params,
        tuple(args.xlims),
        tuple(args.tissue_zlims),
        tuple(args.haze_zlims),
        args.n_scat_tissue,
        args.n_scat_haze,
    )

    max_abs = max(
        np.max(np.abs(tissue_train)),
        np.max(np.abs(tissue_val)),
        np.max(np.abs(haze_train)),
        np.max(np.abs(haze_val)),
    )

    tissue_train = _normalize_and_compand(tissue_train, max_abs, args.mu) * 255.0
    tissue_val = _normalize_and_compand(tissue_val, max_abs, args.mu) * 255.0
    haze_train = _normalize_and_compand(haze_train, max_abs, args.mu) * 255.0
    haze_val = _normalize_and_compand(haze_val, max_abs, args.mu) * 255.0

    output_root = Path(args.output_root)
    _write_dataset(output_root / "tissue" / "train.npz", tissue_train, args.npz_key)
    _write_dataset(output_root / "tissue" / "val.npz", tissue_val, args.npz_key)
    _write_dataset(output_root / "haze" / "train.npz", haze_train, args.npz_key)
    _write_dataset(output_root / "haze" / "val.npz", haze_val, args.npz_key)

    metadata = {
        "seed": args.seed,
        "n_train": args.n_train,
        "n_val": args.n_val,
        "grid_size_z": args.grid_size_z,
        "grid_size_x": args.grid_size_x,
        "n_ax": args.n_ax,
        "n_el": args.n_el,
        "aperture": args.aperture,
        "center_frequency": args.center_frequency,
        "sampling_frequency": args.sampling_frequency,
        "sound_speed": args.sound_speed,
        "n_tx": args.n_tx,
        "tx_angle_deg": args.tx_angle_deg,
        "xlims": args.xlims,
        "tissue_zlims": args.tissue_zlims,
        "haze_zlims": args.haze_zlims,
        "n_scat_tissue": args.n_scat_tissue,
        "n_scat_haze": args.n_scat_haze,
        "mu": args.mu,
        "npz_key": args.npz_key,
        "max_abs": float(max_abs),
    }
    metadata_path = output_root / "metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote datasets to: {output_root}")


if __name__ == "__main__":
    main()
