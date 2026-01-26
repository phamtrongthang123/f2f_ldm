#!/usr/bin/env python3
"""Synthesize 3D ultrasound RF volumes (tissue + haze) with ZEA.

This script stacks multiple elevational slices generated with a 2D linear-array
beamformer to approximate volumetric data.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# Ensure Keras backend is set before importing zea.
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("ZEA_DISABLE_CACHE", "1")

import zea  # noqa: E402
from zea.beamform.delays import compute_t0_delays_planewave  # noqa: E402
from zea.ops import Beamform  # noqa: E402
from zea.probes import Probe  # noqa: E402
from zea.scan import Scan  # noqa: E402
from zea.simulator import simulate_rf  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "dehazing-diffusion"))
import processing  # noqa: E402


def _make_probe(n_el: int, aperture: float, center_frequency: float, sampling_frequency: float):
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


def _random_scatterers_3d(
    rng: np.random.Generator,
    n_scat: int,
    xlims: tuple[float, float],
    ylims: tuple[float, float],
    zlims: tuple[float, float],
    z_bias: tuple[float, float] | None = None,
    magnitude_scale: float = 1.0,
):
    x = rng.uniform(xlims[0], xlims[1], n_scat)
    y = rng.uniform(ylims[0], ylims[1], n_scat)
    if z_bias is None:
        z = rng.uniform(zlims[0], zlims[1], n_scat)
    else:
        z_unit = rng.beta(z_bias[0], z_bias[1], n_scat)
        z = zlims[0] + z_unit * (zlims[1] - zlims[0])
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

    inputs = {beamformer.key: rf_data[0]}
    outputs = beamformer(**inputs, **beamformer_params)
    beamformed = outputs[beamformer.output_key]

    beamformed = np.array(beamformed)
    if beamformed.ndim == 3:
        beamformed = beamformed[:, :, 0]
    return beamformed.astype(np.float32)


def _slice_weighted_magnitudes(
    magnitudes: np.ndarray,
    y_positions: np.ndarray,
    slice_center: float,
    slice_sigma: float,
):
    if slice_sigma <= 0:
        return magnitudes
    weights = np.exp(-0.5 * ((y_positions - slice_center) / slice_sigma) ** 2).astype(
        np.float32
    )
    return magnitudes * weights


def _simulate_slice(
    positions: np.ndarray,
    magnitudes: np.ndarray,
    slice_center: float,
    slice_sigma: float,
    probe: Probe,
    scan: Scan,
    beamformer: Beamform,
    beamformer_params: dict,
):
    positions_shifted = positions.copy()
    positions_shifted[:, 1] = positions_shifted[:, 1] - slice_center
    magnitudes_weighted = _slice_weighted_magnitudes(
        magnitudes, positions[:, 1], slice_center, slice_sigma
    )
    return _simulate_frame(
        positions_shifted, magnitudes_weighted, probe, scan, beamformer, beamformer_params
    )


def _normalize_and_compand(data: np.ndarray, max_abs: float, mu: float):
    if max_abs <= 0:
        return data
    data_norm = np.clip(data / max_abs, -1.0, 1.0)
    data_01 = (data_norm + 1.0) / 2.0
    data_companded = processing.companding(data_01, image_range=(0, 1), mu=mu)
    if hasattr(data_companded, "numpy"):
        data_companded = data_companded.numpy()
    return np.array(data_companded, dtype=np.float32)


def _write_dataset(path: Path, data: np.ndarray, key: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{key: data})


def _generate_volume(
    rng: np.random.Generator,
    kind: str,
    probe: Probe,
    scan: Scan,
    beamformer: Beamform,
    beamformer_params: dict,
    xlims: tuple[float, float],
    ylims: tuple[float, float],
    tissue_zlims: tuple[float, float],
    haze_zlims: tuple[float, float],
    n_scat_tissue: int,
    n_scat_haze: int,
    slice_centers: np.ndarray,
    slice_sigma: float,
):
    if kind == "tissue":
        positions, magnitudes = _random_scatterers_3d(
            rng,
            n_scat_tissue,
            xlims,
            ylims,
            tissue_zlims,
            z_bias=(2.0, 2.0),
            magnitude_scale=1.0,
        )
    elif kind == "haze":
        positions, magnitudes = _random_scatterers_3d(
            rng,
            n_scat_haze,
            xlims,
            ylims,
            haze_zlims,
            z_bias=(0.8, 3.0),
            magnitude_scale=1.5,
        )
    else:
        raise ValueError(f"Unknown dataset kind: {kind}")

    slices = []
    for slice_center in slice_centers:
        frame = _simulate_slice(
            positions,
            magnitudes,
            slice_center,
            slice_sigma,
            probe,
            scan,
            beamformer,
            beamformer_params,
        )
        slices.append(frame)

    return np.stack(slices, axis=0)


def _generate_split(
    rng: np.random.Generator,
    n_samples: int,
    kind: str,
    probe: Probe,
    scan: Scan,
    beamformer: Beamform,
    beamformer_params: dict,
    xlims: tuple[float, float],
    ylims: tuple[float, float],
    tissue_zlims: tuple[float, float],
    haze_zlims: tuple[float, float],
    n_scat_tissue: int,
    n_scat_haze: int,
    slice_centers: np.ndarray,
    slice_sigma: float,
):
    volumes = []
    for _ in range(n_samples):
        volume = _generate_volume(
            rng,
            kind,
            probe,
            scan,
            beamformer,
            beamformer_params,
            xlims,
            ylims,
            tissue_zlims,
            haze_zlims,
            n_scat_tissue,
            n_scat_haze,
            slice_centers,
            slice_sigma,
        )
        volumes.append(volume)

    return np.stack(volumes, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="reprodcue 3D data")
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
    parser.add_argument("--ylims", type=float, nargs=2, default=[-4e-3, 4e-3])
    parser.add_argument("--tissue-zlims", type=float, nargs=2, default=[15e-3, 50e-3])
    parser.add_argument("--haze-zlims", type=float, nargs=2, default=[5e-3, 20e-3])
    parser.add_argument("--n-scat-tissue", type=int, default=2000)
    parser.add_argument("--n-scat-haze", type=int, default=3500)
    parser.add_argument("--n-slices", type=int, default=16)
    parser.add_argument("--slice-sigma", type=float, default=0.8e-3)
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

    slice_centers = np.linspace(args.ylims[0], args.ylims[1], args.n_slices)

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
        tuple(args.ylims),
        tuple(args.tissue_zlims),
        tuple(args.haze_zlims),
        args.n_scat_tissue,
        args.n_scat_haze,
        slice_centers,
        args.slice_sigma,
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
        tuple(args.ylims),
        tuple(args.tissue_zlims),
        tuple(args.haze_zlims),
        args.n_scat_tissue,
        args.n_scat_haze,
        slice_centers,
        args.slice_sigma,
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
        tuple(args.ylims),
        tuple(args.tissue_zlims),
        tuple(args.haze_zlims),
        args.n_scat_tissue,
        args.n_scat_haze,
        slice_centers,
        args.slice_sigma,
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
        tuple(args.ylims),
        tuple(args.tissue_zlims),
        tuple(args.haze_zlims),
        args.n_scat_tissue,
        args.n_scat_haze,
        slice_centers,
        args.slice_sigma,
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
        "ylims": args.ylims,
        "tissue_zlims": args.tissue_zlims,
        "haze_zlims": args.haze_zlims,
        "n_scat_tissue": args.n_scat_tissue,
        "n_scat_haze": args.n_scat_haze,
        "n_slices": args.n_slices,
        "slice_sigma": args.slice_sigma,
        "slice_centers": slice_centers.tolist(),
        "mu": args.mu,
        "npz_key": args.npz_key,
        "max_abs": float(max_abs),
    }
    metadata_path = output_root / "metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote 3D datasets to: {output_root}")


if __name__ == "__main__":
    main()
