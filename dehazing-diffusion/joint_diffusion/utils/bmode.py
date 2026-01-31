"""B-mode conversion utilities for RF data visualization.

Converts raw RF data to B-mode images using the ZEA beamforming pipeline
(envelope detection + log compression), matching the paper's display format.
"""

import os
os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("ZEA_DISABLE_CACHE", "1")

import numpy as np

import zea
from zea import init_device
from zea.probes import Probe
from zea.scan import Scan
from zea.beamform.delays import compute_t0_delays_planewave

init_device(verbose=False)

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

# Extent for imshow (in mm)
extent_mm = [xlims[0] * 1e3, xlims[1] * 1e3, zlims[1] * 1e3, zlims[0] * 1e3]

# Lazily initialized pipeline and parameters
_pipeline = None
_parameters = None


def _get_pipeline(dynamic_range=(-50, 0)):
    """Lazily initialize and cache the ZEA pipeline and parameters."""
    global _pipeline, _parameters
    if _pipeline is None:
        _pipeline = zea.Pipeline.from_default(enable_pfield=False, with_batch_dim=False, baseband=False)
        _parameters = _pipeline.prepare_parameters(probe, scan, dynamic_range=dynamic_range)
    return _pipeline, _parameters


def _single_rf_to_bmode(rf_frame, dynamic_range=(-50, 0)):
    """Convert a single RF frame (n_tx, n_ax, n_el) to B-mode image.

    Args:
        rf_frame: RF data with shape (n_tx, n_ax, n_el).
        dynamic_range: Tuple of (min_db, max_db) for log compression.

    Returns:
        B-mode image as uint8 array.
    """
    pipeline, parameters = _get_pipeline(dynamic_range)
    # Add channel dim: (n_tx, n_ax, n_el) -> (n_tx, n_ax, n_el, 1)
    rf_data = rf_frame[:, :, :, np.newaxis]
    inputs = {pipeline.key: rf_data}
    outputs = pipeline(**inputs, **parameters)
    image = outputs[pipeline.output_key]
    image = zea.display.to_8bit(image, dynamic_range=dynamic_range)
    return image


def undo_normalization(data, image_range=(-1, 1), mu=255, data_min=None, data_max=None):
    """Undo ZeaDataset normalization: image_range → mu-law expand → min-max restore.

    Args:
        data: Normalized data array.
        image_range: The (lo, hi) range used in step 3 of ZeaDataset.
        mu: μ-law companding parameter.
        data_min: Original data minimum (from ZeaDataset.data_min).
        data_max: Original data maximum (from ZeaDataset.data_max).

    Returns:
        Data in the original raw RF scale.
    """
    lo, hi = image_range
    # Step 3 inverse: image_range → [-1, 1]
    data = (data - lo) / (hi - lo) * 2.0 - 1.0
    # Step 2 inverse: mu-law expand
    data = np.sign(data) * ((1 + mu) ** np.abs(data) - 1.0) / mu
    # Step 1 inverse: [-1, 1] → original min-max range
    if data_min is not None and data_max is not None:
        data = (data + 1.0) / 2.0 * (data_max - data_min) + data_min
    return data


def rf_to_bmode(rf_data, dynamic_range=(-50, 0)):
    """Convert RF data batch to B-mode images.

    Args:
        rf_data: RF data with shape (B, C, H, W) where C=n_tx, H=n_ax, W=n_el
            (standard PyTorch BCHW format from the diffusion model).
        dynamic_range: Tuple of (min_db, max_db) for log compression.

    Returns:
        List of B-mode images (each a 2D uint8 array).
    """
    # rf_data shape: (B, C, H, W) = (batch, n_tx, n_ax, n_el)
    if rf_data.ndim == 4:
        frames = rf_data  # already (B, n_tx, n_ax, n_el)
    elif rf_data.ndim == 3:
        frames = rf_data[np.newaxis]  # add batch dim
    else:
        raise ValueError(f"Expected 3D or 4D RF data, got shape {rf_data.shape}")

    bmode_images = []
    for i in range(len(frames)):
        frame = frames[i]  # (n_tx, n_ax, n_el)
        bmode = _single_rf_to_bmode(frame, dynamic_range=dynamic_range)
        bmode_images.append(bmode)
    return bmode_images
