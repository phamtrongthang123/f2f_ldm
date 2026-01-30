"""Corruptors — PyTorch version.
Author(s): Tristan Stevens
"""
import abc
from pathlib import Path

import numpy as np
import torch

_CORRUPTORS = {}


def register_corruptor(cls=None, *, name=None):
    """A decorator for registering corruptor classes."""

    def _register(cls):
        local_name = name if name is not None else cls.__name__
        if local_name in _CORRUPTORS:
            raise ValueError(f"Already registered corruptor with name: {local_name}")
        _CORRUPTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_corruptor(name):
    """Get corruptor class for a given name."""
    return _CORRUPTORS[name]


class Corruptor(abc.ABC):
    """Corruptor abstract class."""

    def __init__(self, config, dataset_name=None, task=None, model=None,
                 verbose=True, **kwargs):
        super().__init__()
        self.config = config
        self.dataset_name = dataset_name
        self.task = "denoising" if task is None else task
        self.name = config.corruptor
        self.model = model
        self.verbose = verbose
        self.batch_size = config.batch_size
        self.image_shape = config.image_shape
        self.A = None  # measurement matrix
        self.noise = None
        self.blend_factor = getattr(config, "blend_factor", 1.0)

    def corrupt(self, images):
        """Corrupt input images with noise."""
        raise NotImplementedError


@register_corruptor(name="gaussian")
class GaussianCorruptor(Corruptor):
    """Gaussian corruptor, adds gaussian noise."""

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.noise_stddev = config.noise_stddev

    def corrupt(self, images):
        noise = torch.randn_like(images) * self.noise_stddev
        noisy_images = images + noise
        self.noise = noise
        return noisy_images


@register_corruptor(name="cs")
class CSCorruptor(Corruptor):
    """Compressed sensing corruptor."""

    def __init__(self, config, **kwargs):
        super().__init__(config, task="compressive-sensing", **kwargs)
        self.noise_stddev = config.noise_stddev
        self.subsample_factor = config.subsample_factor
        self.image_shape = config.image_shape

        self.n = int(np.prod(self.image_shape))
        self.m = int(self.n * (1 / self.subsample_factor))
        self.A = self.get_sensing_matrix()

    def corrupt(self, images):
        noise = torch.randn_like(images) * self.noise_stddev
        noisy_images = images + noise
        noisy_flat = noisy_images.reshape(-1, self.n)
        A_T = torch.from_numpy(self.A.T).float().to(images.device)
        return noisy_flat @ A_T

    def get_sensing_matrix(self):
        A = np.random.normal(0, 1 / np.sqrt(self.m), size=(self.m, self.n))
        return A.astype(np.float32)


@register_corruptor(name="haze")
class HazeCorruptor(Corruptor):
    """Haze corruptor for ultrasound dehazing (additive model y = x + h)."""

    def __init__(self, config, **kwargs):
        super().__init__(config, task="dehazing", **kwargs)
        self.noise_stddev = getattr(config, "noise_stddev", 0.5)
        self.blend_factor = getattr(config, "blend_factor", 1.0)
        self._haze_data = None
        self._haze_idx = 0
        
        # Try to load haze dataset for inference mode
        data_root = getattr(config, "data_root", None)
        if data_root:
            haze_path = Path(data_root) / "zea_synth" / "haze" / "val.npz"
            if haze_path.exists():
                import numpy as np
                npz_key = getattr(config, "npz_key", "rf")
                self._haze_data = np.load(haze_path)[npz_key]
                # Ensure channel-first format (N, C, H, W)
                if self._haze_data.ndim == 3:
                    self._haze_data = self._haze_data[:, np.newaxis, :, :]
                print(f"Loaded haze data from {haze_path}: shape {self._haze_data.shape}")

    def corrupt(self, tissue, haze=None):
        """Create hazy measurement: y = tissue + noise_stddev * haze.

        Args:
            tissue: clean tissue RF data (B, C, H, W)
            haze: haze RF data (B, C, H, W). If None, samples from loaded haze dataset.

        Returns:
            y: hazy measurement
        """
        import numpy as np
        
        # If haze not provided, sample from loaded haze dataset
        if haze is None:
            if self._haze_data is None:
                raise ValueError(
                    "HazeCorruptor requires haze data. Either provide haze argument "
                    "or ensure data_root config points to zea_synth with haze/val.npz"
                )
            # Get batch size
            batch_size = tissue.shape[0]
            
            # Sample haze (cycle through if needed)
            haze_indices = np.arange(self._haze_idx, self._haze_idx + batch_size) % len(self._haze_data)
            self._haze_idx = (self._haze_idx + batch_size) % len(self._haze_data)
            
            haze = self._haze_data[haze_indices]
            haze = torch.from_numpy(haze).to(tissue.device).float()
        
        # Additive haze model: y = tissue + gamma * haze
        gamma = self.noise_stddev
        y = tissue + gamma * haze
        self.noise = haze
        return y
