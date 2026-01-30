"""Corruptors — PyTorch version.
Author(s): Tristan Stevens
"""
import abc

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
        self.noise_stddev = getattr(config, "noise_stddev", 0.0)
        self.blend_factor = getattr(config, "blend_factor", 1.0)

    def corrupt(self, tissue, haze):
        """Create hazy measurement: y = (1-alpha)*tissue + alpha*haze.

        Args:
            tissue: clean tissue RF data (B, C, H, W)
            haze: haze RF data (B, C, H, W)

        Returns:
            y: hazy measurement
        """
        alpha = self.blend_factor
        y = (1 - alpha) * tissue + alpha * haze
        self.noise = haze
        return y
