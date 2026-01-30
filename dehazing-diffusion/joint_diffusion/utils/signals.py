"""Utility functions for signals (PyTorch version)
Author(s): Tristan Stevens
Ported to PyTorch: Jan 2026

Note: Most functions here were TF-specific and only used by legacy datasets.
For ZEA training, these are not needed. Kept as stubs for compatibility.
"""
import numpy as np
import torch


def add_gaussian_noise(x, sigma):
    """Add i.i.d. Gaussian noise with std `sigma`.
    
    Args:
        x: Input tensor (PyTorch or numpy)
        sigma: Standard deviation of noise
    
    Returns:
        Noisy tensor of same type as input
    """
    if isinstance(x, torch.Tensor):
        return x + torch.randn_like(x) * sigma
    else:
        return x + np.random.randn(*x.shape).astype(x.dtype) * sigma


def grayscale_to_random_rgb(images, image_shape=None, single_channel=True):
    """Converts grayscale image to a random color channel (rgb).
    
    Note: This was a TF-specific function for MNIST/TMNIST datasets.
    Not used in ZEA training pipeline.
    
    Args:
        images: Batch of grayscale images
        image_shape: Output shape (not used in stub)
        single_channel: Whether output is single R, G, or B channel
    
    Returns:
        RGB images (stub returns grayscale repeated 3x)
    """
    if isinstance(images, torch.Tensor):
        if images.dim() == 3:
            images = images.unsqueeze(0)
        # Just repeat channels
        return images.repeat(1, 3, 1, 1) if images.shape[1] == 1 else images
    else:
        # numpy path
        if len(images.shape) == 3:
            images = np.expand_dims(images, axis=0)
        if images.shape[-1] == 1:
            return np.repeat(images, 3, axis=-1)
        return images


class RandomTranslation:
    """Random translation of images (PyTorch version).
    
    Note: This was a TF-specific layer for data augmentation.
    Not used in ZEA training pipeline.
    """

    def __init__(self, width_factor, height_factor):
        """
        Args:
            width_factor (float): Maximum fraction of image width for translation.
            height_factor (float): Maximum fraction of image height for translation.
        """
        self.width_factor = width_factor
        self.height_factor = height_factor

    def __call__(self, inputs, image_shape=None):
        """Apply random translations.
        
        Args:
            inputs: Tensor of shape (B, C, H, W) for PyTorch
            image_shape: Not used (compatibility)
        
        Returns:
            Translated tensor
        """
        if isinstance(inputs, torch.Tensor):
            B, C, H, W = inputs.shape
            # Generate random translations
            tx = (torch.rand(B) * 2 - 1) * self.width_factor * W
            ty = (torch.rand(B) * 2 - 1) * self.height_factor * H
            
            # Use grid_sample for translation
            # Create identity grid
            theta = torch.zeros(B, 2, 3, device=inputs.device)
            theta[:, 0, 0] = 1
            theta[:, 1, 1] = 1
            theta[:, 0, 2] = 2 * tx / W
            theta[:, 1, 2] = 2 * ty / H
            
            grid = torch.nn.functional.affine_grid(
                theta, inputs.size(), align_corners=False
            )
            return torch.nn.functional.grid_sample(
                inputs, grid, mode='bilinear', padding_mode='zeros', align_corners=False
            )
        else:
            # numpy fallback - just return as-is
            return inputs
