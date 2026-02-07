#!/bin/bash
# Download pre-trained weights for drifting model training.
# SD-VAE (stabilityai/sd-vae-ft-mse) is auto-downloaded by diffusers.
# Latent-MAE encoder must be pre-trained via train_mae.py.

set -euo pipefail

echo "=== SD-VAE (stabilityai/sd-vae-ft-mse) ==="
echo "Will be auto-downloaded by diffusers on first use."
echo "To pre-download, run:"
echo "  python -c \"from diffusers import AutoencoderKL; AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse')\""

echo ""
echo "=== Latent-MAE Feature Encoder ==="
echo "Must be pre-trained via train_mae.py before generator training."
echo "Example:"
echo "  python train_mae.py --latent_dir /path/to/precomputed_latents --output_dir ./checkpoints/mae"

echo ""
echo "Done."
