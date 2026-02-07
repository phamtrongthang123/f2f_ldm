#!/bin/bash
# Download pre-trained weights for drifting model training.
# SD-VAE (stabilityai/sd-vae-ft-mse) is auto-downloaded by diffusers.
# MoCo v2 checkpoint needs manual download.

set -euo pipefail

WEIGHT_DIR="${1:-./weights}"
mkdir -p "$WEIGHT_DIR"

echo "=== Downloading MoCo v2 ResNet-50 (800 epoch) ==="
MOCO_URL="https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar"
MOCO_PATH="$WEIGHT_DIR/moco_v2_800ep_pretrain.pth.tar"

if [ -f "$MOCO_PATH" ]; then
    echo "MoCo v2 checkpoint already exists at $MOCO_PATH"
else
    echo "Downloading from $MOCO_URL ..."
    wget -O "$MOCO_PATH" "$MOCO_URL"
    echo "Saved to $MOCO_PATH"
fi

echo ""
echo "=== SD-VAE (stabilityai/sd-vae-ft-mse) ==="
echo "Will be auto-downloaded by diffusers on first use."
echo "To pre-download, run:"
echo "  python -c \"from diffusers import AutoencoderKL; AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse')\""

echo ""
echo "Done."
