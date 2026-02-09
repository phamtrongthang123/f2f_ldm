# Drifting Models with Latent-MAE Feature Encoder

This repository implements **Drifting Models** for one-step generative modeling, using a custom **Latent-MAE** feature encoder for efficient and high-quality training on ImageNet.

## Environment Setup

```bash
conda create -n drifting python=3.11 -y
conda activate drifting
pip install -r requirements.txt
# or using uv
pip install uv
uv pip install -r requirements.txt
```

## Data Preparation

### 1. ImageNet Dataset
1. Download `ILSVRC2012_img_train.tar` and `ILSVRC2012_img_val.tar`.
2. Extract them into a folder structure suitable for `torchvision.datasets.ImageFolder` (e.g., `dataset/imagenet/train/...` and `dataset/imagenet/val/...`).
   * You can use [this script](https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh) for extraction.

### 2. Pre-compute Latents (Recommended)
To speed up training, pre-encode all ImageNet images into the SD-VAE latent space (32x32x4).

**Using SLURM:**
```bash
sbatch scripts/slurm_precompute.sh
```
*   Input: `./dataset/imagenet` (Default)
*   Output: `./dataset/imagenet_latents`

**Manual Run:**
```bash
python data/imagenet.py \
    --imagenet_dir ./dataset/imagenet \
    --output_dir ./dataset/imagenet_latents \
    --batch_size 64
```

## Training Pipeline

The training consists of two stages: first training the feature encoder, then training the generator.

### Stage 1: Train Latent-MAE Feature Encoder
We pre-train a ResNet-style Masked Autoencoder (MAE) directly on the VAE latents. This encoder provides the feature space for the drifting loss.

**Using SLURM:**
```bash
sbatch scripts/slurm_train_mae.sh
```
*   Trains for 192 epochs on single GPU (effective batch size 8192 via gradient accumulation).
*   Saves checkpoints to `checkpoints/mae/`.

**Manual Run:**
```bash
python train_mae.py \
    --latent_dir ./dataset/imagenet_latents \
    --output_dir ./checkpoints/mae \
    --base_width 256
```

### Stage 2: Train Drifting Generator
Train the DiT-based generator using the drifting objective and the frozen MAE encoder from Stage 1.

**Using SLURM:**
```bash
sbatch scripts/slurm_train_generator.sh
```
*   Config: `configs/ablation_default.yaml`
*   Requires `checkpoints/mae/mae_final.pt` (ensure this exists or update config).

**Manual Run (Multi-GPU):**
```bash
torchrun --nproc_per_node=8 train_imagenet.py \
    --config configs/ablation_default.yaml \
    --latent_dir ./dataset/imagenet_latents \
    --output_dir ./checkpoints \
    --bf16
```

## Configuration

Key configurations are in `configs/ablation_default.yaml`:
*   **Generator**: DiT-B/2 architecture.
*   **Feature Encoder**: Points to `latent_mae` and the checkpoint path.
*   **Drifting**: Temperatures, sample counts ($N_{pos}, N_{neg}, N_{uncond}$), and CFG settings.

## Project Structure

*   `models/`: DiT generator and Latent-MAE architecture.
*   `data/`: ImageNet loading and on-the-fly/pre-computed latent handling.
*   `scripts/`: SLURM submission scripts.
*   `drifting_loss.py`: Vectorized drifting loss computation.