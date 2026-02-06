"""
eval_diffusion.py — Evaluate a diffusion model with FID score.

Generates samples from the model and computes FID against the
EchoNet-Dynamic val set. Also saves sample image grids.

Usage:
  # Pretrained ZEA model:
  python eval_diffusion.py --model diffusion-echonet-dynamic

  # Trained checkpoint:
  python eval_diffusion.py --model outputs/training/final_model
"""

import env_setup  # noqa: F401 — must be first

import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from zea import init_device
from zea.models.diffusion import DiffusionModel
from zea.visualize import plot_image_grid


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate diffusion model")
    p.add_argument("--model", default="diffusion-echonet-dynamic",
                    help="ZEA preset name or path to saved model directory")
    p.add_argument("--data-dir", default="EchoNet-Dynamic",
                    help="Dir with val_frames.npy")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-samples", type=int, default=1000,
                    help="Number of generated samples for FID")
    p.add_argument("--n-real", type=int, default=10000,
                    help="Number of real images for FID")
    p.add_argument("--n-steps", type=int, default=200,
                    help="Diffusion steps for sampling")
    p.add_argument("--output-dir", default="outputs/eval")
    return p.parse_args()


# ---------------------------------------------------------------------------
# FID helpers
# ---------------------------------------------------------------------------

def build_feature_extractor():
    """InceptionV3 feature extractor (pool layer, 2048-dim)."""
    print("Loading InceptionV3 for FID ...")
    inception = keras.applications.InceptionV3(
        include_top=False, pooling="avg", input_shape=(299, 299, 3))
    inception.trainable = False
    return inception


def preprocess_for_inception(images_nhw1):
    """Convert (B, 112, 112, 1) float32 [-1,1] → (B, 299, 299, 3) for Inception."""
    # Resize to 299x299
    resized = jax.image.resize(images_nhw1, (*images_nhw1.shape[:1], 299, 299, 1),
                                method="bilinear")
    # Grayscale → RGB
    rgb = jnp.tile(resized, (1, 1, 1, 3))
    # InceptionV3 expects [-1, 1] which we already have
    return rgb


def extract_features(inception, images_nhw1, batch_size=32):
    """Extract InceptionV3 features from images in batches."""
    n = images_nhw1.shape[0]
    features = []
    for i in range(0, n, batch_size):
        batch = images_nhw1[i:i + batch_size]
        batch_rgb = preprocess_for_inception(batch)
        feats = inception(batch_rgb, training=False)
        features.append(np.array(feats))
        if (i // batch_size + 1) % 10 == 0:
            print(f"    Features: {min(i + batch_size, n)}/{n}", flush=True)
    return np.concatenate(features, axis=0)


def compute_fid(real_features, gen_features):
    """Compute Fréchet Inception Distance between two feature sets."""
    mu1 = np.mean(real_features, axis=0)
    mu2 = np.mean(gen_features, axis=0)
    sigma1 = np.cov(real_features, rowvar=False)
    sigma2 = np.cov(gen_features, rowvar=False)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

    # Numerical stability — discard small imaginary parts
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))
    return fid


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def get_batch(data, indices):
    """Read a batch from mmap, convert to float32 [-1, 1] with channel dim."""
    batch = data[indices].astype(np.float32) / 127.5 - 1.0
    return batch[..., np.newaxis]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    init_device(verbose=False)
    print(f"JAX devices: {jax.devices()}")

    # ---- Model ----
    print(f"Loading model: {args.model}")
    model = DiffusionModel.from_preset(args.model)
    print(f"  Input shape: {model.input_shape}")

    # ---- Data ----
    val_npy = os.path.join(args.data_dir, "val_frames.npy")
    print(f"Loading val data: {val_npy}")
    val_data = np.load(val_npy, mmap_mode='r')
    n_val = val_data.shape[0]
    print(f"  Val: {val_data.shape}, dtype={val_data.dtype}")

    # ---- Generate samples ----
    n_samples = args.n_samples
    print(f"\nGenerating {n_samples} samples ({args.n_steps} steps) ...")
    t0 = time.time()

    # Generate in batches to avoid OOM
    gen_batch_size = 16
    all_samples = []
    n_generated = 0
    while n_generated < n_samples:
        this_batch = min(gen_batch_size, n_samples - n_generated)
        samples = model.sample(n_samples=this_batch, n_steps=args.n_steps, verbose=False)
        all_samples.append(np.array(samples))
        n_generated += this_batch
        print(f"  Generated {n_generated}/{n_samples}", flush=True)

    gen_images = np.concatenate(all_samples, axis=0)  # (n_samples, 112, 112, 1)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")
    print(f"  Range: [{gen_images.min():.3f}, {gen_images.max():.3f}]")

    # Save sample grid (first 16)
    grid_images = [gen_images[i, :, :, 0] for i in range(min(16, n_samples))]
    fig, _ = plot_image_grid(grid_images, ncols=4, cmap="gray", vmin=-1, vmax=1)
    model_name = os.path.basename(args.model.rstrip("/"))
    fig.suptitle(f"Samples — {model_name}")
    sample_path = os.path.join(args.output_dir, f"samples_{model_name}.png")
    fig.savefig(sample_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved grid to {sample_path}")

    # ---- Prepare real images for FID ----
    n_real = min(args.n_real, n_val)
    print(f"\nPreparing {n_real} real images for FID ...")
    rng = np.random.default_rng(42)
    real_idx = rng.choice(n_val, size=n_real, replace=False)
    real_idx.sort()  # sequential access is faster on mmap

    # Load real images in chunks
    chunk_size = 1000
    real_images = []
    for i in range(0, n_real, chunk_size):
        idx = real_idx[i:i + chunk_size]
        real_images.append(get_batch(val_data, idx))
        print(f"  Loaded {min(i + chunk_size, n_real)}/{n_real} real images", flush=True)
    real_images = np.concatenate(real_images, axis=0)  # (n_real, 112, 112, 1)

    # ---- Extract features ----
    inception = build_feature_extractor()

    print(f"\nExtracting features from {n_real} real images ...")
    t0 = time.time()
    real_features = extract_features(inception, jnp.array(real_images), batch_size=32)
    print(f"  Done in {time.time() - t0:.1f}s, shape: {real_features.shape}")

    print(f"Extracting features from {n_samples} generated images ...")
    t0 = time.time()
    gen_features = extract_features(inception, jnp.array(gen_images), batch_size=32)
    print(f"  Done in {time.time() - t0:.1f}s, shape: {gen_features.shape}")

    # ---- Compute FID ----
    fid = compute_fid(real_features, gen_features)

    print(f"\n{'='*60}")
    print(f"Model: {args.model}")
    print(f"FID: {fid:.2f}")
    print(f"  (real: {n_real}, generated: {n_samples}, steps: {args.n_steps})")
    print(f"{'='*60}")

    # ---- Save results ----
    results = {
        "model": args.model,
        "fid": fid,
        "n_real": n_real,
        "n_generated": n_samples,
        "n_steps": args.n_steps,
        "gen_range": [float(gen_images.min()), float(gen_images.max())],
    }
    results_path = os.path.join(args.output_dir, f"results_{model_name}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {results_path}")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
