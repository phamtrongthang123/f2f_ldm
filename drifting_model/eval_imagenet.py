"""Evaluation script: generate 50K images and compute FID/IS.

Single forward pass per image (1-NFE). Sweep CFG alpha in [1.0, 3.5].

Usage:
    python eval_imagenet.py --checkpoint checkpoints/checkpoint_final.pt \
        --output_dir eval_output --cfg_alpha 2.0 --batch_size 64
"""

import os
import argparse
import math
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from PIL import Image
from diffusers import AutoencoderKL

from models.dit import DiTGenerator, dit_b2, dit_l2


def load_generator(checkpoint_path, device):
    """Load generator from checkpoint, using EMA weights."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    gen_cfg = cfg["generator"]

    if gen_cfg["arch"] == "DiT-B/2":
        generator = dit_b2(
            input_size=gen_cfg["input_size"],
            input_channels=gen_cfg["input_channels"],
            num_classes=gen_cfg["num_classes"],
            num_register_tokens=gen_cfg["num_register_tokens"],
            num_style_tokens=gen_cfg["num_style_tokens"],
            style_codebook_size=gen_cfg["style_codebook_size"],
        )
    elif gen_cfg["arch"] == "DiT-L/2":
        generator = dit_l2(
            input_size=gen_cfg["input_size"],
            input_channels=gen_cfg["input_channels"],
            num_classes=gen_cfg["num_classes"],
            num_register_tokens=gen_cfg["num_register_tokens"],
            num_style_tokens=gen_cfg["num_style_tokens"],
            style_codebook_size=gen_cfg["style_codebook_size"],
        )
    else:
        raise ValueError(f"Unknown architecture: {gen_cfg['arch']}")

    generator.load_state_dict(ckpt["ema_generator"])
    generator = generator.to(device)
    generator.eval()
    return generator, cfg


@torch.no_grad()
def generate_images(generator, vae, cfg, cfg_alpha, num_images=50000,
                    batch_size=64, device="cuda", output_dir=None):
    """Generate images using the trained generator.

    For each of 1000 classes, generate num_images/1000 = 50 images.
    Single forward pass per image (1-NFE).

    Args:
        generator: trained DiTGenerator (EMA weights)
        vae: SD-VAE decoder
        cfg: config dict
        cfg_alpha: CFG strength for inference
        num_images: total images to generate (default 50K)
        batch_size: generation batch size
        device: torch device
        output_dir: if provided, save images to this directory

    Returns:
        all_images: list of numpy arrays [H, W, 3] uint8
    """
    gen_cfg = cfg["generator"]
    num_classes = gen_cfg["num_classes"]
    images_per_class = num_images // num_classes
    vae_scaling_factor = 0.18215

    all_images = []
    save_dir = None
    if output_dir is not None:
        save_dir = os.path.join(output_dir, f"alpha_{cfg_alpha:.1f}")
        os.makedirs(save_dir, exist_ok=True)

    img_idx = 0
    for cls in tqdm(range(num_classes), desc=f"Generating (alpha={cfg_alpha})"):
        remaining = images_per_class
        while remaining > 0:
            bs = min(batch_size, remaining)

            # Sample noise
            noise = torch.randn(
                bs, gen_cfg["input_channels"],
                gen_cfg["input_size"], gen_cfg["input_size"],
                device=device,
            )

            # Class labels
            labels = torch.full((bs,), cls, device=device, dtype=torch.long)

            # CFG alpha
            alpha = torch.full((bs,), cfg_alpha, device=device)

            # Style indices
            style_indices = torch.randint(
                0, gen_cfg["style_codebook_size"],
                (bs, gen_cfg.get("num_style_tokens", 32)),
                device=device,
            )

            # Generate latent (1-NFE)
            latent = generator(noise, labels, alpha, style_indices)

            # Decode to pixels
            pixels = vae.decode(latent / vae_scaling_factor).sample
            pixels = pixels.clamp(-1, 1)

            # Convert to uint8 images
            pixels = ((pixels + 1) / 2 * 255).byte()
            pixels = pixels.permute(0, 2, 3, 1).cpu().numpy()  # [B, H, W, 3]

            for j in range(bs):
                img = pixels[j]
                all_images.append(img)
                if save_dir is not None:
                    Image.fromarray(img).save(
                        os.path.join(save_dir, f"{img_idx:06d}.png")
                    )
                img_idx += 1

            remaining -= bs

    return all_images


def compute_fid(generated_dir, dataset_name="imagenet", dataset_split="custom",
                dataset_dir=None):
    """Compute FID using clean-fid.

    Args:
        generated_dir: directory with generated PNG images
        dataset_name: reference dataset name for clean-fid
        dataset_split: split name
        dataset_dir: path to reference images (if custom)

    Returns:
        fid_score: float
    """
    from cleanfid import fid

    if dataset_dir is not None:
        score = fid.compute_fid(generated_dir, dataset_dir)
    else:
        # Use pre-computed stats for ImageNet
        score = fid.compute_fid(
            generated_dir,
            dataset_name="imagenet",
            dataset_res=256,
            dataset_split="val",
        )
    return score


def compute_is_score(generated_dir):
    """Compute Inception Score using torch-fidelity.

    Args:
        generated_dir: directory with generated PNG images

    Returns:
        (is_mean, is_std)
    """
    import torch_fidelity

    metrics = torch_fidelity.calculate_metrics(
        input1=generated_dir,
        isc=True,
        isc_splits=10,
    )
    return metrics["inception_score_mean"], metrics["inception_score_std"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./eval_output")
    parser.add_argument("--cfg_alpha", type=float, default=None,
                        help="CFG alpha. If not set, sweep [1.0, 3.5]")
    parser.add_argument("--num_images", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--imagenet_val_dir", type=str, default=None,
                        help="Path to ImageNet validation images for FID reference")
    parser.add_argument("--skip_metrics", action="store_true",
                        help="Only generate images, skip FID/IS computation")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load models
    generator, cfg = load_generator(args.checkpoint, args.device)

    vae = AutoencoderKL.from_pretrained(
        cfg["vae"]["model_id"]
    ).to(args.device)
    vae.eval()

    # Determine alpha values
    if args.cfg_alpha is not None:
        alphas = [args.cfg_alpha]
    else:
        alphas = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]

    results = {}
    for alpha in alphas:
        print(f"\n--- CFG alpha = {alpha} ---")

        images = generate_images(
            generator, vae, cfg, alpha,
            num_images=args.num_images,
            batch_size=args.batch_size,
            device=args.device,
            output_dir=args.output_dir,
        )

        if not args.skip_metrics:
            gen_dir = os.path.join(args.output_dir, f"alpha_{alpha:.1f}")

            # FID
            try:
                fid_score = compute_fid(gen_dir, dataset_dir=args.imagenet_val_dir)
                print(f"FID: {fid_score:.2f}")
            except Exception as e:
                print(f"FID computation failed: {e}")
                fid_score = None

            # IS
            try:
                is_mean, is_std = compute_is_score(gen_dir)
                print(f"IS: {is_mean:.2f} ± {is_std:.2f}")
            except Exception as e:
                print(f"IS computation failed: {e}")
                is_mean, is_std = None, None

            results[alpha] = {
                "fid": fid_score,
                "is_mean": is_mean,
                "is_std": is_std,
            }

    # Print summary
    if results:
        print("\n=== Summary ===")
        print(f"{'Alpha':>8} {'FID':>10} {'IS':>15}")
        for alpha, r in sorted(results.items()):
            fid_str = f"{r['fid']:.2f}" if r["fid"] is not None else "N/A"
            is_str = (
                f"{r['is_mean']:.2f}±{r['is_std']:.2f}"
                if r["is_mean"] is not None else "N/A"
            )
            print(f"{alpha:>8.1f} {fid_str:>10} {is_str:>15}")


if __name__ == "__main__":
    main()
