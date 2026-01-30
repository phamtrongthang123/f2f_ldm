"""Test script for verifying retrained checkpoints.

Loads each model (tissue and haze) and runs unconditional sampling
to verify no NaN/Inf and that samples look meaningful.

Usage:
    python test_checkpoints.py \
        --tissue_run <path_to_tissue_wandb_run_files> \
        --haze_run <path_to_haze_wandb_run_files> \
        [--device 0] [--num_samples 4] [--num_scales 200]
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from easydict import EasyDict as edict

from generators.models import get_model
from generators.SGM.sampling import ScoreSampler
from utils.checkpoints import ModelCheckpoint
from utils.runs import init_config


def parse_args():
    parser = argparse.ArgumentParser(description="Test retrained checkpoints")
    parser.add_argument("--tissue_run", type=str, required=True,
                        help="Path to tissue model wandb run files dir")
    parser.add_argument("--haze_run", type=str, required=True,
                        help="Path to haze model wandb run files dir")
    parser.add_argument("--device", type=int, default=0,
                        help="GPU device index (default: 0)")
    parser.add_argument("--num_samples", type=int, default=4,
                        help="Number of samples to generate (default: 4)")
    parser.add_argument("--num_scales", type=int, default=200,
                        help="Number of diffusion steps (default: 200)")
    parser.add_argument("--output", type=str, default="test_checkpoints.png",
                        help="Output figure path (default: test_checkpoints.png)")
    return parser.parse_args()


def load_model(run_path, device, num_scales=None):
    """Load a model from a wandb run path."""
    config = init_config(run_path, verbose=False)
    if num_scales is not None:
        config.num_scales = num_scales

    # Ensure image_shape is in (C, H, W) format
    if not hasattr(config, "image_shape"):
        h, w = config.image_size
        c = 1 if config.color_mode == "grayscale" else 3
        config.image_shape = (c, h, w)

    model = get_model(config, training=False)
    ckpt = ModelCheckpoint(model, config=config)
    ckpt.restore()

    model = model.to(device)
    model.eval()
    return model, config


def test_score_output(model, config, device, num_samples=4):
    """Test that model score outputs contain no NaN/Inf."""
    shape = (num_samples, *config.image_shape)
    x = torch.randn(*shape, device=device)
    t = torch.ones(num_samples, device=device) * 0.5

    with torch.no_grad():
        score = model.get_score(x, t, training=False)

    has_nan = torch.isnan(score).any().item()
    has_inf = torch.isinf(score).any().item()
    score_norm = torch.norm(score.reshape(num_samples, -1), dim=-1).mean().item()

    return has_nan, has_inf, score_norm


def unconditional_sample(model, config, device, num_samples=4, num_scales=200):
    """Run unconditional sampling and return samples."""
    shape = (num_samples, *config.image_shape)

    sampler = ScoreSampler(
        model=model,
        image_shape=config.image_shape,
        sde=model.sde,
        sampling_method="pc",
        predictor="euler_maruyama",
        corrector="none",
        guidance=None,
        n_corrector_steps=1,
        corrector_snr=0.17,
    )

    with torch.no_grad():
        samples = sampler(z=None, shape=shape, progress_bar=True)

    return samples


def main():
    args = parse_args()
    os.environ["WANDB_MODE"] = "disabled"

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    results = {}
    for name, run_path in [("tissue", args.tissue_run), ("haze", args.haze_run)]:
        print(f"{'='*60}")
        print(f"Testing {name} model: {run_path}")
        print(f"{'='*60}")

        # Load model
        model, config = load_model(run_path, device, num_scales=args.num_scales)

        # Test score outputs
        has_nan, has_inf, score_norm = test_score_output(
            model, config, device, args.num_samples
        )
        print(f"  Score NaN: {has_nan}, Inf: {has_inf}, norm: {score_norm:.4f}")
        if has_nan or has_inf:
            print(f"  WARNING: {name} model produces NaN/Inf scores!")

        # Unconditional sampling
        print(f"  Running unconditional sampling ({args.num_scales} steps)...")
        samples = unconditional_sample(
            model, config, device, args.num_samples, args.num_scales
        )

        sample_nan = torch.isnan(samples).any().item()
        sample_inf = torch.isinf(samples).any().item()
        print(f"  Sample NaN: {sample_nan}, Inf: {sample_inf}")
        print(f"  Sample range: [{samples.min().item():.4f}, {samples.max().item():.4f}]")
        print(f"  Sample mean: {samples.mean().item():.4f}, std: {samples.std().item():.4f}")

        results[name] = {
            "samples": samples.cpu().numpy(),
            "config": config,
            "score_nan": has_nan,
            "score_inf": has_inf,
            "sample_nan": sample_nan,
            "sample_inf": sample_inf,
        }
        print()

    # Plot results
    n = args.num_samples
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    if n == 1:
        axes = axes.reshape(2, 1)

    for row, name in enumerate(["tissue", "haze"]):
        samples = results[name]["samples"]
        for col in range(n):
            img = samples[col, 0]  # (H, W), grayscale
            axes[row, col].imshow(img, cmap="gray", aspect="auto")
            axes[row, col].set_title(f"{name} #{col}")
            axes[row, col].axis("off")

    fig.suptitle("Unconditional Samples from Retrained Models", fontsize=14)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Figure saved to {args.output}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    all_ok = True
    for name in ["tissue", "haze"]:
        r = results[name]
        ok = not (r["score_nan"] or r["score_inf"] or r["sample_nan"] or r["sample_inf"])
        status = "PASS" if ok else "FAIL"
        print(f"  {name}: {status}")
        if not ok:
            all_ok = False

    if all_ok:
        print("\nAll checks passed.")
    else:
        print("\nSome checks failed. See details above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
