"""Visualize all images in zea_synth dataset.

Saves grid images to an `output_vis/` folder:
  - tissue_train.png, tissue_val.png
  - haze_train.png,   haze_val.png

Each image in the grid is one (sample, tx_angle) slice.
RF data is converted to B-mode (envelope detection + log compression) for display.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert


def rf_to_bmode(rf: np.ndarray, dynamic_range_db: float = 50.0) -> np.ndarray:
    """Convert RF data to B-mode image via envelope detection and log compression.

    Args:
        rf: RF data array, can be 2D (H, W) or 3D (N, H, W)
        dynamic_range_db: Dynamic range in dB for display (default 50 dB)

    Returns:
        B-mode image(s) normalized to [0, 1]
    """
    # Handle both 2D and 3D inputs
    if rf.ndim == 2:
        rf = rf[np.newaxis, ...]
        squeeze = True
    else:
        squeeze = False

    bmode_list = []
    for frame in rf:
        # Envelope detection via Hilbert transform (along axial/depth dimension)
        analytic = hilbert(frame, axis=0)
        envelope = np.abs(analytic)

        # Avoid log(0)
        envelope = np.maximum(envelope, 1e-10)

        # Log compression (dB scale)
        log_env = 20.0 * np.log10(envelope)

        # Normalize to dynamic range
        max_val = log_env.max()
        log_env = log_env - max_val  # Now max is 0 dB
        log_env = np.clip(log_env, -dynamic_range_db, 0)

        # Scale to [0, 1]
        bmode = (log_env + dynamic_range_db) / dynamic_range_db
        bmode_list.append(bmode)

    result = np.stack(bmode_list, axis=0)
    if squeeze:
        result = result[0]
    return result


def make_grid(images: np.ndarray, ncols: int = 10, vmin: float = None, vmax: float = None) -> plt.Figure:
    """Create a grid figure from a stack of 2D images."""
    n = len(images)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2 * ncols, 2.5 * nrows))
    axes = np.atleast_2d(axes)
    for idx in range(nrows * ncols):
        ax = axes[idx // ncols, idx % ncols]
        if idx < n:
            ax.imshow(images[idx], cmap="gray", aspect="auto", vmin=vmin, vmax=vmax)
            ax.set_title(f"{idx}", fontsize=7)
        ax.axis("off")
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize zea_synth dataset")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(Path(__file__).resolve().parent),
        help="Path to zea_synth directory",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: <data-dir>/output_vis)",
    )
    parser.add_argument("--ncols", type=int, default=10, help="Columns in the grid")
    parser.add_argument(
        "--dynamic-range",
        type=float,
        default=50.0,
        help="Dynamic range in dB for B-mode display (default: 50)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir) if args.out_dir else data_dir / "output_vis"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(data_dir / "metadata.json") as f:
        meta = json.load(f)
    key = meta["npz_key"]
    data_type = meta.get("data_type", "bmode")  # Support both old and new format

    for category in ["tissue", "haze"]:
        for split in ["train", "val"]:
            path = data_dir / category / f"{split}.npz"
            if not path.exists():
                print(f"Skipping {path} (not found)")
                continue
            data = np.load(path)[key]  # (N, H, W)
            print(f"{category}/{split}: {data.shape[0]} frames, shape {data.shape[1:]}, "
                  f"RF range [{data.min():.2e}, {data.max():.2e}]")

            # Convert RF to B-mode for visualization
            if data_type == "rf":
                imgs = rf_to_bmode(data, dynamic_range_db=args.dynamic_range)
                print(f"  Converted to B-mode (dynamic range: {args.dynamic_range} dB)")
            else:
                # Legacy B-mode data, normalize to [0, 1]
                imgs = data.astype(np.float32) / 255.0

            fig = make_grid(imgs, ncols=args.ncols, vmin=0, vmax=1)
            fig.suptitle(f"{category} / {split}  ({imgs.shape[0]} images)", fontsize=14, y=1.01)
            save_path = out_dir / f"{category}_{split}.png"
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved → {save_path}")


if __name__ == "__main__":
    main()
