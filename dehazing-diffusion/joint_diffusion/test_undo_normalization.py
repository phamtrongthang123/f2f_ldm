"""Sanity test: raw RF → ZeaDataset normalize → undo_normalization → B-mode
should produce the same image as raw RF → B-mode directly.

Usage:
    python test_undo_normalization.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

NPZ_PATH = "/home/tp030/f2f_ldm/data/zea_synth_test/tissue/val.npz"
NPZ_KEY = "rf"
SAMPLE_INDEX = 0
IMAGE_RANGE = (-1, 1)
MU = 255


def main():
    from utils.bmode import rf_to_bmode, undo_normalization

    # Load raw RF data
    raw = np.load(NPZ_PATH)[NPZ_KEY].astype(np.float32)
    sample_raw = raw[SAMPLE_INDEX]  # (n_tx, n_ax, n_el)
    print(f"Raw sample shape: {sample_raw.shape}")
    print(f"Raw range: [{raw.min():.4f}, {raw.max():.4f}]")

    # --- Path A: raw RF → B-mode directly ---
    bmode_direct = rf_to_bmode(sample_raw[np.newaxis])[0]

    # --- Also get B-mode from min-max normalized (no mu-law) for scale-invariance check ---
    data_min, data_max = raw.min(), raw.max()
    minmax_only = 2.0 * (raw - data_min) / (data_max - data_min) - 1.0
    bmode_minmax = rf_to_bmode(minmax_only[SAMPLE_INDEX][np.newaxis])[0]

    # --- Path B: raw RF → ZeaDataset normalize → undo → B-mode ---
    # Reproduce ZeaDataset normalization (on full dataset for min/max)
    normalized = minmax_only.copy()
    # mu-law compress
    normalized = np.sign(normalized) * np.log1p(MU * np.abs(normalized)) / np.log1p(MU)
    # rescale to image_range
    lo, hi = IMAGE_RANGE
    normalized = (normalized + 1.0) / 2.0 * (hi - lo) + lo

    print(f"Normalized range: [{normalized.min():.4f}, {normalized.max():.4f}]")

    # Now undo normalization (with full min-max restore)
    recovered = undo_normalization(
        normalized, image_range=(lo, hi), mu=MU,
        data_min=data_min, data_max=data_max,
    )
    sample_recovered = recovered[SAMPLE_INDEX]

    print(f"Recovered range: [{recovered.min():.6f}, {recovered.max():.6f}]")
    print(f"Raw range was:   [{data_min:.6f}, {data_max:.6f}]")

    # Check: recovered should equal the original raw data
    max_err = np.max(np.abs(recovered - raw))
    print(f"Max abs error (recovered vs original raw): {max_err:.2e}")

    bmode_roundtrip = rf_to_bmode(sample_recovered[np.newaxis])[0]

    # Also show what happens WITHOUT undo (the original bug)
    bmode_no_undo = rf_to_bmode(normalized[SAMPLE_INDEX][np.newaxis])[0]

    # Convert to numpy arrays (handles ZEA Image objects or plain arrays)
    def to_array(img):
        return np.asarray(img, dtype=np.uint8)

    bmode_direct = to_array(bmode_direct)
    bmode_minmax = to_array(bmode_minmax)
    bmode_roundtrip = to_array(bmode_roundtrip)
    bmode_no_undo = to_array(bmode_no_undo)

    # Compare B-mode images
    diff = np.abs(bmode_direct.astype(float) - bmode_roundtrip.astype(float))
    print("\nB-mode comparison:")
    print(
        f"  Direct   : dtype={bmode_direct.dtype}, shape={bmode_direct.shape}, "
        f"range=[{bmode_direct.min()}, {bmode_direct.max()}]"
    )
    print(
        f"  Roundtrip: dtype={bmode_roundtrip.dtype}, shape={bmode_roundtrip.shape}, "
        f"range=[{bmode_roundtrip.min()}, {bmode_roundtrip.max()}]"
    )
    print(f"  Max pixel diff: {diff.max():.1f}")
    print(f"  Mean pixel diff: {diff.mean():.2f}")

    if diff.max() < 2:
        print("\nPASS: B-mode images match (max diff < 2 uint8 levels)")
    else:
        print(
            f"\nFAIL: B-mode images differ significantly (max diff = {diff.max():.0f})"
        )

    diff_no_undo = np.abs(bmode_direct.astype(float) - bmode_no_undo.astype(float))
    print("\nWithout undo_normalization (original bug):")
    print(f"  Max pixel diff: {diff_no_undo.max():.1f}")
    print(f"  Mean pixel diff: {diff_no_undo.mean():.2f}")

    # Save images for visual comparison
    from PIL import Image

    out_dir = os.path.join(os.path.dirname(__file__), "sanity_check_output")
    os.makedirs(out_dir, exist_ok=True)

    Image.fromarray(bmode_direct).save(os.path.join(out_dir, "1_direct_raw.png"))
    Image.fromarray(bmode_minmax).save(os.path.join(out_dir, "2_minmax_only.png"))
    Image.fromarray(bmode_roundtrip).save(os.path.join(out_dir, "3_roundtrip_undo.png"))
    Image.fromarray(bmode_no_undo).save(os.path.join(out_dir, "4_no_undo_bug.png"))
    print(f"\nImages saved to {out_dir}/")
    print("  1_direct_raw.png     — ground truth (raw RF → B-mode)")
    print("  2_minmax_only.png    — min-max normalized → B-mode (scale-invariance check)")
    print("  3_roundtrip_undo.png — normalize → undo → B-mode (should match 2)")
    print("  4_no_undo_bug.png    — normalize → B-mode WITHOUT undo (the bug)")

    # Numerical comparison: 2 vs 3 should be near-identical (same scale, same data)
    diff_2v3 = np.abs(bmode_minmax.astype(float) - bmode_roundtrip.astype(float))
    print(f"\nImage 2 vs 3 (minmax vs roundtrip, same scale):")
    print(f"  Max pixel diff: {diff_2v3.max():.1f}")
    print(f"  Mean pixel diff: {diff_2v3.mean():.2f}")

    # Scale-invariance check: 1 vs 2
    diff_1v2 = np.abs(bmode_direct.astype(float) - bmode_minmax.astype(float))
    print(f"\nImage 1 vs 2 (raw vs minmax — ZEA scale-invariance check):")
    print(f"  Max pixel diff: {diff_1v2.max():.1f}")
    print(f"  Mean pixel diff: {diff_1v2.mean():.2f}")


if __name__ == "__main__":
    main()
