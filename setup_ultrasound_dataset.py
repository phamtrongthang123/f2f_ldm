#!/usr/bin/env python3
"""
Script to prepare ultrasound video frames for F2FLDM training.
This will:
1. Crop out UI elements from ultrasound images
2. Resize to standard resolution (512x512)
3. Create dataset structure compatible with train.py
4. Generate metadata.csv files
"""

import os
import csv
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse

def crop_ultrasound_roi(image_path, output_path, crop_box=None):
    """
    Crop ultrasound image to remove UI elements.

    Args:
        image_path: Path to input image
        output_path: Path to save cropped image
        crop_box: (left, top, right, bottom) or None for auto-detect
    """
    img = Image.open(image_path)

    # Default crop to remove left UI and right scale (adjust based on your images)
    # From your images: 1220x836, ultrasound region is roughly x:240-1020, y:130-700
    if crop_box is None:
        width, height = img.size
        # Crop to central ultrasound region (adjust these values if needed)
        left = int(width * 0.2)   # Skip left UI (~240px)
        top = int(height * 0.15)   # Skip top text (~130px)
        right = int(width * 0.84)  # Skip right scale (~1020px)
        bottom = int(height * 0.84) # Skip bottom (~700px)
        crop_box = (left, top, right, bottom)

    img_cropped = img.crop(crop_box)

    # Resize to standard resolution (512x512 for efficiency, or 1024x1024 for quality)
    img_resized = img_cropped.resize((512, 512), Image.Resampling.LANCZOS)

    # Save as PNG
    img_resized.save(output_path, 'PNG')
    return img_resized

def prepare_dataset(
    source_m1_dir,
    source_m3_dir,
    output_base_dir,
    target_resolution=512,
    crop_box=None,
    max_frames=None
):
    """
    Prepare ultrasound dataset for training.

    Creates structure:
    output_base_dir/
        train/
            m1/  (domain A - source)
            m3/  (domain B - target)
            metadata_m1.csv
            metadata_m3.csv
            metadata_combined.csv
    """
    # Create output directories
    output_base = Path(output_base_dir)
    train_m1_dir = output_base / "train" / "m1"
    train_m3_dir = output_base / "train" / "m3"

    train_m1_dir.mkdir(parents=True, exist_ok=True)
    train_m3_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing m_1 frames (source domain)...")
    m1_frames = sorted(Path(source_m1_dir).glob("*.png"))
    if max_frames:
        m1_frames = m1_frames[:max_frames]

    m1_metadata = []
    for frame_path in tqdm(m1_frames):
        output_name = f"m1_{frame_path.stem}.png"
        output_path = train_m1_dir / output_name
        crop_ultrasound_roi(frame_path, output_path, crop_box)
        m1_metadata.append({"image": output_name, "text": "ultrasound image"})

    print(f"\nProcessing m_3 frames (target domain)...")
    m3_frames = sorted(Path(source_m3_dir).glob("*.png"))
    if max_frames:
        m3_frames = m3_frames[:max_frames]

    m3_metadata = []
    for frame_path in tqdm(m3_frames):
        output_name = f"m3_{frame_path.stem}.png"
        output_path = train_m3_dir / output_name
        crop_ultrasound_roi(frame_path, output_path, crop_box)
        m3_metadata.append({"image": output_name, "text": "ultrasound image"})

    # Save metadata CSV files
    print("\nSaving metadata files...")

    # Metadata for m1 only
    with open(output_base / "train" / "metadata_m1.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["image", "text"])
        writer.writeheader()
        writer.writerows(m1_metadata)

    # Metadata for m3 only
    with open(output_base / "train" / "metadata_m3.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["image", "text"])
        writer.writeheader()
        writer.writerows(m3_metadata)

    # Combined metadata (for joint training)
    combined_metadata = m1_metadata + m3_metadata
    with open(output_base / "train" / "metadata_combined.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["image", "text"])
        writer.writeheader()
        writer.writerows(combined_metadata)

    print(f"\n✓ Dataset preparation complete!")
    print(f"  m1 frames: {len(m1_metadata)}")
    print(f"  m3 frames: {len(m3_metadata)}")
    print(f"  Total: {len(combined_metadata)}")
    print(f"\nOutput structure:")
    print(f"  {train_m1_dir}")
    print(f"  {train_m3_dir}")
    print(f"  {output_base / 'train' / 'metadata_m1.csv'}")
    print(f"  {output_base / 'train' / 'metadata_m3.csv'}")
    print(f"  {output_base / 'train' / 'metadata_combined.csv'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare ultrasound dataset")
    parser.add_argument("--source_m1", type=str, required=True, help="Path to m_1 frames")
    parser.add_argument("--source_m3", type=str, required=True, help="Path to m_3 frames")
    parser.add_argument("--output", type=str, default="data/ultrasound_dataset", help="Output directory")
    parser.add_argument("--resolution", type=int, default=512, help="Target resolution")
    parser.add_argument("--max_frames", type=int, default=None, help="Max frames per video")
    parser.add_argument("--crop", type=str, default=None,
                        help="Crop box as 'left,top,right,bottom'")

    args = parser.parse_args()

    crop_box = None
    if args.crop:
        crop_box = tuple(map(int, args.crop.split(',')))

    prepare_dataset(
        source_m1_dir=args.source_m1,
        source_m3_dir=args.source_m3,
        output_base_dir=args.output,
        target_resolution=args.resolution,
        crop_box=crop_box,
        max_frames=args.max_frames
    )
