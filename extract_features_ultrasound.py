#!/usr/bin/env python3
"""
Extract DINO/DINOv2 features from ultrasound images for embedding translation.
This creates the feature dataset needed for training the CycleGAN.
"""

import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse

# Use the ultrasound-adapted feature extractor
from feature_extractor_ultrasound import get_feat_model, get_transform

def extract_features(image_dir, output_dir, model_name="dinov2", device="cuda"):
    """
    Extract features from all images in a directory.

    Args:
        image_dir: Directory containing images
        output_dir: Directory to save .npy feature files
        model_name: "dino" or "dinov2"
        device: "cuda" or "cpu"
    """
    # Load model
    print(f"Loading {model_name} model...")
    model, feat_dim = get_feat_model(model_name)
    model = model.to(device)
    model.eval()

    # Get transform
    transform = get_transform(model_name, is_ultrasound=True)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get all images
    image_paths = sorted(Path(image_dir).glob("*.png"))
    print(f"Found {len(image_paths)} images in {image_dir}")

    # Extract features
    print("Extracting features...")
    with torch.no_grad():
        for img_path in tqdm(image_paths):
            # Load and transform image
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)

            # Extract features
            features = model(img_tensor)

            # Save features
            output_path = Path(output_dir) / f"{img_path.stem}.npy"
            np.save(output_path, features.cpu().numpy())

    print(f"✓ Extracted {len(image_paths)} feature files to {output_dir}")
    print(f"  Feature dimension: {feat_dim}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from ultrasound images")
    parser.add_argument("--m1_dir", type=str, required=True, help="Directory with m1 images")
    parser.add_argument("--m3_dir", type=str, required=True, help="Directory with m3 images")
    parser.add_argument("--output_base", type=str, default="data/ultrasound_dataset/features",
                        help="Base output directory")
    parser.add_argument("--model", type=str, default="dinov2", choices=["dino", "dinov2"],
                        help="Feature extraction model")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")

    args = parser.parse_args()

    # Extract features for m1 (trainA)
    print("\n" + "="*60)
    print("Extracting features for m_1 (domain A)...")
    print("="*60)
    extract_features(
        image_dir=args.m1_dir,
        output_dir=f"{args.output_base}/trainA",
        model_name=args.model,
        device=args.device
    )

    # Extract features for m3 (trainB)
    print("\n" + "="*60)
    print("Extracting features for m_3 (domain B)...")
    print("="*60)
    extract_features(
        image_dir=args.m3_dir,
        output_dir=f"{args.output_base}/trainB",
        model_name=args.model,
        device=args.device
    )

    print("\n" + "="*60)
    print("✓ All features extracted!")
    print("="*60)
    print(f"\nFeature structure:")
    print(f"  {args.output_base}/trainA/  (m_1 features)")
    print(f"  {args.output_base}/trainB/  (m_3 features)")
    print(f"\nNext: Train embedding translation with:")
    print(f"  cd embedding_translation")
    print(f"  bash train_et_ultrasound.sh")
