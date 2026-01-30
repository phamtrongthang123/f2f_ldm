#!/usr/bin/env python3
"""Convert wandb config.yaml format to flat format for inference."""
import argparse
import yaml
from pathlib import Path


def convert_wandb_config(input_path: Path, output_path: Path = None):
    """Convert wandb nested config to flat config.
    
    Wandb format:
        key:
            value: actual_value
    
    Flat format:
        key: actual_value
    """
    with open(input_path) as f:
        wandb_config = yaml.safe_load(f)
    
    # Extract flat config, skipping _wandb metadata
    flat_config = {}
    for key, val in wandb_config.items():
        if key.startswith("_"):
            continue  # Skip wandb internal keys
        if isinstance(val, dict) and "value" in val:
            flat_config[key] = val["value"]
        else:
            flat_config[key] = val
    
    # Write output
    if output_path is None:
        output_path = input_path.parent / "config_flat.yaml"
    
    with open(output_path, "w") as f:
        yaml.dump(flat_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Converted {input_path} -> {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Convert wandb config to flat format")
    parser.add_argument("input", type=Path, help="Input wandb config.yaml path")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output path (default: config_flat.yaml in same dir)")
    args = parser.parse_args()
    
    convert_wandb_config(args.input, args.output)


if __name__ == "__main__":
    main()
