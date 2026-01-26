#!/usr/bin/env python3
"""Update inference config with latest ZEA tissue/haze wandb runs."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def _load_wandb_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    cfg = {}
    for key, value in raw.items():
        if isinstance(value, dict) and "value" in value:
            cfg[key] = value.get("value")
        else:
            cfg[key] = value
    return cfg


def _find_latest_run(wandb_dir: Path, dataset_name: str) -> Path | None:
    candidates = []
    for config_path in wandb_dir.glob("**/files/*.yaml"):
        cfg = _load_wandb_config(config_path)
        if cfg.get("dataset_name") != dataset_name:
            continue
        candidates.append(config_path)
    if not candidates:
        return None
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wandb-dir",
        default="dehazing-diffusion/joint_diffusion/wandb",
        help="Wandb root directory to scan",
    )
    parser.add_argument(
        "--inference-config",
        default="dehazing-diffusion/joint_diffusion/configs/inference/paper/zea_dehaze_pigdm.yaml",
    )
    args = parser.parse_args()

    wandb_dir = Path(args.wandb_dir)
    if not wandb_dir.is_dir():
        raise FileNotFoundError(f"wandb dir not found: {wandb_dir}")

    tissue_run = _find_latest_run(wandb_dir, "zea_tissue")
    haze_run = _find_latest_run(wandb_dir, "zea_haze")

    if tissue_run is None or haze_run is None:
        missing = []
        if tissue_run is None:
            missing.append("zea_tissue")
        if haze_run is None:
            missing.append("zea_haze")
        raise FileNotFoundError(f"Missing wandb runs for: {', '.join(missing)}")

    cfg_path = Path(args.inference_config)
    with cfg_path.open("r", encoding="utf-8") as handle:
        inf_cfg = yaml.safe_load(handle) or {}

    inf_cfg.setdefault("run_id", {})
    inf_cfg["run_id"]["sgm"] = str(tissue_run)
    inf_cfg.setdefault("sgm", {})
    inf_cfg["sgm"]["corruptor_run_id"] = str(haze_run)

    with cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(inf_cfg, handle, sort_keys=False)

    print(f"Updated {cfg_path}:")
    print(f"  run_id.sgm = {tissue_run}")
    print(f"  sgm.corruptor_run_id = {haze_run}")


if __name__ == "__main__":
    main()
