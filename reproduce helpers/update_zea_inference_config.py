#!/usr/bin/env python3
"""Update inference config with latest ZEA tissue/haze wandb runs."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def _load_wandb_config(path: Path) -> dict:
    """Load a wandb config YAML and unwrap `value` fields.

    Role:
        Reads a wandb-generated YAML config and returns a flat dict where entries
        like `{"value": 123}` become `123`.

    Example:
        >>> cfg = _load_wandb_config(Path("wandb/run-123/files/config.yaml"))
        >>> isinstance(cfg, dict)
        True

    Output / Expectation:
        Returns a plain dict with best-effort unwrapping of wandb `value` nodes.
    """
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
    """Find the most recently modified wandb run for a given dataset.

    Role:
        Scans `wandb_dir/**/files/*.yaml` for configs whose `dataset_name`
        matches the requested dataset, then returns the newest run directory.

    Example:
        >>> run_dir = _find_latest_run(Path("wandb"), "zea_tissue")
        >>> run_dir is None or run_dir.is_dir()
        True

    Output / Expectation:
        Returns a Path to the run directory or None if no matches exist.
    """
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
    """CLI entry point to update inference config from latest wandb runs.

    Role:
        Locates the newest `zea_tissue` and `zea_haze` runs and writes their
        paths into the inference config YAML.

    Example:
        $ python update_zea_inference_config.py --wandb-dir dehazing-diffusion/joint_diffusion/wandb

    Output / Expectation:
        Updates `run_id.sgm` and `sgm.corruptor_run_id` in the target config file
        and prints the new paths.
    """
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
