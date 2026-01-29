#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JD_DIR="$ROOT_DIR/dehazing-diffusion/joint_diffusion"

export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=${WANDB_DIR:-$JD_DIR/wandb}

pushd "$JD_DIR" >/dev/null
python train.py -c configs/training/score_zea_tissue.yaml
python train.py -c configs/training/score_zea_haze.yaml
popd >/dev/null
