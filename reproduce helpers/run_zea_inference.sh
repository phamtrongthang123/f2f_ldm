#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JD_DIR="$ROOT_DIR/dehazing-diffusion/joint_diffusion"

pushd "$JD_DIR" >/dev/null
python inference.py -e paper/zea_dehaze_pigdm -t denoise -m sgm
popd >/dev/null
