#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JD_DIR="${ROOT_DIR}/joint_diffusion"

EXPERIMENT="${1:-paper/celeba_mnist_pigdm}"
TASK="${2:-denoise}"
MODELS="${3:-sgm}"

cd "${JD_DIR}"
python inference.py -e "${EXPERIMENT}" -t "${TASK}" -m ${MODELS}
