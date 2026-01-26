#!/usr/bin/env bash
set -euo pipefail

ENV_DIR=${1:-.venv_joint}
REQ_FILE="dehazing-diffusion/joint_diffusion/requirements/requirements.txt"

python -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"

python -m pip install --upgrade pip
if [[ -f "$REQ_FILE" ]]; then
  python -m pip install -r <(grep -v "^tensorflow-gpu" "$REQ_FILE")
fi
python -m pip install "tensorflow<2.11"

echo "Joint diffusion env ready in: $ENV_DIR"
