#!/usr/bin/env bash
set -euo pipefail

ENV_DIR=${1:-.venv_zea}
python -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install zea tensorflow

echo "ZEA env ready in: $ENV_DIR"
