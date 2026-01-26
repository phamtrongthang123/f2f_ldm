#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

python "$SCRIPT_DIR/zea_synthesize_dataset_3d.py" \
  --output-root "$OUTPUT_DIR"
