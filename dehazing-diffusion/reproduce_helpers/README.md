# Reproduce Helpers: UV Environment Setup

This folder contains helper scripts for ZEA synthesis and inference-config updates.
Use `uv` to create a virtual environment that you can `source` and reuse.

## Prerequisites
- Python 3.10+
- `uv` installed (https://github.com/astral-sh/uv)

## One env for all helper scripts (recommended)

Create a venv and install dependencies:
```
uv venv .venv_helpers
source .venv_helpers/bin/activate
uv pip install zea tensorflow pyyaml
```

Run the scripts:
```
python "reproduce helpers/zea_synthesize_dataset.py" --output-root data/zea_synth
python "reproduce helpers/update_zea_inference_config.py" --wandb-dir dehazing-diffusion/joint_diffusion/wandb
```

## Separate envs (optional)

If you want to keep ZEA/tensorflow isolated from other tools:

ZEA synthesis only:
```
uv venv .venv_zea
source .venv_zea/bin/activate
uv pip install zea tensorflow
python "reproduce helpers/zea_synthesize_dataset.py"
```

Config update only:
```
uv venv .venv_cfg
source .venv_cfg/bin/activate
uv pip install pyyaml
python "reproduce helpers/update_zea_inference_config.py"
```

## Notes
- `source` the env before running any helper scripts.
- The scripts use paths relative to the repo root, so run them from the repo root.
