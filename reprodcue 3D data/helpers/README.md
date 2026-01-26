# 3D Helpers: UV Environment Setup

This folder contains helper scripts for generating 3D ZEA volumes.
Use `uv` to create a virtual environment that you can `source`.

## Prerequisites
- Python 3.10+
- `uv` installed (https://github.com/astral-sh/uv)

## One env for 3D synthesis

Create a venv and install dependencies:
```
uv venv .venv_zea_3d
source .venv_zea_3d/bin/activate
uv pip install zea tensorflow
```

Run the script:
```
python "reprodcue 3D data/helpers/zea_synthesize_dataset_3d.py" --output-root "reprodcue 3D data"
```

## Notes
- `source` the env before running the script.
- Run from the repo root so relative paths resolve correctly.
