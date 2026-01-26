# Reprodcue 3D Data

This folder stores ZEA-synthesized 3D ultrasound RF volumes (tissue + haze).

## Generate
- Script: `reprodcue 3D data/helpers/zea_synthesize_dataset_3d.py`
- Runner: `./reprodcue\ 3D\ data/helpers/run_zea_synthesis_3d.sh`
- Dependencies: `zea` + `tensorflow` (see `reproduce helpers/setup_zea_env.sh`)

## Output layout
- `tissue/train.npz`
- `tissue/val.npz`
- `haze/train.npz`
- `haze/val.npz`
- `metadata.json`

## Data shape
Arrays are stored under the `rf` key and have shape `(N, S, Z, X)` where:
- `N` = number of volumes
- `S` = number of elevational slices (`--n-slices`)
- `Z` = axial samples (`--grid-size-z`)
- `X` = lateral samples (`--grid-size-x`)

Note: volumes are built by stacking multiple 2D slices with a linear-array beamformer, so this is an approximate 3D volume.
