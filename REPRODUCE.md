# Reproduce: Dehazing Diffusion with ZEA-Synthesized Data

## Goal
Reproduce the dehazing diffusion paper workflow using synthetic RF data generated with ZEA (tissue + haze), then train two score models and run joint posterior sampling for dehazing.

## Assumptions (picked to match the paper as closely as possible)
- Use beamformed RF data at 128x64 (axial x lateral), matching the patch size in the paper.
- Apply mu-law companding (mu=255) after normalizing RF to [-1, 1], then map to [0, 1] for training.
- Generate two independent datasets:
  - Tissue RF (clean signal)
  - Haze RF (structured clutter)
- Measurement model is additive haze: y = x + gamma * h.

## Helper Scripts (new)
All scripts live in `reproduce helpers/`:
- `reproduce helpers/setup_zea_env.sh`: Create a venv and install ZEA for data synthesis.
- `reproduce helpers/run_zea_synthesis.sh`: Generate synthetic tissue/haze datasets into `data/zea_synth`.
- `reproduce helpers/setup_joint_diffusion_env.sh`: Create a venv for joint diffusion training/inference.
- `reproduce helpers/train_zea_models.sh`: Train tissue and haze score models (offline wandb).
- `reproduce helpers/update_zea_inference_config.py`: Auto-fill inference config with latest wandb run paths.
- `reproduce helpers/run_zea_inference.sh`: Run joint diffusion inference with the ZEA config.

## Environment Expectations
- Python 3.10+ is assumed (matches the original codebase).
- GPU is strongly recommended for training/inference; CPU will be very slow.
- Two separate virtual environments are used: one for ZEA synthesis and one for joint diffusion.
- `wandb` runs in offline mode by default during training (set in `train_zea_models.sh`).

## Data Format (synthetic output)
- Output path: `data/zea_synth/` with two subfolders: `tissue/` and `haze/`.
- Each folder contains `train.npz` and `val.npz`.
- Array key is `rf` (configurable via `--npz-key`).
- Array shape is `(N, 128, 64)` by default (axial x lateral).
- Values are companded and scaled to `[0, 255]` before saving.

## Configs (new)
- `dehazing-diffusion/joint_diffusion/configs/training/score_zea_tissue.yaml`
- `dehazing-diffusion/joint_diffusion/configs/training/score_zea_haze.yaml`
- `dehazing-diffusion/joint_diffusion/configs/inference/paper/zea_dehaze_pigdm.yaml`

## Progress
- [x] Added ZEA dataset support in `dehazing-diffusion/joint_diffusion/datasets.py`.
- [x] Added haze corruptor + guidance path for additive haze in `dehazing-diffusion/joint_diffusion/utils/corruptors.py` and `dehazing-diffusion/joint_diffusion/generators/SGM/guidance.py`.
- [x] Implemented ZEA synthesis script (`reproduce helpers/zea_synthesize_dataset.py`).
- [x] Added training/inference configs and helper scripts.

## How to Run (expected flow)
1) Create a ZEA environment and generate data:
   - `./reproduce\ helpers/setup_zea_env.sh`
   - `source .venv_zea/bin/activate`
   - `./reproduce\ helpers/run_zea_synthesis.sh`

2) Create the joint-diffusion environment and train:
   - `./reproduce\ helpers/setup_joint_diffusion_env.sh`
   - `source .venv_joint/bin/activate`
   - `./reproduce\ helpers/train_zea_models.sh`

3) Update inference config and run dehazing:
   - `python ./reproduce\ helpers/update_zea_inference_config.py`
   - `./reproduce\ helpers/run_zea_inference.sh`

## Notes
- `update_zea_inference_config.py` scans `dehazing-diffusion/joint_diffusion/wandb` for the latest runs with datasets `zea_tissue` and `zea_haze` and updates `zea_dehaze_pigdm.yaml` accordingly.
- This update step is required because `zea_dehaze_pigdm.yaml` ships with placeholder paths (`REPLACE_*`). If you skip it, inference will fail.
- Haze strength is set via `haze_strength` in `zea_dehaze_pigdm.yaml`. Increase it for more severe haze.
- If you want larger images, regenerate data with bigger `--grid-size-z`/`--grid-size-x` and adjust `image_size` in the training configs.
- Plots are saved to `dehazing-diffusion/joint_diffusion/figures/` by default; metric summaries (if you run `run_metrics`) go to `dehazing-diffusion/joint_diffusion/results/`.
- The folder name `reproduce helpers` contains a space; keep the escape (`\\`) or quote the path.
