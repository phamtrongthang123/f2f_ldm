# Dehazing Ultrasound using Diffusion Models

PyTorch reproduction of [Dehazing Ultrasound using Diffusion Models](https://arxiv.org/abs/2307.11204) (Stevens et al., 2023).

Uses joint posterior sampling with score-based diffusion models to remove structured haze noise from ultrasound RF data. The core implementation lives in `joint_diffusion/`.

## Project Structure

```
dehazing-diffusion/
├── joint_diffusion/       # Main codebase (PyTorch)
│   ├── train.py           # Training script
│   ├── inference.py        # Inference script
│   ├── datasets.py         # ZEA dataset loader (mu-law companding)
│   ├── configs/
│   │   ├── training/       # Training configs (tissue & haze models)
│   │   └── inference/      # Inference configs
│   ├── generators/         # Score models (NCSNv2), SDE, sampling, guidance
│   └── utils/              # Corruptors, metrics, B-mode conversion, plotting
├── reproduce_helpers/      # Data synthesis & visualization scripts
└── paper/                  # LaTeX source & plan docs
```

## Quick Start

### 1. Generate Data

```bash
cd reproduce_helpers
python synthesize_dataset.py --out-dir /path/to/data/zea_synth --n-train 1000 --n-val 100
```

### 2. Train Models

Train the tissue (score) and haze (corruptor) models:

Two score models are trained independently:

| Model | Config | Dataset | Description |
|-------|--------|---------|-------------|
| Tissue (score) | `configs/training/score_zea_tissue.yaml` | `zea_tissue` | Clean tissue RF distribution |
| Haze (corruptor) | `configs/training/score_zea_haze.yaml` | `zea_haze` | Haze noise distribution |
```bash
cd joint_diffusion
sbatch slurm_train_tissue.sh
sbatch slurm_train_haze.sh
```
Checkpoints are saved to `wandb/run-<timestamp>-<id>/files/training_checkpoints/`.

Or run directly:

```bash
python train.py -c configs/training/score_zea_tissue.yaml --data_root /path/to/data
python train.py -c configs/training/score_zea_haze.yaml --data_root /path/to/data
```

### 3. Run Inference

Update checkpoint paths in `configs/inference/paper/zea_dehaze_pigdm.yaml`, then:

```bash
sbatch slurm_inference_zea.sh
```

Or:

```bash
python inference.py -e paper/zea_dehaze_pigdm -t denoise
```

Output figures are saved to `figures/`. Set `display_bmode: true` in the inference config to get B-mode images (envelope detection + log compression) matching the paper figures.

[Ground Truth] | [Noisy Input] | [Diffusion Output] | [Noise Posterior (optional)]
    clean tissue    y = tissue+haze   estimated tissue     estimated haze


The inference config (`configs/inference/paper/zea_dehaze_pigdm.yaml`) requires:
- `run_id.sgm`: path to tissue model checkpoint directory
- `sgm.corruptor_run_id`: path to haze model checkpoint directory

### Config Options

| Key | Description |
|-----|-------------|
| `noise_stddev` | Haze strength gamma (default: 0.5) |
| `num_scales` | Number of SDE time steps (default: 200) |
| `guidance` | Guidance method: `pigdm`, `dps`, `projection` |
| `display_bmode` | Convert RF output to B-mode for plotting (default: false) |
| `dynamic_range` | B-mode log compression range in dB (default: [-50, 0]) |
| `keep_track` | Save intermediate steps for animation (default: true) |

## Architecture

- **Score model**: NCSNv2 backbone, 32 base channels, batch norm
- **SDE**: Simple VE-SDE with sigma=25
- **Sampling**: Predictor-Corrector (Euler-Maruyama predictor, no corrector)
- **Guidance**: PiGDM (Pseudoinverse-Guided Diffusion Models)
- **Data**: RF frames `(N, 3, 1024, 64)` with mu-law companding, normalized to [0, 1]

