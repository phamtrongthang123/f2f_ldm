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

```bash
cd joint_diffusion
sbatch slurm_train_tissue.sh
sbatch slurm_train_haze.sh
```

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

## References

```bib
@article{stevens2023dehazing,
  title={Dehazing Ultrasound using Diffusion Models},
  author={Stevens, Tristan and Meral, Can and Yu, Jason and Apostolakis, Iason and Robert, Jean-Luc and van Sloun, Ruud},
  journal={arXiv preprint arXiv:2307.11204},
  year={2023}
}
```
