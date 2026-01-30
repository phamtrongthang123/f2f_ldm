# Joint Diffusion (PyTorch)

PyTorch port of the joint posterior sampling framework for removing structured noise with diffusion models.

## Training

Two score models are trained independently:

| Model | Config | Dataset | Description |
|-------|--------|---------|-------------|
| Tissue (score) | `configs/training/score_zea_tissue.yaml` | `zea_tissue` | Clean tissue RF distribution |
| Haze (corruptor) | `configs/training/score_zea_haze.yaml` | `zea_haze` | Haze noise distribution |

```bash
python train.py -c configs/training/score_zea_tissue.yaml --data_root /path/to/data
python train.py -c configs/training/score_zea_haze.yaml --data_root /path/to/data
```

SLURM: `sbatch slurm_train_tissue.sh` / `sbatch slurm_train_haze.sh`

Checkpoints are saved to `wandb/run-<timestamp>-<id>/files/training_checkpoints/`.

## Inference

```bash
python inference.py -e paper/zea_dehaze_pigdm -t denoise
```

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
