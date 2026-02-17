# High Volume Rate 3D Ultrasound Reconstruction with Diffusion Models

Reproducing the paper's pipeline using the [ZEA toolbox](https://github.com/zeahub/zea). This demo uses a pretrained 2D cardiac echo diffusion model and the CAMUS dataset to demonstrate the full algorithm: DPS-based interpolation of missing elevation planes, TV smoothness regularization, and SeqDiff temporal acceleration.

All runnable code is at the repository root.

## Quick Start

All jobs run inside an Apptainer container (`qwen3vl-cu128.sif`) to avoid glibc issues on HPC nodes.

### 1. Setup (install deps + download data)

```bash
sbatch setup_env.sh
```

This submits a SLURM job that runs `run_setup_env.sh` inside the container. It installs/upgrades `zea`, verifies JAX GPU + Keras + model loading, and downloads the CAMUS dataset.

### 2. Run reconstruction pipeline

```bash
sbatch slurm_reconstruct.sh
```

This submits a SLURM job that runs `run_reconstruct.sh` inside the container, executing all 5 steps sequentially.

### File structure

```
setup_env.sh          ← sbatch wrapper (submit this)
run_setup_env.sh      ← actual setup logic (runs inside apptainer)

slurm_reconstruct.sh  ← sbatch wrapper (submit this)
run_reconstruct.sh    ← actual pipeline logic (runs inside apptainer)
```

### 3. Quick test (faster iteration)

Edit these values in the scripts before running:

| Parameter | Quick test | Paper-faithful |
|-----------|-----------|----------------|
| `N_STEPS` | 50 | 200 |
| `N_ELEVATION` | 8 | 16 |
| `ACCEL_RATE` | 2 | 4 |
| `OMEGA` | 35.0 | 35.0 |
| `ZETA` | 0.001 | 0.001 |
| `SEQDIFF_TAU` | 50 | 50 |

## Pipeline Overview

```
00_download_data.py       Download + cache CAMUS dataset
        |
01_verify_prior.py        Unconditional sampling to verify model works
        |
02_prepare_pseudo_volume.py   Stack 2D images as pseudo-3D volume
        |
03_reconstruct_volume.py  DPS reconstruction of missing planes + TV smoothness
        |
   +---------+----------+
   |                     |
04_seqdiff_temporal.py   05_evaluate.py
SeqDiff speedup demo     Metrics + visualization
```

## What Each Script Does

| Script | Description |
|--------|-------------|
| `setup_env.sh` | SLURM wrapper → runs `run_setup_env.sh` inside apptainer |
| `run_setup_env.sh` | Installs zea, verifies JAX/GPU/Keras, downloads CAMUS data |
| `slurm_reconstruct.sh` | SLURM wrapper → runs `run_reconstruct.sh` inside apptainer |
| `run_reconstruct.sh` | Runs all 5 Python steps sequentially |
| `00_download_data.py` | Pre-downloads CAMUS dataset for offline HPC use |
| `01_verify_prior.py` | Generates 8 unconditional samples, saves `outputs/01_prior_samples.png` |
| `02_prepare_pseudo_volume.py` | Creates `outputs/pseudo_volume.npy` (16 planes) and `pseudo_volume_t2.npy` |
| `03_reconstruct_volume.py` | Reconstructs missing planes via DPS + TV, saves `outputs/reconstructed_volume.npy` |
| `04_seqdiff_temporal.py` | Compares cold-start (200 steps) vs warm-start (50 steps), saves `outputs/04_seqdiff_comparison.png` |
| `05_evaluate.py` | Computes PSNR/SSIM/LPIPS, saves `outputs/05_evaluation.png` and `outputs/05_elevation_profile.png` |

## Outputs

All outputs go to `outputs/`. Key files:

- `01_prior_samples.png` — sanity check that the diffusion prior works
- `pseudo_volume.npy` — ground truth pseudo-3D volume
- `reconstructed_volume.npy` — reconstructed volume after DPS + TV
- `04_seqdiff_comparison.png` — cold vs warm-start comparison
- `05_evaluation.png` — GT / reconstruction / difference side-by-side
- `05_elevation_profile.png` — inter-plane smoothness plot

## Expected Metrics Behavior

The CAMUS sample dataset only contains **2 unique images**. The pseudo-volume cycles them as (A, B, A, B, ...). With `ACCEL_RATE=4`, observed planes (0, 4, 8, ...) are all image A. This creates an adversarial scenario where DPS guidance only ever sees image A, but half the missing planes are image B (a completely different patient).

**Observed results** (N_STEPS=200, OMEGA=35, ZETA=0.001):

| Metric | Overall (missing planes) |
|--------|--------------------------|
| PSNR   | ~13.0 dB                 |
| SSIM   | ~0.12                    |
| LPIPS  | ~0.37                    |

Per-plane metrics show a clear bimodal pattern:

| Missing plane type | Example planes | PSNR | SSIM | Why |
|--------------------|---------------|------|------|-----|
| Even midpoints (image A) | 2, 6, 10, 14, ... | ~15-16 dB | ~0.3 | Same content as observed planes; DPS + TV can partially reconstruct |
| Odd planes (image B) | 1, 3, 5, 7, ... | ~11-12 dB | ~0.03 | Different patient image; no information available from observations |

**These low scores are expected and do not indicate an algorithm bug.** The ~0.03 SSIM for odd planes simply measures how different two unrelated patient images are — no reconstruction method can recover image B from observations of image A alone.

With real 3D ultrasound data (smooth, continuous variation along elevation), all missing planes would be similar to their observed neighbors, and scores should be substantially higher.

## Known Limitations

**Data limitations** (CAMUS sample dataset, not the paper's proprietary 3D cardiac data):
- Only 2 unique source images available — causes bimodal metrics (see above)
- No real elevation coherence between stacked planes (different patients)
- Pretrained model is on EchoNet-Dynamic (A4C views), not actual B-plane slices
- No speckle tracking or out-of-distribution experiments possible without real 3D volumes
