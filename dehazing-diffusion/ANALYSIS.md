# Dehazing Diffusion: Codebase and Paper Guide

This note focuses on how to read, use, and connect the paper with the code in this repository. It avoids methodological critique and instead explains the structure, flow, and entry points.

## Scope and entry points

- Top-level repository: `dehazing-diffusion/`
- Paper source: `dehazing-diffusion/dehaze-paper/main.tex`
- Utilities used by the paper concepts: `dehazing-diffusion/processing.py`, `dehazing-diffusion/patches.py`
- Joint diffusion framework (general structured noise removal): `dehazing-diffusion/joint_diffusion/`
- Repro script for joint diffusion demo: `dehazing-diffusion/repro/run_joint_diffusion_example.sh`

If you want a quick orientation, start with the paper, then read the sampler/guidance classes, and finally the inverse task wrappers.

## Paper overview and how to read it

The paper proposes ultrasound dehazing as a joint posterior sampling problem with two priors:
- Clean tissue signal (x)
- Structured haze (h)

The forward model is additive RF data:
- y = x + h (beamformed RF)

Key ideas to track while reading `dehazing-diffusion/dehaze-paper/main.tex`:
- Score-based diffusion models for each prior (x and h)
- Joint conditional reverse-time SDE with data-consistency terms
- Companding (mu-law) to normalize RF dynamic range before training and inference
- Patch-based inference with patch interleaving to avoid seams
- Tunable haze strength via gamma in the likelihood term

The algorithm pseudocode is in `dehazing-diffusion/dehaze-paper/algorithm.tex`. Results and metrics are in `dehazing-diffusion/dehaze-paper/results.tex`.

## Codebase map

### 1) Joint diffusion engine (generic)
- `dehazing-diffusion/joint_diffusion/generators/SGM/sampling.py`
  - Implements predictor/corrector samplers for score-based diffusion.
  - Supports conditional sampling with optional structured noise model.
- `dehazing-diffusion/joint_diffusion/generators/SGM/guidance.py`
  - Implements data-consistency / guidance methods (PIGDM, DPS, Projection).
  - Joint update supports x (signal) and n (noise) simultaneously.
- `dehazing-diffusion/joint_diffusion/utils/corruptors.py`
  - Defines corruption models and measurement operators.
  - Used in conditional sampling to enforce data consistency.

These are the main building blocks for joint posterior sampling and correspond to the paper's conditioning step and joint sampling logic.

### 2) Inference and task orchestration
- `dehazing-diffusion/joint_diffusion/inference.py`
  - CLI entry point used by the demo script and configs.
  - Loads dataset, builds model, and runs denoising/sampling tasks.
- `dehazing-diffusion/joint_diffusion/utils/inverse.py`
  - Wraps denoisers and metrics, handles plotting and evaluation.

The inference script is configuration-driven. Configs live under `dehazing-diffusion/joint_diffusion/configs/`.

### 3) Ultrasound-specific utilities
- `dehazing-diffusion/processing.py`
  - Companding, histogram tools, gCNR, KS test, FWHM and autocorrelation helpers.
- `dehazing-diffusion/patches.py`
  - Patch extraction/stitching, windowing, and overlap logic.

These functions align with the paper's companding and patch-based inference sections.

### 4) Paper sources
- `dehazing-diffusion/dehaze-paper/main.tex`
- `dehazing-diffusion/dehaze-paper/algorithm.tex`
- `dehazing-diffusion/dehaze-paper/results.tex`
- `dehazing-diffusion/dehaze-paper/references.bib`

## How the paper maps to code

Paper concept -> Code location

- Score-based diffusion (reverse-time SDE):
  - `dehazing-diffusion/joint_diffusion/generators/SGM/sde_lib.py` (SDE definitions)
  - `dehazing-diffusion/joint_diffusion/generators/SGM/sampling.py` (samplers)

- Joint posterior sampling (signal + haze):
  - `dehazing-diffusion/joint_diffusion/generators/SGM/sampling.py` (noise_model path)
  - `dehazing-diffusion/joint_diffusion/generators/SGM/guidance.py` (joint_update_fn)

- Data-consistency steps (likelihood gradients):
  - `dehazing-diffusion/joint_diffusion/generators/SGM/guidance.py`

- Companding (mu-law):
  - `dehazing-diffusion/processing.py` (companding_tf, companding)

- Patch-based inference and stitching:
  - `dehazing-diffusion/patches.py`

## Practical flow (from data to dehazed output)

A typical end-to-end flow as described in the paper looks like this:

1) Acquire RF ultrasound data (y_RF).
2) Normalize and compand RF data into diffusion domain.
3) Train two score networks:
   - Tissue model s_theta(x, t)
   - Haze model s_phi(h, t)
4) For inference, run joint posterior sampling:
   - Initialize x_t and h_t from y
   - Iterate reverse diffusion with data-consistency updates
   - Use gamma, lambda, kappa to balance reconstruction vs prior
5) If using patches, reconstruct the final image by stitching patches.
6) Expand (inverse compand) to get RF output; render B-mode as needed.

In code, the core steps are implemented by the sampler and guidance classes.

## Running the provided demo (joint diffusion)

This repository includes a general structured denoising demo (CelebA + MNIST) in `joint_diffusion`:

```bash
cd dehazing-diffusion/repro
./run_joint_diffusion_example.sh
```

This script runs `joint_diffusion/inference.py` with a preconfigured experiment and uses the diffusion sampler + guidance logic. It is a good place to understand how configs, datasets, and samplers connect.

## Understanding configs

Configuration is split into training and inference YAML files:

- Training configs: `dehazing-diffusion/joint_diffusion/configs/training/`
- Inference configs: `dehazing-diffusion/joint_diffusion/configs/inference/paper/`
- Sweeps: `dehazing-diffusion/joint_diffusion/configs/sweeps/`

Key config fields you will see:
- `dataset_name`, `image_shape`, `image_range`
- `sde` parameters and diffusion steps
- `lambda_coeff`, `kappa_coeff` for data consistency
- `start_diffusion` (CCDF-style initialization)
- `corruptor` and measurement settings

`inference.py` merges dataset config with inference config and then constructs the model and sampler accordingly.

## Where to add ultrasound data

The `joint_diffusion/datasets.py` file provides standard datasets (MNIST, CelebA, sine noise). To plug in ultrasound RF data, define a dataset loader there or add a parallel dataset module and point the config to it.

Once the dataset yields tensors with the expected shape and range, the existing sampler, guidance, and denoiser classes can be reused.

## Notes on companding usage

Companding is central to the paper's RF-domain training strategy:
- Use `dehazing-diffusion/processing.py` -> `companding()` to map into a balanced range for training.
- Use the inverse companding during reconstruction to return to RF domain.

When reading the algorithm in `dehaze-paper/algorithm.tex`, the companding step appears early in the pipeline and the inverse step appears at the end.

## Key files to read in order

1) Paper core method: `dehazing-diffusion/dehaze-paper/main.tex`
2) Algorithm pseudocode: `dehazing-diffusion/dehaze-paper/algorithm.tex`
3) Sampler and guidance:
   - `dehazing-diffusion/joint_diffusion/generators/SGM/sampling.py`
   - `dehazing-diffusion/joint_diffusion/generators/SGM/guidance.py`
4) Inference flow:
   - `dehazing-diffusion/joint_diffusion/inference.py`
   - `dehazing-diffusion/joint_diffusion/utils/inverse.py`
5) RF utilities:
   - `dehazing-diffusion/processing.py`
   - `dehazing-diffusion/patches.py`

