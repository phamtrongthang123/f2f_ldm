# Drifting Models for Robotics Control

This repository reproduces the **Robotics Control** results from the Drifting Model paper. It applies the Drifting Model (a one-step generative model) to multiple tasks (Push-T, Lift, Can, Square, Kitchen) using the Diffusion Policy codebase conventions.

## 1. Environment Setup

Create a conda environment with the necessary dependencies:

```bash
conda env create -f environment_robotics.yaml
conda activate drifting_robotics
```

## 2. Data Preparation

We use the datasets from the Diffusion Policy project.

1.  **Clone the Diffusion Policy repository** (already included if you see `diffusion_policy/` folder).
2.  **Download the datasets**:

```bash
bash download_data.sh
```

This will download `pusht.zip`, `robomimic_*.zip`, etc., to `data/` and extract them.

## 3. Training

Train the Drifting Policy on a specific task.

**Push-T:**
```bash
python train_robotics.py --task pusht --epochs 300
```

**Robomimic Tasks (Lift, Can, Square):**
```bash
python train_robotics.py --task lift --data_dir data/robomimic_lift_ph.hdf5
```

**Kitchen:**
```bash
python train_robotics.py --task kitchen --data_dir data/kitchen
```

- **Model**: A 1D DiT-based generator (`models/policy.py`).
- **Loss**: Drifting Loss on raw action trajectories.

## 4. Evaluation

Evaluate the trained policy in the simulation (requires MuJoCo for non-PushT tasks).

```bash
python eval_robotics.py \
    --ckpt checkpoints/robotics/pusht/epoch_300.pt \
    --render
```

## HPC Training

Use the provided SLURM script. You can configure the task and other parameters using environment variables:

```bash
# Default (Push-T)
sbatch scripts/slurm_train_robotics.sh

# Specific Task (e.g., Lift)
export TASK=lift
export DATA_DIR=data/robomimic_lift_ph.hdf5
sbatch scripts/slurm_train_robotics.sh
```

## Project Structure

- `train_robotics.py`: Training script supporting multiple tasks.
- `eval_robotics.py`: Evaluation script (currently optimized for Push-T).
- `models/policy.py`: 1D DiT architecture for control.
- `drifting_loss.py`: Core drifting loss implementation.