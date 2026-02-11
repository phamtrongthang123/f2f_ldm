# Drifting Model Debugging Guide

This document serves as a reference for debugging and understanding the Drifting Model implementation in this codebase.

## 📂 Key Files
- **`diffusion_policy/model/drifting/drifting_util.py`**: **Core Logic.** Contains the drifting field calculation (`compute_V`) and the training objective (`compute_drifting_loss`). This is where normalization ($S_j, \lambda_j$) and temperatures are handled.
   - The compute V is exactly the same as Algo 2. 
- **`diffusion_policy/policy/drifting_unet_hybrid_image_policy.py`**: **Policy Wrapper.** Manages the observation encoding and interfaces the U-Net with the drifting loss.

## cloned from the diffusion version. not much changes here. 
- **`diffusion_policy/workspace/train_drifting_unet_hybrid_workspace.py`**: **Training Loop.** Handles the epoch-by-epoch logic, optimization, and logging (e.g., `train_action_mse_error`). There is not much here to compare. This is just the diffusion version cloned. 
- **`drifting_pusht_image.yaml`**: **Main Config.** Defines hyper-parameters like `temperatures`, `batch_size`, and `rollout_every`. There is not much here to compare with the paper. 
- **`drifting_reference/demo/drifting_model_demo.py`**: **Toy Reference.** A standalone PyTorch implementation for 2D tasks used to verify core math.

## 📈 Monitoring Progress
- **`train/mean_score`**: The primary evaluation metric (average of max rewards).
- **`train_action_mse_error`**: A proxy for how well the 1-step generator maps noise to the action space.
- **Batch Size Sensitivity**: Drifting Models require a good estimate of the current distribution ($q$). If the batch size is too small, the "repulsion" field will be noisy, causing instability.
