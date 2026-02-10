# Drifting Model Debugging Guide

This document serves as a reference for debugging and understanding the Drifting Model implementation in this codebase.

## 📂 Key Files
- **`diffusion_policy/model/drifting/drifting_util.py`**: **Core Logic.** Contains the drifting field calculation (`compute_V`) and the training objective (`compute_drifting_loss`). This is where normalization ($S_j, \lambda_j$) and temperatures are handled.
- **`diffusion_policy/policy/drifting_unet_hybrid_image_policy.py`**: **Policy Wrapper.** Manages the observation encoding and interfaces the U-Net with the drifting loss.
- **`diffusion_policy/workspace/train_drifting_unet_hybrid_workspace.py`**: **Training Loop.** Handles the epoch-by-epoch logic, optimization, and logging (e.g., `train_action_mse_error`).
- **`drifting_pusht_image.yaml`**: **Main Config.** Defines hyper-parameters like `temperatures`, `batch_size`, and `rollout_every`.
- **`drifting_reference/drifting_model_demo.py`**: **Toy Reference.** A standalone PyTorch implementation for 2D tasks used to verify core math.

## 🛠️ Debugging the Loss
The drifting loss is conceptually different from Diffusion/MSE losses. It represents the squared norm of the drifting field ($\|\mathbf{V}\|^2$).

1. **The "Normal" Loss Curve**:
   - The loss should start high (reflecting a large drift toward data) and decrease toward **0.0** as the generated distribution matches the data (Equilibrium).
   - If the loss is stuck at **1.0**, it usually indicates that the drift normalization ($\lambda_j$) is being applied without being detached, or it's masking the true magnitude.

2. **Detaching Normalization**:
   - In `drifting_util.py`, ensure that $S_j$ (feature scale) and $\lambda_j$ (drift scale) are `.detach()`ed. They should act as "frozen" statistics from the batch to stabilize gradients without the optimizer "cheating" the loss by simply scaling weights.

3. **Multi-Temperature Aggregation**:
   - We use multiple temperatures (e.g., `0.02, 0.05, 0.2`).
   - Smaller temperatures focus on local structure; larger temperatures focus on global distribution. If the model is "blurry," check the high temperatures; if it's "spiky/unstable," check the low temperatures.

## 📈 Monitoring Progress
- **`train/mean_score`**: The primary evaluation metric (average of max rewards).
- **`train_action_mse_error`**: A proxy for how well the 1-step generator maps noise to the action space.
- **Batch Size Sensitivity**: Drifting Models require a good estimate of the current distribution ($q$). If the batch size is too small, the "repulsion" field will be noisy, causing instability.
