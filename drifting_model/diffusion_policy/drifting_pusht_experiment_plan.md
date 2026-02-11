# Plan: Close the 0.78 → 0.86 gap for Drifting Policy on PushT Visual

## Context

Your drifting model implementation for PushT Visual peaks at **0.78** test mean score. The paper reports **0.86**. The core algorithm (`compute_V`, anti-symmetric drift, multi-temperature) is correct and matches the paper's Algorithm 2 exactly. The gap likely comes from using the **full ImageNet normalization machinery** (Appendix A.8: S_j feature normalization + λ_j drift normalization) which was designed for multi-feature, multi-scale image generation — not single-feature, low-dimensional robotics control.

The paper's robotics section only says: *"We directly compute drifting loss on the raw representations for control, using no feature space."* It does not specify whether S_j/λ_j normalizations are used. The reference toy demo achieves good results **without any normalization**.

## Experiments (run one at a time, compare against baseline 0.78)

### Experiment 1: Add gradient clipping
**Why**: Paper uses `gradient_clip = 2.0` (config.tex:44). Your workspace has none.
**File**: `diffusion_policy/workspace/train_drifting_unet_hybrid_workspace.py`
**Change**: After `loss.backward()` (line 166), before `self.optimizer.step()` (line 170), add:
```python
torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
```
**Expected**: Stabilizes training, especially early epochs. Low-risk change.

---

### Experiment 2: Remove λ_j drift normalization (HIGHEST PRIORITY)
**Why**: λ_j divides V by its RMS, forcing loss ≈ 1.0 always. When the model nears convergence, V is small but noisy — dividing by tiny λ_j amplifies noise into the target, destabilizing late training. For a single feature (raw actions), there's no need to balance across features.
**File**: `diffusion_policy/model/drifting/drifting_util.py`
**Change** in `compute_drifting_loss`: Remove the λ_j normalization block. Replace lines 101-108:
```python
# BEFORE (remove):
lambda_j = torch.sqrt(torch.mean(torch.sum(V_t**2, dim=-1)) / D + 1e-6)
metrics[f"train/drifting_lambda_T{T}"] = lambda_j.item()
V_t = V_t / (lambda_j.detach() + 1e-6)

# AFTER:
lambda_j = torch.sqrt(torch.mean(torch.sum(V_t**2, dim=-1)) / D + 1e-6)
metrics[f"train/drifting_lambda_T{T}"] = lambda_j.item()
# Keep logging lambda_j for monitoring, but do NOT divide V_t by it
```
**Expected**: Loss should now **decrease during training** (tracks ||V||^2). λ_j metric should also decrease. This is the strongest hypothesis for the gap.

---

### Experiment 3: Wrap V computation in `torch.no_grad()`
**Why**: The reference demo computes V inside `no_grad()`. Your code builds computation graph for V before detaching at target — wasting GPU memory on pairwise distance matrices (512x512). Saving memory may improve training stability or allow experimenting with larger batches.
**File**: `diffusion_policy/model/drifting/drifting_util.py`
**Change** in `compute_drifting_loss`: Wrap the V computation loop:
```python
V_total = torch.zeros_like(x_norm)
with torch.no_grad():
    for T in temperatures:
        scaled_T = T * (D ** 0.5)
        V_t = compute_V(x_norm, y_pos_norm, y_neg_norm, scaled_T)

        lambda_j = torch.sqrt(torch.mean(torch.sum(V_t**2, dim=-1)) / D + 1e-6)
        metrics[f"train/drifting_lambda_T{T}"] = lambda_j.item()
        # (keep or remove lambda_j normalization per Experiment 2)

        V_total = V_total + V_t

target = (x_norm + V_total).detach()  # .detach() still needed for safety
loss = F.mse_loss(x_norm, target)
```
**Note**: Also update `V_total = torch.zeros_like(x_norm)` (currently `torch.zeros_like(x)` — same shape but cleaner semantics).
**Expected**: Reduced GPU memory usage. Functionally equivalent but cleaner.

---

### Experiment 4: Remove S_j feature normalization too (fully simple approach)
**Why**: If Experiment 2 helps but doesn't close the gap, S_j may also over-normalize. Actions are already in [-1,1] from LinearNormalizer. Additional rescaling plus temperature scaling may shift effective temperatures.
**File**: `diffusion_policy/model/drifting/drifting_util.py`
**Change**: Replace `compute_drifting_loss` with a simple version matching the demo style:
```python
def compute_drifting_loss(x, y_pos, y_neg, temperatures=[0.05]):
    metrics = {}
    V_total = torch.zeros_like(x)
    with torch.no_grad():
        for T in temperatures:
            V_t = compute_V(x, y_pos, y_neg, T)  # raw features, raw temperature
            lambda_j = torch.sqrt(torch.mean(torch.sum(V_t**2, dim=-1)) / x.shape[-1] + 1e-6)
            metrics[f"train/drifting_lambda_T{T}"] = lambda_j.item()
            V_total = V_total + V_t

    target = (x + V_total).detach()
    loss = F.mse_loss(x, target)
    return loss, metrics
```
**Important**: Without S_j, the default temperatures {0.02, 0.05, 0.2} may not be well-calibrated for your data scale. You may need to tune them. Check the average pairwise distance in your action space to pick sensible temperatures.
**Expected**: Loss now directly reflects ||V||^2. Convergence visible in loss curve.

---

### Experiment 5: Temperature tuning (if Experiment 4 is used)
**Why**: Without S_j normalization, temperatures need to match the raw data scale. The kernel `exp(-dist/τ)` is sensitive to the ratio dist/τ.
**How**: At the start of training, log the average pairwise L2 distance of normalized actions in a batch. Call this `avg_dist`. Good temperatures should be on the order of `avg_dist` (so the kernel doesn't saturate at 0 or 1). Try:
- `temperatures = [avg_dist * 0.1, avg_dist * 0.3, avg_dist * 1.0]`
- Or simply try `[0.1, 0.3, 1.0]` and `[0.5, 1.0, 2.0]` as alternatives.

---

### Experiment 6: Larger effective sample size (if budget allows)
**Why**: Paper ablation shows N_pos and N_neg matter. With batch_size=512, you have N_pos=N_neg=512 but across **mixed observations**. The effective per-distribution sample size is much smaller.
**How**: If GPU memory allows after Experiment 3, try `batch_size=1024`.

---

## Recommended order
1. **Exp 1** (gradient clipping) — quick, safe, always good to have
2. **Exp 2** (remove λ_j) — highest-impact hypothesis
3. **Exp 3** (no_grad wrapper) — memory savings, combine with above
4. **Exp 2+3 combined** — likely the best combo to try
5. **Exp 4** (remove S_j too) — if still stuck after 2+3
6. **Exp 5** (temperature tuning) — only if Exp 4 is used
7. **Exp 6** (larger batch) — if memory budget allows

## Files to modify
- `diffusion_policy/workspace/train_drifting_unet_hybrid_workspace.py` — gradient clipping (Exp 1)
- `diffusion_policy/model/drifting/drifting_util.py` — all other experiments (Exp 2-5)
- `drifting_pusht_image.yaml` — batch size (Exp 6), temperatures (Exp 5)

## How to verify
- Track `test_mean_score` (primary metric) and `train/drifting_lambda_T{T}` (convergence indicator)
- After Exp 2: loss should decrease over training (not stay at ~1.0)
- After Exp 2: λ_j should decrease over training (confirming V→0)
- Target: test_mean_score >= 0.86 (paper result for PushT Visual)

## Also: clean up debug prints
Remove or gate the `print()` statements in `drifting_util.py` (lines 57-59, 81, 88, 96, 105, 111, 117) before production runs. They spam stdout every batch.
