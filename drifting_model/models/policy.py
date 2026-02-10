import sys
import os
import torch
import torch.nn as nn

# Ensure diffusion_policy is in the path for imports
# This matches the pattern used in the training/eval scripts
diffusion_policy_path = os.path.join(os.getcwd(), 'diffusion_policy')
if diffusion_policy_path not in sys.path:
    sys.path.append(diffusion_policy_path)

try:
    from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
except ImportError as e:
    # Fallback for different directory structures if needed
    try:
        from model.diffusion.conditional_unet1d import ConditionalUnet1D
    except ImportError:
        raise ImportError(f"Could not import ConditionalUnet1D from diffusion_policy: {e}")

class DriftingPolicy(nn.Module):
    """CNN-based generator for robotics control via Drifting.
    
    Wraps the reference ConditionalUnet1D from the diffusion_policy repository
    to implement the Drifting Model training and inference logic.
    """
    
    def __init__(
        self,
        action_dim,
        obs_dim,
        horizon=16,
        down_dims=[256, 512, 1024],
        diffusion_step_embed_dim=256,
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True
    ):
        super().__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.horizon = horizon
        self.normalizer = None
        
        # Use the reference implementation directly
        self.unet = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=obs_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )
        
    def set_normalizer(self, normalizer):
        self.normalizer = normalizer

    def forward(self, noise, obs, timestep=None):
        """
        Args:
            noise: [B, T, action_dim] Gaussian noise
            obs: [B, obs_dim] Observation vector (already flattened history)
            timestep: [B] or scalar. Defaults to 0 for one-step drifting.
            
        Returns:
            action: [B, T, action_dim] (UNNORMALIZED if normalizer is set)
        """
        if timestep is None:
            timestep = torch.zeros((noise.shape[0],), device=noise.device, dtype=torch.long)
            
        # The reference Unet expects (B, T, D) and returns (B, T, D)
        # after internal rearrangements.
        x = self.unet(noise, timestep, global_cond=obs)
        
        if self.normalizer is not None:
            x = self.normalizer['action'].unnormalize(x)
            
        return x

    def compute_loss(self, obs, action_gt, temperatures=[0.02, 0.05, 0.2]):
        """
        Implements Drifting Model training objective.
        
        Args:
            obs: [B, obs_dim]
            action_gt: [B, T, action_dim] real data samples (y+) (UNNORMALIZED)
        """
        from drifting_loss import compute_drifting_loss
        
        if self.normalizer is not None:
            action_gt = self.normalizer['action'].normalize(action_gt)
        
        B, T, Da = action_gt.shape
        
        # Sample noise (epsilon)
        noise = torch.randn_like(action_gt)
        
        # Generate samples (x) - we need NORMALIZED x for loss
        timestep = torch.zeros((B,), device=noise.device, dtype=torch.long)
        x_gen = self.unet(noise, timestep, global_cond=obs)
        
        # Reshape for compute_drifting_loss: [B, 1, D] where D = T*Da
        gen_latent = x_gen.reshape(B, -1)
        pos_latent = action_gt.reshape(B, -1)
        
        # Drift computation
        loss = compute_drifting_loss(
            gen_features=[],
            pos_features=[],
            neg_features=[],
            uncond_features=[],
            temperatures=temperatures,
            cfg_weights=None,
            gen_latent=gen_latent,
            pos_latent=pos_latent,
            neg_latent=gen_latent, # x is its own negative
        )
        
        return loss
