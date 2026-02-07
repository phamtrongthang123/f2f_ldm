"""Latent-MAE feature encoder with multi-scale feature extraction.

For latent-space generation, the pipeline is:
    latent → LatentMAE encoder → multi-scale features

The feature encoder extracts features at multiple scales and spatial granularities
as described in Appendix B.4 of the paper.

Latent-MAE ResNet encoder stages (BasicBlocks, GroupNorm, base width C=256):
    conv1 → stage1 (3 basic blocks, 32×32×C)
           → stage2 (4 basic blocks, 16×16×2C)
           → stage3 (6 basic blocks, 8×8×4C)
           → stage4 (3 basic blocks, 4×4×8C)

Feature extraction: output of every 2 residual blocks + final output per stage.
For each feature map, produce per-location, global, and patch-level statistics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .latent_mae import ResNetEncoder


class LatentMAEFeatureExtractor(nn.Module):
    """Multi-scale feature extractor using pre-trained latent-MAE encoder.

    Operates directly on 32×32×4 VAE latents — no VAE decode needed.
    """

    def __init__(self, checkpoint_path=None, base_width=256):
        super().__init__()

        self.encoder = ResNetEncoder(
            in_channels=4,
            base_width=base_width,
        )

        if checkpoint_path is not None:
            self._load_mae_weights(checkpoint_path)

        # Freeze all parameters — gradients flow through to the generator
        self.requires_grad_(False)

    def _load_mae_weights(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # MAE checkpoint stores encoder under 'ema_model' with 'encoder.' prefix
        state_dict = checkpoint.get("ema_model", checkpoint.get("model", checkpoint))

        encoder_state = {}
        for k, v in state_dict.items():
            if k.startswith("encoder."):
                encoder_state[k[len("encoder."):]] = v

        if not encoder_state:
            # Fallback: try loading directly (in case it's already encoder-only)
            encoder_state = state_dict

        missing, unexpected = self.encoder.load_state_dict(encoder_state, strict=False)
        if missing:
            print(f"MAE loading — missing keys: {missing}")
        if unexpected:
            print(f"MAE loading — unexpected keys: {unexpected}")

    def _get_stage_features(self, stage, x):
        """Run through a stage and extract features every 2 blocks + final.

        Args:
            stage: nn.Sequential of residual blocks
            x: input tensor

        Returns:
            features: list of feature maps at extraction points
            x: output of the stage
        """
        features = []
        for i, block in enumerate(stage):
            x = block(x)
            # Extract every 2 blocks (0-indexed: after block 1, 3, 5, ...)
            # and always the final block
            if (i + 1) % 2 == 0 or i == len(stage) - 1:
                features.append(x)
        return features, x

    def _extract_multiscale(self, feature_map):
        """Extract multi-scale feature vectors from a single feature map.

        For a feature map of shape [B, C, H, W], produce:
        (a) H×W per-location vectors (each C-dim)
        (b) 1 global mean + 1 global std (each C-dim)
        (c) (H/2)×(W/2) means + stds over 2×2 patches (each C-dim)
        (d) (H/4)×(W/4) means + stds over 4×4 patches (each C-dim)

        Returns list of [B, num_vectors, C] tensors.
        """
        B, C, H, W = feature_map.shape
        result = []

        # (a) Per-location: [B, H*W, C]
        per_loc = feature_map.reshape(B, C, H * W).permute(0, 2, 1)  # [B, H*W, C]
        result.append(per_loc)

        # (b) Global mean and std: each [B, 1, C]
        spatial = feature_map.reshape(B, C, -1)  # [B, C, H*W]
        global_mean = spatial.mean(dim=-1, keepdim=True).permute(0, 2, 1)  # [B, 1, C]
        global_std = spatial.std(dim=-1, keepdim=True).permute(0, 2, 1)  # [B, 1, C]
        result.append(global_mean)
        result.append(global_std)

        # (c) 2×2 patch means and stds
        if H >= 2 and W >= 2:
            patches_2 = feature_map.reshape(B, C, H // 2, 2, W // 2, 2)
            patches_2 = patches_2.permute(0, 2, 4, 1, 3, 5).reshape(B, (H // 2) * (W // 2), C, 4)
            patch_mean_2 = patches_2.mean(dim=-1)  # [B, (H/2)*(W/2), C]
            patch_std_2 = patches_2.std(dim=-1)  # [B, (H/2)*(W/2), C]
            result.append(patch_mean_2)
            result.append(patch_std_2)

        # (d) 4×4 patch means and stds
        if H >= 4 and W >= 4:
            patches_4 = feature_map.reshape(B, C, H // 4, 4, W // 4, 4)
            patches_4 = patches_4.permute(0, 2, 4, 1, 3, 5).reshape(B, (H // 4) * (W // 4), C, 16)
            patch_mean_4 = patches_4.mean(dim=-1)  # [B, (H/4)*(W/4), C]
            patch_std_4 = patches_4.std(dim=-1)  # [B, (H/4)*(W/4), C]
            result.append(patch_mean_4)
            result.append(patch_std_4)

        return result

    def extract_features(self, latents):
        """Extract multi-scale features from VAE latents.

        Args:
            latents: [B, 4, 32, 32] VAE-encoded latents

        Returns:
            features: list of [B, num_vectors, C] tensors, one per feature group.
                      Each tensor contains C-dimensional feature vectors.
        """
        all_features = []

        # Encoder stem
        x0 = self.encoder.relu(self.encoder.gn1(self.encoder.conv1(latents)))  # [B, C, 32, 32]

        # (e) For encoder input layer: mean of x^2 per channel → [B, 1, C]
        x0_sq_mean = x0.pow(2).mean(dim=[2, 3], keepdim=False).unsqueeze(1)  # [B, 1, C]
        all_features.append(x0_sq_mean)

        # Stage 1: 3 basic blocks, 32×32×C
        feats1, x = self._get_stage_features(self.encoder.stage1, x0)
        for fm in feats1:
            all_features.extend(self._extract_multiscale(fm))

        # Stage 2: 4 basic blocks, 16×16×2C
        feats2, x = self._get_stage_features(self.encoder.stage2, x)
        for fm in feats2:
            all_features.extend(self._extract_multiscale(fm))

        # Stage 3: 6 basic blocks, 8×8×4C
        feats3, x = self._get_stage_features(self.encoder.stage3, x)
        for fm in feats3:
            all_features.extend(self._extract_multiscale(fm))

        # Stage 4: 3 basic blocks, 4×4×8C
        feats4, x = self._get_stage_features(self.encoder.stage4, x)
        for fm in feats4:
            all_features.extend(self._extract_multiscale(fm))

        return all_features

    def forward(self, latents):
        """Alias for extract_features."""
        return self.extract_features(latents)
