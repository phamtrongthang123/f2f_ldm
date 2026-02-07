"""MoCo v2 ResNet-50 feature encoder with multi-scale feature extraction.

For latent-space generation, the pipeline is:
    latent → VAE decoder → pixel image → MoCo ResNet-50 → multi-scale features

The feature encoder extracts features at multiple scales and spatial granularities
as described in Appendix B.4 of the paper.

MoCo v2 ResNet-50 stages:
    conv1 → layer1 (3 bottleneck blocks, 64×64×256)
           → layer2 (4 bottleneck blocks, 32×32×512)
           → layer3 (6 bottleneck blocks, 16×16×1024)
           → layer4 (3 bottleneck blocks, 8×8×2048)

Feature extraction: output of every 2 residual blocks + final output per stage.
For each feature map, produce per-location, global, and patch-level statistics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class MoCoV2FeatureExtractor(nn.Module):
    """Multi-scale feature extractor using pre-trained MoCo v2 ResNet-50."""

    def __init__(self, checkpoint_path=None):
        super().__init__()

        # ImageNet normalization for MoCo v2
        self.register_buffer("pixel_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # Load ResNet-50 backbone
        resnet = models.resnet50(weights=None)

        # Store layers for explicit forward pass
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # 3 bottleneck blocks → 256 channels
        self.layer2 = resnet.layer2  # 4 bottleneck blocks → 512 channels
        self.layer3 = resnet.layer3  # 6 bottleneck blocks → 1024 channels
        self.layer4 = resnet.layer4  # 3 bottleneck blocks → 2048 channels

        # Load MoCo v2 weights if provided
        if checkpoint_path is not None:
            self._load_moco_weights(checkpoint_path)

        # Freeze all parameters — we don't train the feature encoder itself,
        # but gradients flow through it to the generator (via VAE decoder).
        for param in self.parameters():
            param.requires_grad = False

    def _load_moco_weights(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)

        # MoCo v2 stores encoder weights with "module.encoder_q." prefix
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module.encoder_q."):
                new_k = k.replace("module.encoder_q.", "")
                # Skip FC head
                if new_k.startswith("fc."):
                    continue
                new_state_dict[new_k] = v

        # Load into the resnet components
        resnet_state = {}
        for k, v in new_state_dict.items():
            resnet_state[k] = v

        # Map to our module structure
        missing, unexpected = self.load_state_dict(resnet_state, strict=False)
        if missing:
            print(f"MoCo v2 loading — missing keys (expected for FC): {missing}")

    def _get_stage_features(self, stage, x):
        """Run through a stage and extract features every 2 blocks + final.

        Args:
            stage: nn.Sequential of bottleneck blocks
            x: input tensor

        Returns:
            features: list of (feature_map, H, W, C) at extraction points
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

    def extract_features(self, images):
        """Extract multi-scale features from images.

        Args:
            images: [B, 3, 256, 256] pixel-space images in [0, 1] range

        Returns:
            features: list of [B, num_vectors, C] tensors, one per feature group.
                      Each tensor contains C-dimensional feature vectors.
        """
        all_features = []

        # ImageNet normalization
        images = (images - self.pixel_mean) / self.pixel_std

        # Stem
        x = self.conv1(images)    # [B, 64, 128, 128]
        x = self.bn1(x)
        x0 = self.relu(x)        # input layer features

        # (e) For encoder input layer: mean of x^2 per channel → [B, 1, C0]
        x0_sq_mean = x0.pow(2).mean(dim=[2, 3], keepdim=False).unsqueeze(1)  # [B, 1, 64]
        all_features.append(x0_sq_mean)

        x = self.maxpool(x0)     # [B, 64, 64, 64]

        # Stage 1: layer1 — 3 blocks, output 64×64×256
        feats1, x = self._get_stage_features(self.layer1, x)
        for fm in feats1:
            all_features.extend(self._extract_multiscale(fm))

        # Stage 2: layer2 — 4 blocks, output 32×32×512
        feats2, x = self._get_stage_features(self.layer2, x)
        for fm in feats2:
            all_features.extend(self._extract_multiscale(fm))

        # Stage 3: layer3 — 6 blocks, output 16×16×1024
        feats3, x = self._get_stage_features(self.layer3, x)
        for fm in feats3:
            all_features.extend(self._extract_multiscale(fm))

        # Stage 4: layer4 — 3 blocks, output 8×8×2048
        feats4, x = self._get_stage_features(self.layer4, x)
        for fm in feats4:
            all_features.extend(self._extract_multiscale(fm))

        return all_features

    def forward(self, images):
        """Alias for extract_features."""
        return self.extract_features(images)
