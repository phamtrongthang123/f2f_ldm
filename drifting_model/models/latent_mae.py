"""Latent-space MAE (Masked Autoencoder) feature encoder.

ResNet with BasicBlocks (two 3×3 convs), GroupNorm, operating directly on
32×32×4 VAE latents. Used as the feature encoder for drifting model training.

Architecture per appendix_impl.tex:
- Encoder: ResNet-34 style with [3,4,6,3] basic blocks, base width C=256
- Decoder: U-Net style with skip connections, 4 output channels
- MAE training: 50% random masking with 2×2 patches, L2 reconstruction loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Basic residual block: two 3×3 convs with GroupNorm and ReLU.

    Matches appendix_impl.tex:99: "each consisting of two 3×3 convolutions".
    """

    def __init__(self, in_channels, out_channels, stride=1, num_groups=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups, out_channels)

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.GroupNorm(num_groups, out_channels),
            )

    def forward(self, x):
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out = self.relu(out + self.shortcut(x))
        return out


class ResNetEncoder(nn.Module):
    """ResNet encoder with BasicBlocks and GroupNorm.

    Per appendix_impl.tex:90-103:
    - conv1: 3×3, stride 1, no maxpool
    - 4 stages: [3,4,6,3] blocks
    - Channels: {C, 2C, 4C, 8C} for base width C
    - Spatial sizes: {32², 16², 8², 4²}
    """

    def __init__(self, in_channels=4, base_width=256, num_groups=32):
        super().__init__()
        C = base_width

        # Stem: 3×3 conv, stride 1, no downsampling, no maxpool
        self.conv1 = nn.Conv2d(in_channels, C, 3, stride=1, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups, C)
        self.relu = nn.ReLU(inplace=True)

        # Stage 1: 3 blocks at C, 32×32
        self.stage1 = self._make_stage(C, C, 3, stride=1, num_groups=num_groups)
        # Stage 2: 4 blocks at 2C, 16×16
        self.stage2 = self._make_stage(C, 2 * C, 4, stride=2, num_groups=num_groups)
        # Stage 3: 6 blocks at 4C, 8×8
        self.stage3 = self._make_stage(2 * C, 4 * C, 6, stride=2, num_groups=num_groups)
        # Stage 4: 3 blocks at 8C, 4×4
        self.stage4 = self._make_stage(4 * C, 8 * C, 3, stride=2, num_groups=num_groups)

    @staticmethod
    def _make_stage(in_channels, out_channels, num_blocks, stride, num_groups):
        blocks = [BasicBlock(in_channels, out_channels, stride=stride, num_groups=num_groups)]
        for _ in range(1, num_blocks):
            blocks.append(BasicBlock(out_channels, out_channels, stride=1, num_groups=num_groups))
        return nn.Sequential(*blocks)

    def forward(self, x):
        """Returns (x0, f1, f2, f3, f4) for decoder skip connections."""
        x0 = self.relu(self.gn1(self.conv1(x)))  # [B, C, 32, 32]
        f1 = self.stage1(x0)   # [B, C,  32, 32]
        f2 = self.stage2(f1)   # [B, 2C, 16, 16]
        f3 = self.stage3(f2)   # [B, 4C, 8,  8]
        f4 = self.stage4(f3)   # [B, 8C, 4,  4]
        return x0, f1, f2, f3, f4


class UpsampleBlock(nn.Module):
    """Bilinear 2× upsample → cat skip → GN → two 3×3 convs + GN + ReLU."""

    def __init__(self, in_channels, skip_channels, out_channels, num_groups=32):
        super().__init__()
        cat_channels = in_channels + skip_channels
        self.conv1 = nn.Conv2d(cat_channels, out_channels, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.gn1(self.conv1(x)))
        x = self.relu(self.gn2(self.conv2(x)))
        return x


class FinalBlock(nn.Module):
    """Final block: cat with skip (no upsample, already at target resolution)."""

    def __init__(self, in_channels, skip_channels, out_channels, num_groups=32):
        super().__init__()
        cat_channels = in_channels + skip_channels
        self.conv1 = nn.Conv2d(cat_channels, out_channels, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x, skip):
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.gn1(self.conv1(x)))
        x = self.relu(self.gn2(self.conv2(x)))
        return x


class UNetDecoder(nn.Module):
    """U-Net decoder with skip connections from the encoder.

    Per appendix_impl.tex:105-113:
    - 3×3 conv on f4 (4×4)
    - 3 UpsampleBlocks: 4→8, 8→16, 16→32
    - 1 FinalBlock at 32×32 (cat with x0, no upsample)
    - 1×1 conv → 4 output channels
    """

    def __init__(self, base_width=256, out_channels=4, num_groups=32):
        super().__init__()
        C = base_width

        # Initial 3×3 conv on f4
        self.init_conv = nn.Conv2d(8 * C, 8 * C, 3, padding=1, bias=False)
        self.init_gn = nn.GroupNorm(num_groups, 8 * C)
        self.init_relu = nn.ReLU(inplace=True)

        # Block 1: (8C + 4C) → 4C, 4→8
        self.up1 = UpsampleBlock(8 * C, 4 * C, 4 * C, num_groups)
        # Block 2: (4C + 2C) → 2C, 8→16
        self.up2 = UpsampleBlock(4 * C, 2 * C, 2 * C, num_groups)
        # Block 3: (2C + C) → C, 16→32
        self.up3 = UpsampleBlock(2 * C, C, C, num_groups)
        # Block 4: (C + C) → C, 32×32 (no upsample)
        self.final = FinalBlock(C, C, C, num_groups)

        # Output projection
        self.out_conv = nn.Conv2d(C, out_channels, 1)

    def forward(self, x0, f1, f2, f3, f4):
        """Decode encoder features to reconstruction.

        Args:
            x0: [B, C, 32, 32] conv1 output (stem)
            f1: [B, C, 32, 32] stage1 output
            f2: [B, 2C, 16, 16] stage2 output
            f3: [B, 4C, 8, 8] stage3 output
            f4: [B, 8C, 4, 4] stage4 output

        Returns:
            [B, 4, 32, 32] reconstructed latent
        """
        x = self.init_relu(self.init_gn(self.init_conv(f4)))  # [B, 8C, 4, 4]
        x = self.up1(x, f3)   # [B, 4C, 8, 8]
        x = self.up2(x, f2)   # [B, 2C, 16, 16]
        x = self.up3(x, f1)   # [B, C, 32, 32]
        x = self.final(x, x0) # [B, C, 32, 32]
        x = self.out_conv(x)   # [B, 4, 32, 32]
        return x


class LatentMAE(nn.Module):
    """Masked Autoencoder for VAE latent space.

    Training: randomly mask 50% of 2×2 patches on 32×32 latents,
    reconstruct, L2 loss on masked regions only.

    Feature extraction: run encoder only (no masking).
    """

    def __init__(self, in_channels=4, base_width=256, num_groups=32, mask_ratio=0.5):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.encoder = ResNetEncoder(in_channels, base_width, num_groups)
        self.decoder = UNetDecoder(base_width, in_channels, num_groups)

    def _create_mask(self, batch_size, device):
        """Create random 2×2 patch mask on 32×32 grid.

        Returns:
            mask: [B, 1, 32, 32] binary mask (1 = visible, 0 = masked)
        """
        # 16×16 grid of 2×2 patches
        grid = torch.ones(batch_size, 1, 16, 16, device=device)
        # Independent Bernoulli — mask_ratio fraction are masked (set to 0)
        grid = (torch.rand(batch_size, 1, 16, 16, device=device) > self.mask_ratio).float()
        # Upsample to 32×32 with nearest (each patch covers 2×2 pixels)
        mask = F.interpolate(grid, scale_factor=2, mode="nearest")  # [B, 1, 32, 32]
        return mask

    def forward(self, latents):
        """MAE forward: mask → encode → decode → L2 loss on masked regions.

        Args:
            latents: [B, 4, 32, 32] VAE-encoded latents

        Returns:
            loss: scalar reconstruction loss on masked regions
            recon: [B, 4, 32, 32] full reconstruction
        """
        mask = self._create_mask(latents.shape[0], latents.device)

        # Zero out masked patches
        masked_input = latents * mask

        # Encode and decode
        x0, f1, f2, f3, f4 = self.encoder(masked_input)
        recon = self.decoder(x0, f1, f2, f3, f4)

        # L2 loss on masked regions only
        inv_mask = 1.0 - mask
        num_masked = inv_mask.sum().clamp(min=1.0)
        loss = ((recon - latents).pow(2) * inv_mask).sum() / num_masked

        return loss, recon

    def forward_encoder(self, latents):
        """Encoder-only forward (for feature extraction, no masking).

        Args:
            latents: [B, 4, 32, 32] VAE-encoded latents

        Returns:
            (x0, f1, f2, f3, f4) encoder feature maps
        """
        return self.encoder(latents)
