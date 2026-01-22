import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from typing import Optional


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class BarRemovalUNet(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 3, base_channels: int = 32, residual_scale: float = 1.0) -> None:
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 16)
        self.dec4 = UpBlock(base_channels * 16, base_channels * 8)
        self.dec3 = UpBlock(base_channels * 8, base_channels * 4)
        self.dec2 = UpBlock(base_channels * 4, base_channels * 2)
        self.dec1 = UpBlock(base_channels * 2, base_channels)
        self.out = nn.Conv2d(base_channels, out_ch, kernel_size=1)
        self.residual_scale = residual_scale
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        residual = self.out(d1)
        return inp + residual * self.residual_scale


def load_bar_removal_model(model_path: str, device: torch.device, weight_dtype: Optional[torch.dtype] = None) -> BarRemovalUNet:
    checkpoint = torch.load(model_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        base_channels = int(checkpoint.get("base_channels", 32))
        residual_scale = float(checkpoint.get("residual_scale", 1.0))
    else:
        state_dict = checkpoint
        base_channels = 32
        residual_scale = 1.0
    model = BarRemovalUNet(base_channels=base_channels, residual_scale=residual_scale)
    model.load_state_dict(state_dict, strict=True)
    dtype = weight_dtype if device.type == "cuda" and weight_dtype is not None else torch.float32
    model.to(device=device, dtype=dtype)
    model.eval()
    return model


def apply_bar_removal(model: BarRemovalUNet, image: Image.Image, device: torch.device, resolution: int) -> Image.Image:
    transform = T.Compose(
        [
            T.Resize((resolution, resolution), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
        ]
    )
    dtype = next(model.parameters()).dtype
    with torch.inference_mode():
        tensor = transform(image).unsqueeze(0).to(device=device, dtype=dtype)
        output = model(tensor).clamp(0.0, 1.0).squeeze(0).cpu()
    return T.ToPILImage()(output)
