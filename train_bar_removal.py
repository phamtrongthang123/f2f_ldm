import argparse
import glob
import os
from typing import Dict, List

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from tqdm import tqdm

from bar_removal import BarRemovalUNet


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def list_images(image_dir: str) -> List[str]:
    paths: List[str] = []
    for ext in IMAGE_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(image_dir, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(image_dir, f"*{ext.upper()}")))
    return sorted(set(paths))


def make_bar_mask(width: int, height: int, bar_width: int, bar_gap: int, offset: int) -> torch.Tensor:
    mask = torch.zeros(1, width)
    pos = offset
    while pos < width:
        end = min(width, pos + bar_width)
        mask[:, pos:end] = 1.0
        pos += bar_width + bar_gap
    return mask.repeat(height, 1)


def add_vertical_bars(image: torch.Tensor, params: Dict[str, float]) -> torch.Tensor:
    if torch.rand(1).item() > params["bar_prob"]:
        return image
    _, height, width = image.shape
    bar_width = int(torch.randint(params["bar_width_min"], params["bar_width_max"] + 1, (1,)).item())
    bar_gap = int(torch.randint(params["bar_gap_min"], params["bar_gap_max"] + 1, (1,)).item())
    offset = int(torch.randint(0, max(1, bar_gap), (1,)).item())
    strength = float(torch.empty(1).uniform_(params["bar_strength_min"], params["bar_strength_max"]).item())
    sign = -1.0 if torch.rand(1).item() < 0.5 else 1.0
    mask = make_bar_mask(width, height, bar_width, bar_gap, offset)
    noise = sign * strength * mask.unsqueeze(0)
    return (image + noise).clamp(0.0, 1.0)


def total_variation_loss(image: torch.Tensor) -> torch.Tensor:
    x_diff = torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]).mean()
    y_diff = torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]).mean()
    return x_diff + y_diff


class BarRemovalDataset(Dataset):
    def __init__(self, image_dir: str, resolution: int, params: Dict[str, float], max_samples: int) -> None:
        self.paths = list_images(image_dir)
        if max_samples:
            self.paths = self.paths[:max_samples]
        self.params = params
        self.transform = T.Compose(
            [
                T.Resize((resolution, resolution), interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img = Image.open(self.paths[idx]).convert("RGB")
        clean = self.transform(img)
        noisy = add_vertical_bars(clean, self.params)
        return {"noisy": noisy, "clean": clean}


def save_checkpoint(path: str, model: BarRemovalUNet, args: argparse.Namespace, epoch: int, loss: float) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "epoch": epoch,
            "loss": loss,
            "base_channels": args.base_channels,
            "resolution": args.resolution,
            "residual_scale": args.residual_scale,
        },
        path,
    )


def make_args() -> argparse.Namespace:
    argparser = argparse.ArgumentParser(description="Train a bar-removal model using synthetic vertical bars.")
    argparser.add_argument("--clean_dir", type=str, default="data/ultrasound_dataset/train/m1", help="Path to clean images.")
    argparser.add_argument("--output_dir", type=str, default="checkpoints/bar_removal", help="Where to save checkpoints.")
    argparser.add_argument("--resolution", type=int, default=512, help="Training resolution.")
    argparser.add_argument("--epochs", type=int, default=50, help="Number of epochs.")
    argparser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    argparser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    argparser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers.")
    argparser.add_argument("--base_channels", type=int, default=32, help="Base channel width.")
    argparser.add_argument("--residual_scale", type=float, default=1.0, help="Residual scale applied at output.")
    argparser.add_argument("--save_every", type=int, default=5, help="Save checkpoint every N epochs.")
    argparser.add_argument("--max_samples", type=int, default=0, help="Limit dataset size (0 = all).")
    argparser.add_argument("--seed", type=int, default=42, help="Random seed.")
    argparser.add_argument("--tv_weight", type=float, default=0.0, help="Total variation weight.")
    argparser.add_argument("--bar_prob", type=float, default=1.0, help="Probability to add bars.")
    argparser.add_argument("--bar_width_min", type=int, default=1, help="Minimum bar width.")
    argparser.add_argument("--bar_width_max", type=int, default=6, help="Maximum bar width.")
    argparser.add_argument("--bar_gap_min", type=int, default=8, help="Minimum gap between bars.")
    argparser.add_argument("--bar_gap_max", type=int, default=24, help="Maximum gap between bars.")
    argparser.add_argument("--bar_strength_min", type=float, default=0.08, help="Minimum bar strength.")
    argparser.add_argument("--bar_strength_max", type=float, default=0.3, help="Maximum bar strength.")
    return argparser.parse_args()


def main() -> None:
    args = make_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = {
        "bar_prob": args.bar_prob,
        "bar_width_min": args.bar_width_min,
        "bar_width_max": args.bar_width_max,
        "bar_gap_min": args.bar_gap_min,
        "bar_gap_max": args.bar_gap_max,
        "bar_strength_min": args.bar_strength_min,
        "bar_strength_max": args.bar_strength_max,
    }

    dataset = BarRemovalDataset(args.clean_dir, args.resolution, params, args.max_samples)
    if len(dataset) == 0:
        raise ValueError(f"No images found in {args.clean_dir}")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = BarRemovalUNet(base_channels=args.base_channels, residual_scale=args.residual_scale).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.L1Loss()

    best_loss = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        progress = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in progress:
            noisy = batch["noisy"].to(device, non_blocking=True)
            clean = batch["clean"].to(device, non_blocking=True)
            pred = model(noisy)
            loss = loss_fn(pred, clean)
            if args.tv_weight > 0:
                loss = loss + args.tv_weight * total_variation_loss(pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * noisy.size(0)
            progress.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(dataset)
        if best_loss is None or epoch_loss < best_loss:
            best_loss = epoch_loss
            save_checkpoint(os.path.join(args.output_dir, "bar_removal_best.pt"), model, args, epoch, epoch_loss)

        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_checkpoint(os.path.join(args.output_dir, f"bar_removal_epoch_{epoch}.pt"), model, args, epoch, epoch_loss)

        print(f"Epoch {epoch} average loss: {epoch_loss:.6f}")


if __name__ == "__main__":
    main()
