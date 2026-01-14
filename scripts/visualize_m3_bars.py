import argparse
from math import ceil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


def load_grayscale(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return array


def normalize_for_display(image: np.ndarray, percentile: float = 1.0) -> np.ndarray:
    lo = np.percentile(image, percentile)
    hi = np.percentile(image, 100.0 - percentile)
    if hi <= lo:
        return np.clip(image, 0.0, 1.0)
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0)


def to_pil(image: np.ndarray) -> Image.Image:
    scaled = (normalize_for_display(image) * 255.0).astype(np.uint8)
    return Image.fromarray(scaled)


def temporal_mean(paths: list[Path]) -> np.ndarray:
    total = None
    count = 0
    for path in paths:
        frame = load_grayscale(path)
        total = frame if total is None else total + frame
        count += 1
    if total is None:
        raise ValueError("No frames provided for temporal mean.")
    return total / float(count)


def temporal_abs_residual_mean(paths: list[Path], mean_image: np.ndarray) -> np.ndarray:
    total = np.zeros_like(mean_image)
    count = 0
    for path in paths:
        frame = load_grayscale(path)
        total += np.abs(frame - mean_image)
        count += 1
    if count == 0:
        raise ValueError("No frames provided for temporal residual mean.")
    return total / float(count)


def horizontal_blur(image: np.ndarray, ksize: int) -> np.ndarray:
    if ksize % 2 == 0:
        raise ValueError("ksize must be odd")
    tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    kernel = torch.ones(1, 1, 1, ksize, dtype=tensor.dtype) / float(ksize)
    pad = ksize // 2
    tensor = F.pad(tensor, (pad, pad, 0, 0), mode="reflect")
    blurred = F.conv2d(tensor, kernel)
    return blurred[0, 0].numpy()


def sobel_x(image: np.ndarray) -> np.ndarray:
    kernel = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    kernel = kernel.view(1, 1, 3, 3)
    tensor = F.pad(tensor, (1, 1, 1, 1), mode="reflect")
    edges = F.conv2d(tensor, kernel)
    return edges[0, 0].numpy()


def gaussian_kernel(ksize: int, sigma: float) -> torch.Tensor:
    if ksize % 2 == 0:
        raise ValueError("ksize must be odd")
    radius = ksize // 2
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel_1d = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d


def gaussian_blur(image: np.ndarray, ksize: int, sigma: float) -> np.ndarray:
    kernel = gaussian_kernel(ksize, sigma)
    tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    kernel = kernel.view(1, 1, ksize, ksize)
    pad = ksize // 2
    tensor = F.pad(tensor, (pad, pad, pad, pad), mode="reflect")
    blurred = F.conv2d(tensor, kernel)
    return blurred[0, 0].numpy()


def fft_bandpass(image: np.ndarray, fx_low: float, fx_high: float, fy_cutoff: float) -> np.ndarray:
    height, width = image.shape
    fft = np.fft.fft2(image)
    fy = np.fft.fftfreq(height)
    fx = np.fft.fftfreq(width)
    fy_grid, fx_grid = np.meshgrid(fy, fx, indexing="ij")
    mask = (np.abs(fy_grid) < fy_cutoff) & (np.abs(fx_grid) > fx_low) & (np.abs(fx_grid) < fx_high)
    filtered = fft * mask
    return np.real(np.fft.ifft2(filtered))


def make_panel(image: np.ndarray, label: str) -> Image.Image:
    panel = to_pil(image).convert("RGB")
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, panel.width, 18), fill=(0, 0, 0))
    draw.text((4, 2), label, fill=(255, 255, 255))
    return panel


def build_grid(panels: list[Image.Image], columns: int, margin: int = 12) -> Image.Image:
    rows = ceil(len(panels) / columns)
    width, height = panels[0].size
    grid = Image.new("RGB", (columns * width + (columns + 1) * margin, rows * height + (rows + 1) * margin), (20, 20, 20))
    for idx, panel in enumerate(panels):
        row = idx // columns
        col = idx % columns
        x = margin + col * (width + margin)
        y = margin + row * (height + margin)
        grid.paste(panel, (x, y))
    return grid


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize vertical bar enhancement filters for m3 frames.")
    parser.add_argument("--m3-dir", type=Path, default=Path("data/ultrasound_dataset/train/m3"))
    parser.add_argument("--num-frames", type=int, default=20)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--use-all", action="store_true", help="Use all frames in the directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/m3_bar_viz"))
    parser.add_argument("--blur-ksize", type=int, default=51)
    parser.add_argument("--fx-low", type=float, default=0.06)
    parser.add_argument("--fx-high", type=float, default=0.25)
    parser.add_argument("--fy-cutoff", type=float, default=0.02)
    parser.add_argument("--gauss-ksize", type=int, default=9)
    parser.add_argument("--gauss-sigma", type=float, default=2.0)
    parser.add_argument("--cancel-alpha", type=float, default=1.0)
    args = parser.parse_args()

    frame_paths = sorted(args.m3_dir.glob("*.png"))
    if not frame_paths:
        raise FileNotFoundError(f"No m3 frames found in {args.m3_dir}")

    if args.use_all or args.num_frames <= 0:
        selected = frame_paths
    else:
        selected = frame_paths[0 : args.num_frames * args.frame_step : args.frame_step]
    reference = load_grayscale(selected[0])
    temporal = temporal_mean(selected)
    temporal_residual = temporal_abs_residual_mean(selected, temporal)
    temporal_blur = horizontal_blur(temporal, args.blur_ksize)
    bar_component = temporal - temporal_blur
    temporal_bandpass = np.abs(bar_component)
    reference_residual = np.abs(reference - temporal)
    sobel_edges = np.abs(sobel_x(reference))
    gauss_blur = gaussian_blur(reference, args.gauss_ksize, args.gauss_sigma)
    sobel_highpass = np.abs(sobel_x(reference - gauss_blur))
    fft_filtered = np.abs(fft_bandpass(reference, args.fx_low, args.fx_high, args.fy_cutoff))
    bar_canceled = reference - args.cancel_alpha * bar_component

    args.output_dir.mkdir(parents=True, exist_ok=True)

    panels = [
        ("original_m3", reference),
        ("temporal_mean", temporal),
        ("temporal_residual_mean", temporal_residual),
        ("frame_minus_temporal", reference_residual),
        ("temporal_bandpass", temporal_bandpass),
        ("sobel_x_edges", sobel_edges),
        ("sobel_highpass", sobel_highpass),
        ("fft_bandpass", fft_filtered),
        ("bar_canceled", bar_canceled),
    ]

    for label, image in panels:
        out_path = args.output_dir / f"{label}.png"
        to_pil(image).save(out_path)

    labeled_panels = [make_panel(image, label) for label, image in panels]
    grid = build_grid(labeled_panels, columns=2)
    grid.save(args.output_dir / "m3_bar_filters_grid.png")


if __name__ == "__main__":
    main()
