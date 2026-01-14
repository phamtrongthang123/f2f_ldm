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


def normalize_pair(a: np.ndarray, b: np.ndarray, percentile: float = 99.0) -> tuple[np.ndarray, np.ndarray]:
    vmax = max(np.percentile(a, percentile), np.percentile(b, percentile), 1e-6)
    a_scaled = np.clip(a / vmax, 0.0, 1.0)
    b_scaled = np.clip(b / vmax, 0.0, 1.0)
    return a_scaled, b_scaled


def column_means(image: np.ndarray) -> np.ndarray:
    return image.mean(axis=0)


def histogram_overlay(
    a: np.ndarray,
    b: np.ndarray,
    label_a: str,
    label_b: str,
    title: str,
    bins: int = 128,
    size: tuple[int, int] = (640, 360),
) -> Image.Image:
    hist_a, edges = np.histogram(a, bins=bins, range=(0.0, 1.0))
    hist_b, _ = np.histogram(b, bins=bins, range=(0.0, 1.0))
    max_count = max(hist_a.max(), hist_b.max(), 1)

    width, height = size
    margin_x = 48
    margin_y = 40
    plot_w = width - 2 * margin_x
    plot_h = height - 2 * margin_y

    image = Image.new("RGB", size, (20, 20, 20))
    draw = ImageDraw.Draw(image)
    draw.rectangle((margin_x, margin_y, margin_x + plot_w, margin_y + plot_h), outline=(80, 80, 80))

    bin_centers = (edges[:-1] + edges[1:]) * 0.5
    xs = margin_x + (bin_centers * plot_w)
    ys_a = margin_y + plot_h * (1.0 - hist_a / max_count)
    ys_b = margin_y + plot_h * (1.0 - hist_b / max_count)

    points_a = list(zip(xs.tolist(), ys_a.tolist()))
    points_b = list(zip(xs.tolist(), ys_b.tolist()))
    draw.line(points_a, fill=(230, 120, 60), width=2)
    draw.line(points_b, fill=(100, 180, 220), width=2)

    draw.text((margin_x, 8), title, fill=(230, 230, 230))
    legend_y = margin_y + plot_h + 8
    draw.text((margin_x, legend_y), label_a, fill=(230, 120, 60))
    draw.text((margin_x + 160, legend_y), label_b, fill=(100, 180, 220))
    return image


def build_grid(panels: list[Image.Image], columns: int, margin: int = 12) -> Image.Image:
    rows = ceil(len(panels) / columns)
    width, height = panels[0].size
    grid = Image.new(
        "RGB",
        (columns * width + (columns + 1) * margin, rows * height + (rows + 1) * margin),
        (15, 15, 15),
    )
    for idx, panel in enumerate(panels):
        row = idx // columns
        col = idx % columns
        x = margin + col * (width + margin)
        y = margin + row * (height + margin)
        grid.paste(panel, (x, y))
    return grid


def compute_metrics(
    paths: list[Path],
    blur_ksize: int,
    fx_low: float,
    fx_high: float,
    fy_cutoff: float,
    gauss_ksize: int,
    gauss_sigma: float,
) -> dict[str, np.ndarray]:
    reference = load_grayscale(paths[0])
    temporal = temporal_mean(paths)
    temporal_residual = temporal_abs_residual_mean(paths, temporal)
    temporal_blur = horizontal_blur(temporal, blur_ksize)
    temporal_bandpass = np.abs(temporal - temporal_blur)
    sobel_edges = np.abs(sobel_x(reference))
    gauss_blur = gaussian_blur(reference, gauss_ksize, gauss_sigma)
    sobel_highpass = np.abs(sobel_x(reference - gauss_blur))
    fft_filtered = np.abs(fft_bandpass(reference, fx_low, fx_high, fy_cutoff))
    return {
        "original": reference,
        "temporal_mean": temporal,
        "temporal_residual_mean": temporal_residual,
        "temporal_bandpass": temporal_bandpass,
        "sobel_x_edges": sobel_edges,
        "sobel_highpass": sobel_highpass,
        "fft_bandpass": fft_filtered,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare histograms between two datasets.")
    parser.add_argument("--dir-a", type=Path, required=True)
    parser.add_argument("--dir-b", type=Path, required=True)
    parser.add_argument("--label-a", type=str, default="set_a")
    parser.add_argument("--label-b", type=str, default="set_b")
    parser.add_argument("--num-frames", type=int, default=20)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--use-all", action="store_true", help="Use all frames in each directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/bar_histograms"))
    parser.add_argument("--blur-ksize", type=int, default=51)
    parser.add_argument("--fx-low", type=float, default=0.06)
    parser.add_argument("--fx-high", type=float, default=0.25)
    parser.add_argument("--fy-cutoff", type=float, default=0.02)
    parser.add_argument("--gauss-ksize", type=int, default=9)
    parser.add_argument("--gauss-sigma", type=float, default=2.0)
    args = parser.parse_args()

    paths_a = sorted(args.dir_a.glob("*.png"))
    paths_b = sorted(args.dir_b.glob("*.png"))
    if not paths_a or not paths_b:
        raise FileNotFoundError("One of the input directories has no png files.")

    if args.use_all or args.num_frames <= 0:
        select_a = paths_a
        select_b = paths_b
    else:
        select_a = paths_a[0 : args.num_frames * args.frame_step : args.frame_step]
        select_b = paths_b[0 : args.num_frames * args.frame_step : args.frame_step]

    metrics_a = compute_metrics(
        select_a,
        args.blur_ksize,
        args.fx_low,
        args.fx_high,
        args.fy_cutoff,
        args.gauss_ksize,
        args.gauss_sigma,
    )
    metrics_b = compute_metrics(
        select_b,
        args.blur_ksize,
        args.fx_low,
        args.fx_high,
        args.fy_cutoff,
        args.gauss_ksize,
        args.gauss_sigma,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    panels = []
    for key in metrics_a.keys():
        a_scaled, b_scaled = normalize_pair(metrics_a[key], metrics_b[key])
        cols_a = column_means(a_scaled)
        cols_b = column_means(b_scaled)
        panel = histogram_overlay(cols_a, cols_b, args.label_a, args.label_b, f"hist_colmean_{key}")
        panel.save(args.output_dir / f"hist_{key}.png")
        panels.append(panel)

    grid = build_grid(panels, columns=2)
    grid.save(args.output_dir / "hist_grid.png")


if __name__ == "__main__":
    main()
