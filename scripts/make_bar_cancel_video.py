import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def load_grayscale(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    return np.asarray(image, dtype=np.float32) / 255.0


def to_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)


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


def horizontal_blur(image: np.ndarray, ksize: int) -> np.ndarray:
    if ksize % 2 == 0:
        raise ValueError("ksize must be odd")
    tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    kernel = torch.ones(1, 1, 1, ksize, dtype=tensor.dtype) / float(ksize)
    pad = ksize // 2
    tensor = F.pad(tensor, (pad, pad, 0, 0), mode="reflect")
    blurred = F.conv2d(tensor, kernel)
    return blurred[0, 0].numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create side-by-side bar-cancel frames for a video.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/ultrasound_dataset/train/m3"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/m3_bar_cancel_frames"))
    parser.add_argument("--video-path", type=Path, default=None, help="Optional mp4 path to write.")
    parser.add_argument("--fps", type=float, default=None, help="Frames per second for the output video.")
    parser.add_argument("--source-video", type=Path, default=None, help="Optional source video to infer fps.")
    parser.add_argument("--num-frames", type=int, default=0, help="0 means use all frames.")
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--blur-ksize", type=int, default=51)
    parser.add_argument("--cancel-alpha", type=float, default=1.0)
    parser.add_argument("--mode", type=str, choices=["global", "ema"], default="global")
    parser.add_argument("--ema-alpha", type=float, default=0.02)
    parser.add_argument("--ema-bias-correct", action="store_true")
    parser.add_argument("--no-save", action="store_true", help="Skip writing per-frame PNGs.")
    args = parser.parse_args()

    paths = sorted(args.input_dir.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No png files found in {args.input_dir}")

    if args.num_frames > 0:
        selected = paths[0 : args.num_frames * args.frame_step : args.frame_step]
    else:
        selected = paths[0:: args.frame_step]

    mean_image = None
    bar_component = None
    if args.mode == "global":
        mean_image = temporal_mean(selected)
        blurred_mean = horizontal_blur(mean_image, args.blur_ksize)
        bar_component = mean_image - blurred_mean

    if not args.no_save:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    digits = max(4, len(str(len(selected))))
    writer = None
    if args.video_path is not None:
        import cv2

        if args.fps is None:
            if args.source_video is not None:
                cap = cv2.VideoCapture(str(args.source_video))
                if not cap.isOpened():
                    raise RuntimeError(f"Failed to open source video {args.source_video}")
                args.fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
            if not args.fps:
                args.fps = 30.0

        sample = load_grayscale(selected[0])
        height, width = sample.shape
        out_size = (int(width * 2), int(height))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(args.video_path), fourcc, args.fps, out_size, True)
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {args.video_path}")

    ema = None
    ema_count = 0
    start = time.perf_counter()
    for idx, path in enumerate(selected, start=1):
        frame = load_grayscale(path)
        if args.mode == "ema":
            if ema is None:
                ema = frame.copy()
                ema_count = 1
            else:
                ema = (1.0 - args.ema_alpha) * ema + args.ema_alpha * frame
                ema_count += 1
            if args.ema_bias_correct:
                correction = 1.0 - (1.0 - args.ema_alpha) ** ema_count
                mean_image = ema / max(correction, 1e-6)
            else:
                mean_image = ema
            blurred_mean = horizontal_blur(mean_image, args.blur_ksize)
            bar_component = mean_image - blurred_mean

        canceled = np.clip(frame - args.cancel_alpha * bar_component, 0.0, 1.0)
        left = to_uint8(frame)
        right = to_uint8(canceled)
        composite = np.concatenate([left, right], axis=1)
        if not args.no_save:
            out_path = args.output_dir / f"frame_{idx:0{digits}d}.png"
            Image.fromarray(composite).save(out_path)
        if writer is not None:
            import cv2

            composite_bgr = cv2.cvtColor(composite, cv2.COLOR_GRAY2BGR)
            writer.write(composite_bgr)

    if writer is not None:
        writer.release()

    elapsed = time.perf_counter() - start
    if elapsed > 0:
        fps = len(selected) / elapsed
        print(f"Processed {len(selected)} frames in {elapsed:.2f}s -> {fps:.2f} fps")


if __name__ == "__main__":
    main()
