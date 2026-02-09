"""MAE pre-training script for the latent-space feature encoder.

Single-GPU training with gradient accumulation to reach effective batch 8192.
Per appendix_impl.tex:121-124:
- AdamW, lr 4e-3, betas (0.9, 0.95), weight_decay 0.05
- Effective batch 8192 via grad accum
- EMA decay 0.9995
- 192 epochs
- bf16 autocast
- Cosine LR schedule with 10-epoch warmup

Usage:
    python train_mae.py --latent_dir /path/to/precomputed_latents \
        --output_dir ./checkpoints/mae --base_width 256

    # Or with on-the-fly VAE encoding:
    python train_mae.py --imagenet_dir /path/to/imagenet \
        --output_dir ./checkpoints/mae --base_width 256
"""
from sklearn.externals.array_api_compat.torch import __name

import os
import copy
import math
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from models.latent_mae import LatentMAE
from data.imagenet import ImageNetLatentDataset


@torch.no_grad()
def update_ema(ema_model, model, decay):
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.data.mul_(decay).add_(p.data, alpha=1 - decay)


def cosine_lr(step, total_steps, warmup_steps, base_lr):
    """Cosine LR schedule with linear warmup."""
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def main():
    parser = argparse.ArgumentParser(description="MAE pre-training for latent feature encoder")
    parser.add_argument("--latent_dir", type=str, default=None,
                        help="Pre-computed latent directory (recommended)")
    parser.add_argument("--imagenet_dir", type=str, default=None,
                        help="ImageNet root for on-the-fly encoding")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/mae")
    parser.add_argument("--base_width", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=192)
    parser.add_argument("--micro_batch", type=int, default=64,
                        help="Per-GPU micro-batch size")
    parser.add_argument("--effective_batch", type=int, default=8192,
                        help="Effective batch size via gradient accumulation")
    parser.add_argument("--lr", type=float, default=4e-3)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--ema_decay", type=float, default=0.9995)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--save_every", type=int, default=10,
                        help="Save checkpoint every N epochs")
    args = parser.parse_args()
    print("Run or nah?")
    # --- DDP Setup ---
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        print(f"Initialized DDP: rank {rank}/{world_size}, local_rank {local_rank}")
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running in single-process mode on {device}")

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)

    # --- Model ---
    model = LatentMAE(
        in_channels=4,
        base_width=args.base_width,
        mask_ratio=0.5,
    ).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    ema_model = copy.deepcopy(model.module if world_size > 1 else model)
    ema_model.requires_grad_(False)
    ema_model.eval()

    # --- Data ---
    # Load VAE only if on-the-fly mode is needed
    vae = None
    if args.latent_dir is None and args.imagenet_dir is not None:
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
        vae.eval()
        vae.requires_grad_(False)

    dataset = ImageNetLatentDataset(
        latent_dir=args.latent_dir,
        imagenet_dir=args.imagenet_dir,
        split="train",
    )

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None

    loader = DataLoader(
        dataset,
        batch_size=args.micro_batch,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    # --- Schedule ---
    # Effective batch = micro_batch * world_size * accum_steps
    global_batch_per_step = args.micro_batch * world_size
    accum_steps = args.effective_batch // global_batch_per_step
    if accum_steps < 1:
        accum_steps = 1
        if rank == 0:
            print(f"Warning: Effective batch {args.effective_batch} < actual batch {global_batch_per_step}. Using accum_steps=1.")

    steps_per_epoch = len(loader) // accum_steps
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch
    vae_scaling_factor = 0.18215

    if rank == 0:
        print(f"Dataset size: {len(dataset)}")
        print(f"Micro-batch per GPU: {args.micro_batch}, World size: {world_size}")
        print(f"Global batch per fwd: {global_batch_per_step}, Accum steps: {accum_steps}")
        print(f"Effective batch: {global_batch_per_step * accum_steps}")
        print(f"Steps per epoch: {steps_per_epoch}, Total steps: {total_steps}")


    # --- Resume ---
    start_epoch = 0
    global_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        ema_model.load_state_dict(ckpt["ema_model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt.get("global_step", start_epoch * steps_per_epoch)
        print(f"Resumed from epoch {start_epoch}, step {global_step}")

    # --- Training loop ---
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        optimizer.zero_grad()

        if rank == 0:
            pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        else:
            pbar = loader

        for batch_idx, (data, _labels) in enumerate(pbar):
            data = data.to(device, non_blocking=True)

            # On-the-fly VAE encoding if needed
            if dataset.mode == "onthefly" and vae is not None:
                with torch.no_grad():
                    data = vae.encode(data).latent_dist.sample() * vae_scaling_factor

            # Use no_sync for accumulation steps except the last one
            # to avoid redundant gradient all-reduce
            is_accum_step = ((batch_idx + 1) % accum_steps == 0) or ((batch_idx + 1) == len(loader))
            
            context = model.no_sync() if (world_size > 1 and not is_accum_step) else torch.nullcontext()

            with context:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    loss, _recon = model(data)
                    loss = loss / accum_steps

                scaler.scale(loss).backward()

            # Accumulation step
            if is_accum_step:
                # Update LR
                lr = cosine_lr(global_step, total_steps, warmup_steps, args.lr)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # EMA update
                # Unwrap DDP model for EMA source if needed
                source_model = model.module if world_size > 1 else model
                update_ema(ema_model, source_model, args.ema_decay)

                global_step += 1
                epoch_loss += loss.item() * accum_steps
                num_batches += 1

                if rank == 0 and num_batches % 50 == 0:
                    avg = epoch_loss / num_batches
                    pbar.set_postfix(loss=f"{avg:.4e}", lr=f"{lr:.2e}", step=global_step)

        if rank == 0 and num_batches > 0:
            avg_loss = epoch_loss / num_batches
            print(f"Epoch {epoch}: avg_loss={avg_loss:.4e}")

        # Save checkpoint (Rank 0 only)
        if rank == 0 and ((epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1):
            # Unwrap for saving
            raw_model = model.module if world_size > 1 else model
            ckpt = {
                "epoch": epoch,
                "global_step": global_step,
                "model": raw_model.state_dict(),
                "ema_model": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": {
                    "base_width": args.base_width,
                    "epochs": args.epochs,
                    "lr": args.lr,
                    "effective_batch": args.effective_batch,
                    "ema_decay": args.ema_decay,
                },
            }
            path = os.path.join(args.output_dir, f"mae_epoch{epoch:03d}.pt")
            torch.save(ckpt, path)
            print(f"Saved checkpoint to {path}")

    # Save final (Rank 0 only)
    if rank == 0:
        raw_model = model.module if world_size > 1 else model
        final_path = os.path.join(args.output_dir, "mae_final.pt")
        ckpt = {
            "epoch": args.epochs - 1,
            "global_step": global_step,
            "model": raw_model.state_dict(),
            "ema_model": ema_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": {
                "base_width": args.base_width,
                "epochs": args.epochs,
                "lr": args.lr,
                "effective_batch": args.effective_batch,
                "ema_decay": args.ema_decay,
            },
        }
        torch.save(ckpt, final_path)
        print(f"Training complete. Final checkpoint: {final_path}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()