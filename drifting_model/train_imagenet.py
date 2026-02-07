"""Main ImageNet training script for drifting models.

Implements the training loop from Appendix B.6:
1. Sample N_c class labels
2. For each class, sample CFG alpha ~ p(alpha) ∝ alpha^{-3}
3. Generate samples via generator
4. Extract features directly from latents (latent-MAE encoder)
5. Compute drifting loss
6. Backprop, update, EMA

Usage:
    # Single-GPU:
    python train_imagenet.py --config configs/ablation_default.yaml

    # Multi-GPU:
    torchrun --nproc_per_node=8 train_imagenet.py --config configs/ablation_default.yaml
"""

import os
import copy
import math
import argparse
from contextlib import nullcontext
import yaml
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from models.dit import DiTGenerator, dit_b2, dit_l2
from models.feature_encoder import LatentMAEFeatureExtractor
from drifting_loss import compute_drifting_loss
from data.sample_queue import SampleQueue
from data.imagenet import ImageNetLatentDataset


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def sample_cfg_alpha(batch_size, alpha_range, power, device):
    """Sample CFG alpha from p(alpha) ∝ alpha^power on [alpha_min, alpha_max].

    Uses inverse CDF sampling for the power-law distribution.
    """
    alpha_min, alpha_max = alpha_range
    p = power + 1  # integrate alpha^power → alpha^(power+1)/(power+1)

    if abs(p) < 1e-8:
        # Special case: p(alpha) ∝ 1/alpha → log distribution
        u = torch.rand(batch_size, device=device)
        alpha = alpha_min * (alpha_max / alpha_min) ** u
    else:
        u = torch.rand(batch_size, device=device)
        alpha_min_p = alpha_min ** p
        alpha_max_p = alpha_max ** p
        alpha = (alpha_min_p + u * (alpha_max_p - alpha_min_p)) ** (1.0 / p)

    return alpha


def compute_cfg_weight(alpha, n_neg, n_unc):
    """Compute CFG weight w for unconditional samples.

    From the paper: alpha = ((N_neg-1) + N_unc * w) / (N_neg - 1)
    So: w = (alpha - 1) * (N_neg - 1) / N_unc
    """
    return (alpha - 1.0) * (n_neg - 1) / n_unc


@torch.no_grad()
def update_ema(ema_model, model, decay):
    """Update EMA model parameters."""
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.data.mul_(decay).add_(p.data, alpha=1 - decay)


def get_lr(step, warmup_steps, base_lr):
    """Linear warmup then constant learning rate."""
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    return base_lr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ablation_default.yaml")
    parser.add_argument("--imagenet_dir", type=str, default=None,
                        help="ImageNet root (for on-the-fly encoding)")
    parser.add_argument("--latent_dir", type=str, default=None,
                        help="Pre-computed latent directory")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--bf16", action="store_true",
                        help="Enable bf16 autocast for forward + loss")
    args = parser.parse_args()

    # --- DDP / single-GPU setup ---
    distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1

    if distributed:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    cfg = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    # ---------- Models ----------
    # Generator
    gen_cfg = cfg["generator"]
    if gen_cfg["arch"] == "DiT-B/2":
        generator = dit_b2(
            input_size=gen_cfg["input_size"],
            input_channels=gen_cfg["input_channels"],
            num_classes=gen_cfg["num_classes"],
            num_register_tokens=gen_cfg["num_register_tokens"],
            num_style_tokens=gen_cfg["num_style_tokens"],
            style_codebook_size=gen_cfg["style_codebook_size"],
        )
    elif gen_cfg["arch"] == "DiT-L/2":
        generator = dit_l2(
            input_size=gen_cfg["input_size"],
            input_channels=gen_cfg["input_channels"],
            num_classes=gen_cfg["num_classes"],
            num_register_tokens=gen_cfg["num_register_tokens"],
            num_style_tokens=gen_cfg["num_style_tokens"],
            style_codebook_size=gen_cfg["style_codebook_size"],
        )
    else:
        raise ValueError(f"Unknown architecture: {gen_cfg['arch']}")

    generator = generator.to(device)

    if distributed:
        generator_ddp = DDP(generator, device_ids=[local_rank])
    else:
        generator_ddp = generator

    # EMA model
    gen_module = generator_ddp.module if distributed else generator_ddp
    ema_generator = copy.deepcopy(gen_module)
    ema_generator.requires_grad_(False)
    ema_generator.eval()

    # Feature encoder (frozen, operates on latents directly)
    feat_cfg = cfg["feature_encoder"]
    feature_encoder = LatentMAEFeatureExtractor(
        checkpoint_path=feat_cfg.get("checkpoint_path"),
        base_width=feat_cfg.get("base_width", 256),
    ).to(device)
    feature_encoder.eval()

    # VAE — only needed for on-the-fly encoding (not for feature extraction)
    vae = None
    vae_scaling_factor = 0.18215
    dataset_obj = ImageNetLatentDataset(
        latent_dir=args.latent_dir,
        imagenet_dir=args.imagenet_dir,
        split="train",
    )
    if dataset_obj.mode == "onthefly":
        from diffusers import AutoencoderKL
        vae_cfg = cfg["vae"]
        vae = AutoencoderKL.from_pretrained(vae_cfg["model_id"]).to(device)
        vae.eval()
        vae.requires_grad_(False)

    # ---------- Optimizer ----------
    opt_cfg = cfg["optimizer"]
    optimizer = torch.optim.AdamW(
        generator_ddp.parameters(),
        lr=opt_cfg["lr"],
        betas=tuple(opt_cfg["betas"]),
        weight_decay=opt_cfg["weight_decay"],
    )

    # ---------- Data ----------
    if distributed:
        sampler = DistributedSampler(dataset_obj, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        sampler = None

    loader = DataLoader(
        dataset_obj,
        batch_size=cfg["queue"]["push_per_step"],
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # Sample queue
    q_cfg = cfg["queue"]
    sample_queue = SampleQueue(
        num_classes=gen_cfg["num_classes"],
        class_queue_size=q_cfg["class_queue_size"],
        uncond_queue_size=q_cfg["uncond_queue_size"],
    )

    # ---------- Training config ----------
    drift_cfg = cfg["drifting"]
    train_cfg = cfg["training"]
    n_classes = drift_cfg["n_classes"]
    n_pos = drift_cfg["n_pos"]
    n_neg = drift_cfg["n_neg"]
    n_uncond = drift_cfg["n_uncond"]
    temperatures = drift_cfg["temperatures"]
    alpha_range = drift_cfg["cfg_alpha_range"]
    alpha_power = drift_cfg["cfg_alpha_power"]
    total_steps = train_cfg["total_steps"]
    ema_decay = train_cfg["ema_decay"]
    grad_clip = opt_cfg["grad_clip"]
    warmup_steps = opt_cfg["warmup_steps"]
    base_lr = opt_cfg["lr"]

    # Resume
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        gen_module.load_state_dict(ckpt["generator"])
        ema_generator.load_state_dict(ckpt["ema_generator"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"] + 1
        if rank == 0:
            print(f"Resumed from step {start_step}")

    # ---------- Training loop ----------
    data_iter = iter(loader)
    epoch = 0
    amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if args.bf16 else nullcontext()

    if rank == 0:
        pbar = tqdm(range(start_step, total_steps), desc="Training")
    else:
        pbar = range(start_step, total_steps)

    for step in pbar:
        # Reload data iterator if exhausted
        try:
            batch_data, batch_labels = next(data_iter)
        except StopIteration:
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)
            data_iter = iter(loader)
            batch_data, batch_labels = next(data_iter)

        # Push new data to queue
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)

        # If data is images (on-the-fly mode), encode to latent
        if dataset_obj.mode == "onthefly" and vae is not None:
            with torch.no_grad():
                batch_data = vae.encode(batch_data).latent_dist.sample() * vae_scaling_factor

        sample_queue.push(batch_data, batch_labels)

        # Wait until queue has enough data
        if not sample_queue.is_ready(min_class_samples=n_pos, min_uncond_samples=n_uncond):
            continue

        # --- Actual training step ---
        # 1. Sample class labels
        available = sample_queue.available_classes(n_pos)
        if len(available) < n_classes:
            n_classes_step = len(available)
        else:
            n_classes_step = n_classes

        class_indices = torch.tensor(
            available, device=device
        )[torch.randperm(len(available), device=device)[:n_classes_step]]

        # 2. Sample CFG alpha per class
        alphas = sample_cfg_alpha(n_classes_step, alpha_range, alpha_power, device)

        # 3. Process each class with gradient accumulation
        generator_ddp.train()
        optimizer.zero_grad()

        num_classes_processed = 0
        loss_accum = 0.0

        for i, (cls_label, alpha) in enumerate(zip(class_indices, alphas)):
            cls_label_int = cls_label.item()
            alpha_val = alpha.item()

            # Sample positives and unconditional from queue
            pos_latent = sample_queue.sample_positives(cls_label_int, n_pos)
            if pos_latent is None:
                continue
            unc_latent = sample_queue.sample_unconditional(n_uncond)
            if unc_latent is None:
                continue

            pos_latent = pos_latent.to(device)
            unc_latent = unc_latent.to(device)

            # Generate noise and style indices
            noise = torch.randn(n_neg, gen_cfg["input_channels"],
                                gen_cfg["input_size"], gen_cfg["input_size"],
                                device=device)
            labels_batch = cls_label.expand(n_neg)
            alpha_batch = alpha.expand(n_neg)
            style_indices = torch.randint(
                0, gen_cfg["style_codebook_size"],
                (n_neg, gen_cfg.get("num_style_tokens", 32)),
                device=device,
            )

            # Use no_sync for all but the last class to skip redundant all-reduce
            is_last = (i == len(class_indices) - 1)
            if distributed:
                sync_ctx = nullcontext() if is_last else generator_ddp.no_sync()
            else:
                sync_ctx = nullcontext()

            with sync_ctx:
                with amp_ctx:
                    gen_latent = generator_ddp(noise, labels_batch, alpha_batch, style_indices)

                    # CFG weight for unconditional samples
                    w = compute_cfg_weight(alpha_val, n_neg, n_uncond)
                    cfg_w = torch.ones(n_uncond, device=device) * w

                    # Extract features directly from latents (no VAE decode needed)
                    gen_feats = feature_encoder(gen_latent)
                    with torch.no_grad():
                        pos_feats = feature_encoder(pos_latent)
                        unc_feats = feature_encoder(unc_latent)
                    neg_feats = gen_feats  # reuse generated features as negatives

                    # Flatten latents for vanilla drifting loss
                    gen_lat_flat = gen_latent.reshape(gen_latent.shape[0], -1)
                    pos_lat_flat = pos_latent.reshape(pos_latent.shape[0], -1)
                    neg_lat_flat = gen_lat_flat
                    unc_lat_flat = unc_latent.reshape(unc_latent.shape[0], -1)

                    # Compute drifting loss for this class
                    loss = compute_drifting_loss(
                        gen_features=gen_feats,
                        pos_features=pos_feats,
                        neg_features=neg_feats,
                        uncond_features=unc_feats,
                        temperatures=temperatures,
                        cfg_weights=cfg_w,
                        gen_latent=gen_lat_flat,
                        pos_latent=pos_lat_flat,
                        neg_latent=neg_lat_flat,
                        uncond_latent=unc_lat_flat,
                    )

                # Gradient accumulation: backward per class, divide by total classes
                (loss / n_classes_step).backward()

            loss_accum += loss.item()
            num_classes_processed += 1

        if num_classes_processed == 0:
            continue

        # Gradient clipping and optimizer step
        torch.nn.utils.clip_grad_norm_(generator_ddp.parameters(), grad_clip)

        # Learning rate warmup
        lr = get_lr(step, warmup_steps, base_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.step()

        # Update EMA
        update_ema(ema_generator, gen_module, ema_decay)

        # Logging
        avg_loss = loss_accum / max(num_classes_processed, 1)
        if rank == 0 and step % train_cfg["log_every"] == 0:
            pbar.set_postfix(loss=f"{avg_loss:.4e}", lr=f"{lr:.2e}")

        # Checkpointing
        if rank == 0 and (step + 1) % train_cfg["checkpoint_every"] == 0:
            ckpt = {
                "step": step,
                "generator": gen_module.state_dict(),
                "ema_generator": ema_generator.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": cfg,
            }
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_{step + 1}.pt")
            torch.save(ckpt, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

    # Final save
    if rank == 0:
        ckpt = {
            "step": total_steps - 1,
            "generator": gen_module.state_dict(),
            "ema_generator": ema_generator.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
        }
        torch.save(ckpt, os.path.join(args.output_dir, "checkpoint_final.pt"))
        print("Training complete.")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
