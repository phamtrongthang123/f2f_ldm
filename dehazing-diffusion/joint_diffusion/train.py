"""Training script for score-based generative models (PyTorch version)
Author(s): Tristan Stevens
Ported to PyTorch: Jan 2026
"""
import argparse
from pathlib import Path

import matplotlib
import numpy as np
import torch
import tqdm

matplotlib.use("Agg")

# Optional wandb
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("wandb not available, logging disabled")

from datasets import get_dataset
from generators.models import get_model, create_optimizer, create_lr_scheduler
from utils.callbacks import CallbackList, EvalDataset, Monitor, LossLogger
from utils.checkpoints import ModelCheckpoint, EMAHelper
from utils.gpu_config import set_gpu_usage, get_device
from utils.utils import load_config_from_yaml, set_random_seed, add_args_to_config

# Optional git info
try:
    from utils.git_info import get_git_summary
except ImportError:
    get_git_summary = lambda: {}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train score-based generative model")
    parser.add_argument(
        "-c", "--config",
        default="configs/training/score_zea_tissue.yaml",
        type=str,
        help="Path to config file",
    )
    parser.add_argument(
        "--data_root",
        default=None,
        type=str,
        help="Override data root path",
    )
    parser.add_argument(
        "--epochs",
        default=None,
        type=int,
        help="Override number of epochs",
    )
    parser.add_argument(
        "--batch_size",
        default=None,
        type=int,
        help="Override batch size",
    )
    parser.add_argument(
        "--lr",
        default=None,
        type=float,
        help="Override learning rate",
    )
    parser.add_argument(
        "--device",
        default=None,
        type=str,
        help="Device to use (cuda:0, cpu, etc.)",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable wandb logging",
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="Path to checkpoint to resume from",
    )
    return parser.parse_args()


def train_epoch(model, dataloader, optimizer, device, epoch, config, ema=None, callbacks=None):
    """Train for one epoch.

    Args:
        model: PyTorch model (ScoreNet)
        dataloader: Training DataLoader
        optimizer: PyTorch optimizer
        device: torch.device
        epoch: Current epoch number
        config: Config object
        ema: Optional EMAHelper for per-batch updates
        callbacks: Optional CallbackList

    Returns:
        Average loss for the epoch
    """
    model.train()
    losses = []

    pbar = tqdm.tqdm(
        dataloader,
        desc=f"Epoch {epoch + 1}/{config.epochs}",
        leave=True,
    )

    for batch_idx, batch in enumerate(pbar):
        # Handle tuple batches (paired data)
        if isinstance(batch, (tuple, list)):
            batch = batch[0]

        batch = batch.to(device)

        # Forward pass
        optimizer.zero_grad()
        loss = model.score_loss(batch)

        # Backward pass
        loss.backward()

        # Gradient clipping (optional)
        if config.get("grad_clip"):
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_clip
            )

        optimizer.step()

        # EMA update (per-batch)
        if ema is not None:
            ema.update(model)

        # Track loss
        loss_val = loss.item()
        losses.append(loss_val)

        # Update progress bar
        pbar.set_postfix(loss=f"{loss_val:.4f}")

        # Callback
        if callbacks:
            callbacks.on_batch_end(batch_idx, {"loss": loss_val})

    return np.mean(losses)


def train(config, args):
    """Main training function.
    
    Args:
        config: Configuration dict/object
        args: Command line arguments
    """
    # Set device
    device_str = set_gpu_usage(config.get("device"))
    device = torch.device(device_str if device_str else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set random seed
    set_random_seed(config.get("seed"))
    
    # Load dataset
    train_loader, val_loader = get_dataset(config)
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Create model
    model = get_model(config, plot_summary=True, training=True)
    model = model.to(device)
    
    # Create optimizer
    optimizer = create_optimizer(model, config)
    
    # Create LR scheduler
    scheduler = create_lr_scheduler(optimizer, config)
    
    # EMA
    ema = None
    if config.get("ema"):
        ema_decay = config.ema if isinstance(config.ema, float) else 0.9999
        ema = EMAHelper(model, decay=ema_decay, device=device)
        print(f"Using EMA with decay={ema_decay}")
    
    # Checkpointing
    ckpt_manager = ModelCheckpoint(
        model, config, optimizer=optimizer, ema_model=ema
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        try:
            checkpoint = ckpt_manager.restore(args.resume)
            start_epoch = checkpoint.get("epoch", 0) + 1
            print(f"Resumed from epoch {start_epoch}")
        except Exception as e:
            print(f"Could not resume from checkpoint: {e}")
    
    # Callbacks
    callbacks = CallbackList([
        EvalDataset(model=model, dataset=val_loader, config=config),
        LossLogger(log_freq=10, use_wandb=HAS_WANDB and not args.no_wandb),
    ])
    
    # Training loop
    print(f"\nStarting training for {config.epochs} epochs...")
    callbacks.on_train_begin()
    
    best_loss = float("inf")
    
    for epoch in range(start_epoch, config.epochs):
        callbacks.on_epoch_begin(epoch)
        
        # Train for one epoch
        epoch_loss = train_epoch(
            model, train_loader, optimizer, device, epoch, config, ema=ema, callbacks=callbacks
        )
        
        # Learning rate scheduling
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(epoch_loss)
            else:
                scheduler.step()
        
        # Log
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch + 1}/{config.epochs} - Loss: {epoch_loss:.6f} - LR: {current_lr:.2e}")
        
        if HAS_WANDB and not args.no_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "epoch_loss": epoch_loss,
                "learning_rate": current_lr,
            })
        
        # Callbacks
        logs = {"epoch_loss": epoch_loss, "lr": current_lr}
        callbacks.on_epoch_end(epoch, logs)
        
        # Checkpointing
        ckpt_manager.on_epoch_end(epoch, logs)
        
        # Track best
        if epoch_loss < best_loss:
            best_loss = epoch_loss
    
    callbacks.on_train_end()
    print(f"\nTraining complete! Best loss: {best_loss:.6f}")
    
    return model


def main():
    """Main entry point."""
    args = parse_args()
    
    # Handle config path
    config_path = Path(args.config)
    if not config_path.exists():
        # Try relative to configs/training
        config_path = Path(f"./configs/training/{config_path.stem}.yaml")
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {args.config}")
    
    print(f"Using config file: {config_path}")
    
    # Load config
    config = load_config_from_yaml(config_path, wandb_file=True)
    
    # Apply command line overrides
    config = add_args_to_config(args, config, verbose=True)
    
    # Initialize wandb
    run = None
    if HAS_WANDB and not args.no_wandb:
        run = wandb.init(
            project="deep_generative",
            group="generative",
            config=dict(config),
            job_type="train",
            allow_val_change=True,
        )
        print(f"wandb: {run.job_type} run {run.name}")
        config.update({"log_dir": run.dir})
    else:
        # Create local log directory
        from utils.utils import make_unique_path, get_date_string
        log_dir = make_unique_path(f"./runs/{get_date_string()}")
        config.update({"log_dir": str(log_dir)})
        print(f"Logging to: {config.log_dir}")
    
    # Add git info
    config.update({"git": get_git_summary()})
    
    try:
        # Train
        model = train(config, args)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise
    
    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
