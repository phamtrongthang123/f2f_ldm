"""Load and save checkpoints (PyTorch version)
Author(s): Tristan Stevens
Ported to PyTorch: Jan 2026
"""
from pathlib import Path

import torch

from utils.utils import get_latest_checkpoint


class ModelCheckpoint:
    """Checkpoint manager for PyTorch models.
    
    Handles saving and loading of model weights, optimizer state, and training state.
    """
    
    def __init__(self, model, config, optimizer=None, ema_model=None, **kwargs):
        """
        Args:
            model: PyTorch model (nn.Module)
            config: Config with log_dir, save_freq, epochs, pretrained, model_name
            optimizer: Optional optimizer to save state
            ema_model: Optional EMA model to save
        """
        self.model = model
        self.optimizer = optimizer
        self.ema_model = ema_model

        # Determine checkpoint directory
        log_dir = str(config.get("log_dir", "./runs"))
        # OS independent path
        log_dir = "/".join(log_dir.split("\\"))
        
        # Make relative path if wandb
        if "wandb" in str(log_dir):
            self.checkpoint_dir = Path("./wandb" + log_dir.split("wandb")[-1])
        else:
            self.checkpoint_dir = Path(log_dir)

        self.checkpoint_dir = self.checkpoint_dir / "training_checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_freq = config.get("save_freq", 10)
        self.epochs = config.get("epochs", 100)
        self.checkpoint_prefix = self.checkpoint_dir / "ckpt"

        # Handle pretrained weights
        if config.get("pretrained"):
            self.pretrained = (
                self.checkpoint_dir.parents[2]
                / config.pretrained
                / "files"
                / "training_checkpoints"
            )
        else:
            self.pretrained = None

        self.model_name = config.get("model_name", "score").lower()

    def on_train_begin(self, logs=None):
        """Load pretrained weights if specified."""
        if self.pretrained:
            try:
                self.restore()
                print(f"Loaded pretrained weights from {self.pretrained}")
            except Exception as e:
                print(f"Could not load pretrained weights: {e}")

    def on_epoch_end(self, epoch, logs=None):
        """Save checkpoint at end of epoch if conditions met."""
        if (epoch + 1) % self.save_freq == 0 or ((epoch + 1) == self.epochs):
            self.save(epoch)

    def save(self, epoch, extra_state=None):
        """Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            extra_state: Optional dict of additional state to save
        """
        path = str(self.checkpoint_prefix) + f"-{epoch}.pt"
        Path(path).parent.mkdir(exist_ok=True, parents=True)
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
        }
        
        if self.optimizer is not None:
            checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()
        
        if self.ema_model is not None:
            checkpoint["ema_state_dict"] = self.ema_model.state_dict()
        
        if extra_state is not None:
            checkpoint.update(extra_state)
        
        torch.save(checkpoint, path)
        print(f"--> Successfully saved checkpoint (epoch: {epoch + 1}) to {path}")
        
        return path

    def restore(self, file=None, load_optimizer=True, load_ema=True, map_location=None):
        """Load weights from checkpoint file.
        
        Args:
            file: Path to checkpoint file. If None, loads latest.
            load_optimizer: Whether to restore optimizer state
            load_ema: Whether to restore EMA model state
            map_location: Device mapping for torch.load
        
        Returns:
            Loaded checkpoint dict
        """
        file = self.get_checkpoint(file)
        
        if map_location is None:
            map_location = "cuda" if torch.cuda.is_available() else "cpu"
        
        checkpoint = torch.load(file, map_location=map_location)
        
        # Load model weights
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            # Legacy format - entire state dict was saved directly
            self.model.load_state_dict(checkpoint)
        
        # Load optimizer state
        if load_optimizer and self.optimizer is not None and "optimizer_state_dict" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception as e:
                print(f"Could not load optimizer state: {e}")
        
        # Load EMA model state
        if load_ema and self.ema_model is not None and "ema_state_dict" in checkpoint:
            try:
                self.ema_model.load_state_dict(checkpoint["ema_state_dict"])
            except Exception as e:
                print(f"Could not load EMA state: {e}")

        epoch = checkpoint.get("epoch", -1)
        print(f"--> Successfully loaded weights from {file} (epoch {epoch + 1})")
        
        return checkpoint

    def get_checkpoint(self, file=None):
        """Get checkpoint file path.
        
        Args:
            file: Specific file path, or None to get latest
        
        Returns:
            Path to checkpoint file
        """
        if file is None:
            # Find latest checkpoint
            file = get_latest_checkpoint(self.checkpoint_dir, "pt", split="-")
            if file is None:
                # Also check pretrained directory
                if self.pretrained and self.pretrained.exists():
                    file = get_latest_checkpoint(self.pretrained, "pt", split="-")
                
                if file is None:
                    raise ValueError(
                        f"No checkpoint file found in {self.checkpoint_dir} !"
                    )
        elif not Path(file).is_absolute():
            file = self.checkpoint_dir / file

        file = Path(file)
        if not file.suffix:
            file = file.with_suffix(".pt")

        if not file.is_file():
            raise ValueError(
                f"Checkpoint file: {file.name} not found in {self.checkpoint_dir}!"
            )

        return file

    def get_epoch(self, file=None):
        """Get epoch number from checkpoint.
        
        Args:
            file: Checkpoint file path
        
        Returns:
            Epoch number or -1 if not found
        """
        if file is None:
            try:
                file = self.get_checkpoint()
            except ValueError:
                return -1
        
        checkpoint = torch.load(file, map_location="cpu")
        return checkpoint.get("epoch", -1)


class EMAHelper:
    """Exponential Moving Average helper for model weights.
    
    Maintains a shadow copy of model parameters with EMA updates.
    Based on: https://github.com/ermongroup/ncsnv2
    """
    
    def __init__(self, model, decay=0.9999, device=None):
        """
        Args:
            model: PyTorch model to track
            decay: EMA decay rate (default 0.9999)
            device: Device for shadow parameters
        """
        self.decay = decay
        self.device = device or next(model.parameters()).device
        
        # Create shadow parameters
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().to(self.device)
    
    def update(self, model):
        """Update shadow parameters with EMA.
        
        Args:
            model: Model with updated parameters
        """
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data.to(self.device), alpha=1 - self.decay
                )
    
    def apply_shadow(self, model):
        """Apply shadow parameters to model.
        
        Args:
            model: Model to update
        """
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data.copy_(self.shadow[name])
    
    def restore(self, model, backup):
        """Restore model parameters from backup.
        
        Args:
            model: Model to restore
            backup: Dict of backed up parameters
        """
        for name, param in model.named_parameters():
            if param.requires_grad and name in backup:
                param.data.copy_(backup[name])
    
    def get_backup(self, model):
        """Get backup of current model parameters.
        
        Args:
            model: Model to backup
        
        Returns:
            Dict of parameter tensors
        """
        backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                backup[name] = param.data.clone()
        return backup
    
    def state_dict(self):
        """Get state dict for saving."""
        return {"shadow": self.shadow, "decay": self.decay}
    
    def load_state_dict(self, state_dict):
        """Load state dict."""
        self.shadow = state_dict["shadow"]
        self.decay = state_dict.get("decay", self.decay)
