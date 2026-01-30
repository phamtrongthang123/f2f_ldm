"""Callbacks for PyTorch training
Author(s): Tristan Stevens
Ported to PyTorch: Jan 2026

Note: These are plain Python classes that get called from the training loop,
not Keras Callbacks. The interface is similar for compatibility.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class Callback:
    """Base callback class with PyTorch-compatible interface."""
    
    def on_train_begin(self, logs=None):
        """Called at the beginning of training."""
        pass
    
    def on_train_end(self, logs=None):
        """Called at the end of training."""
        pass
    
    def on_epoch_begin(self, epoch, logs=None):
        """Called at the beginning of each epoch."""
        pass
    
    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch."""
        pass
    
    def on_batch_begin(self, batch, logs=None):
        """Called at the beginning of each batch."""
        pass
    
    def on_batch_end(self, batch, logs=None):
        """Called at the end of each batch."""
        pass


class CallbackList:
    """Container for managing multiple callbacks."""
    
    def __init__(self, callbacks=None):
        self.callbacks = callbacks or []
    
    def append(self, callback):
        self.callbacks.append(callback)
    
    def on_train_begin(self, logs=None):
        for cb in self.callbacks:
            cb.on_train_begin(logs)
    
    def on_train_end(self, logs=None):
        for cb in self.callbacks:
            cb.on_train_end(logs)
    
    def on_epoch_begin(self, epoch, logs=None):
        for cb in self.callbacks:
            cb.on_epoch_begin(epoch, logs)
    
    def on_epoch_end(self, epoch, logs=None):
        for cb in self.callbacks:
            cb.on_epoch_end(epoch, logs)
    
    def on_batch_begin(self, batch, logs=None):
        for cb in self.callbacks:
            cb.on_batch_begin(batch, logs)
    
    def on_batch_end(self, batch, logs=None):
        for cb in self.callbacks:
            cb.on_batch_end(batch, logs)


class EvalDataset(Callback):
    """Callback for evaluating model on validation dataset."""
    
    def __init__(
        self,
        model,
        dataset,
        config,
        name=None,
        **kwargs,
    ):
        """
        Args:
            model: PyTorch model (ScoreNet)
            dataset: Validation DataLoader
            config: Config object
            name: Optional name for this evaluator
        """
        super().__init__()
        self.model = model
        self.dataset = dataset
        self.config = config
        self.name = name or "eval"
        
        self.eval_freq = config.get("eval_freq", 5)
        self.epochs = config.get("epochs", 100)
        self.n_eval_batches = config.get("n_eval_batches", 10)
        
        if config.get("image_range") is not None:
            self.vmin, self.vmax = config.image_range
        else:
            self.vmin, self.vmax = 0, 1
    
    def on_train_begin(self, logs=None):
        """Log sample batch at start of training."""
        if HAS_WANDB and self.dataset is not None:
            try:
                fig = self.plot_batch()
                wandb.log({"test_images": wandb.Image(fig, caption="test images")})
                plt.close(fig)
            except Exception as e:
                print(f"Could not log test images: {e}")
    
    def on_epoch_end(self, epoch, logs=None):
        """Evaluate model on validation set."""
        if (epoch + 1) % self.eval_freq == 0 or ((epoch + 1) == self.epochs):
            if self.model is not None and self.dataset is not None:
                eval_loss = self.get_eval_loss()
                
                if HAS_WANDB:
                    wandb.log({"eval_score_loss": eval_loss})
                
                print(f"Eval score loss: {eval_loss:.6f}")
                
                if logs is not None:
                    logs["eval_loss"] = eval_loss
    
    def get_eval_loss(self):
        """Compute average loss on validation set."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for i, batch in enumerate(self.dataset):
                if i >= self.n_eval_batches:
                    break
                
                if isinstance(batch, (tuple, list)):
                    batch = batch[0]
                
                batch = batch.to(next(self.model.parameters()).device)
                loss = self.model.score_loss(batch)
                total_loss += loss.item()
                n_batches += 1
        
        self.model.train()
        return total_loss / max(n_batches, 1)
    
    def plot_batch(self, num_img=16):
        """Plot a batch of images from the dataset."""
        batch = next(iter(self.dataset))
        if isinstance(batch, (tuple, list)):
            batch = batch[0]
        
        batch = batch[:num_img].cpu().numpy()
        
        # Convert from (B, C, H, W) to (B, H, W, C) for plotting
        if batch.ndim == 4:
            batch = batch.transpose(0, 2, 3, 1)
        
        n = int(np.ceil(np.sqrt(num_img)))
        fig, axs = plt.subplots(n, n, figsize=(8, 8))
        axs = axs.flatten()
        
        for i, ax in enumerate(axs):
            if i < len(batch):
                img = np.squeeze(batch[i])
                if img.ndim == 2:
                    ax.imshow(img, cmap="gray", vmin=self.vmin, vmax=self.vmax)
                else:
                    ax.imshow(img)
            ax.axis("off")
        
        fig.tight_layout()
        return fig


class Monitor(Callback):
    """Callback for monitoring training with sample generation."""
    
    def __init__(self, model, config, num_img=None, **kwargs):
        """
        Args:
            model: PyTorch model (ScoreNet)
            config: Config object
            num_img: Number of images to generate for monitoring
        """
        super().__init__()
        self.model = model
        self.config = config
        self.eval_freq = config.get("eval_freq", 5)
        self.epochs = config.get("epochs", 100)
        self.log_dir = config.get("log_dir", "./runs")
        
        if num_img is None:
            self.num_img = config.get("num_img", 16)
        else:
            self.num_img = num_img
        
        if config.get("image_range") is not None:
            self.vmin, self.vmax = config.image_range
        else:
            self.vmin, self.vmax = 0, 1
    
    def on_epoch_end(self, epoch, logs=None):
        """Generate and log sample images."""
        if (epoch + 1) % self.eval_freq == 0 or ((epoch + 1) == self.epochs):
            try:
                fig = self.plot_samples()
                
                if HAS_WANDB:
                    wandb.log({
                        "generated_images": wandb.Image(fig, caption=f"epoch_{epoch+1}")
                    })
                
                plt.close(fig)
            except Exception as e:
                print(f"Could not generate samples at epoch {epoch}: {e}")
    
    def plot_samples(self):
        """Generate and plot samples from the model."""
        self.model.eval()
        
        # Get image shape
        image_shape = self.config.image_shape
        
        # Generate samples
        with torch.no_grad():
            # Use model's sampler if available
            if hasattr(self.model, "sample"):
                samples = self.model.sample(
                    n_samples=self.num_img,
                    shape=image_shape,
                )
            else:
                # Fallback: just show random noise
                device = next(self.model.parameters()).device
                samples = torch.randn(self.num_img, *image_shape, device=device)
        
        self.model.train()
        
        # Convert to numpy for plotting
        samples = samples.cpu().numpy()
        
        # Plot
        n = int(np.ceil(np.sqrt(self.num_img)))
        fig, axs = plt.subplots(n, n, figsize=(8, 8))
        axs = axs.flatten()
        
        for i, ax in enumerate(axs):
            if i < len(samples):
                img = samples[i]
                # Convert from (C, H, W) to (H, W, C) if needed
                if img.ndim == 3 and img.shape[0] in [1, 3]:
                    img = img.transpose(1, 2, 0)
                img = np.squeeze(img)
                
                if img.ndim == 2:
                    ax.imshow(img, cmap="gray", vmin=self.vmin, vmax=self.vmax)
                else:
                    ax.imshow(np.clip(img, 0, 1))
            ax.axis("off")
        
        fig.tight_layout()
        return fig


class LossLogger(Callback):
    """Callback for logging training loss."""
    
    def __init__(self, log_freq=10, use_wandb=True):
        """
        Args:
            log_freq: How often to log (every N batches)
            use_wandb: Whether to log to wandb
        """
        super().__init__()
        self.log_freq = log_freq
        self.use_wandb = use_wandb and HAS_WANDB
        
        self.epoch_losses = []
        self.batch_count = 0
        self.running_loss = 0.0
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_losses = []
        self.batch_count = 0
        self.running_loss = 0.0
    
    def on_batch_end(self, batch, logs=None):
        if logs is not None and "loss" in logs:
            loss = logs["loss"]
            self.epoch_losses.append(loss)
            self.running_loss += loss
            self.batch_count += 1
            
            if self.batch_count % self.log_freq == 0:
                avg_loss = self.running_loss / self.log_freq
                
                if self.use_wandb:
                    wandb.log({
                        "train_loss": avg_loss,
                        "batch": self.batch_count,
                    })
                
                self.running_loss = 0.0
    
    def on_epoch_end(self, epoch, logs=None):
        if self.epoch_losses:
            epoch_loss = np.mean(self.epoch_losses)
            
            if logs is not None:
                logs["epoch_loss"] = epoch_loss
            
            if self.use_wandb:
                wandb.log({
                    "epoch_loss": epoch_loss,
                    "epoch": epoch,
                })


class ProgressBar(Callback):
    """Progress bar callback using tqdm."""
    
    def __init__(self, epochs, steps_per_epoch=None):
        super().__init__()
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.pbar = None
        self.epoch_pbar = None
    
    def on_train_begin(self, logs=None):
        self.epoch_pbar = tqdm.tqdm(total=self.epochs, desc="Training", unit="epoch")
    
    def on_train_end(self, logs=None):
        if self.epoch_pbar:
            self.epoch_pbar.close()
    
    def on_epoch_begin(self, epoch, logs=None):
        if self.steps_per_epoch:
            self.pbar = tqdm.tqdm(
                total=self.steps_per_epoch,
                desc=f"Epoch {epoch + 1}/{self.epochs}",
                leave=False,
            )
    
    def on_epoch_end(self, epoch, logs=None):
        if self.pbar:
            self.pbar.close()
        if self.epoch_pbar:
            self.epoch_pbar.update(1)
            if logs and "epoch_loss" in logs:
                self.epoch_pbar.set_postfix(loss=f"{logs['epoch_loss']:.4f}")
    
    def on_batch_end(self, batch, logs=None):
        if self.pbar:
            self.pbar.update(1)
            if logs and "loss" in logs:
                self.pbar.set_postfix(loss=f"{logs['loss']:.4f}")
