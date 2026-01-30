"""Test training loop with synthetic data.

This test verifies the full training pipeline works without requiring real data.

Usage:
    python test_training_loop.py
    
Or via SLURM:
    sbatch --wrap="python test_training_loop.py" --gres=gpu:1 --mem=8G --time=00:10:00
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


def create_synthetic_data(tmpdir, n_train=32, n_val=8, shape=(3, 128, 64)):
    """Create synthetic NPZ data files for testing."""
    data_dir = Path(tmpdir) / "zea_synth" / "tissue"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create random RF-like data
    train_data = np.random.randn(n_train, *shape).astype(np.float32)
    val_data = np.random.randn(n_val, *shape).astype(np.float32)
    
    # Normalize to [0, 1]
    train_data = (train_data - train_data.min()) / (train_data.max() - train_data.min())
    val_data = (val_data - val_data.min()) / (val_data.max() - val_data.min())
    
    np.savez(data_dir / "train.npz", rf=train_data)
    np.savez(data_dir / "val.npz", rf=val_data)
    
    print(f"Created synthetic data: {n_train} train, {n_val} val samples")
    print(f"Data shape: {shape}")
    return str(tmpdir)


class SimpleConfig:
    """Minimal config for testing."""
    def __init__(self, **kwargs):
        defaults = {
            "dataset_name": "zea_tissue",
            "data_root": None,
            "batch_size": 4,
            "epochs": 2,
            "lr": 1e-4,
            "seed": 42,
            "image_size": [128, 64],
            "image_range": [0, 1],
            "npz_key": "rf",
            "num_workers": 0,
            "shuffle": True,
            "model_name": "score",
            "score_backbone": "NCSNv2",
            "channels": 16,  # Small for testing
            "activation": "elu",
            "normalization": "instance",
            "kernel_size": 3,
            "sde": "vesde",
            "sigma_min": 0.01,
            "sigma_max": 50,
            "num_scales": 10,
            "reduce_mean": True,
            "likelihood_weighting": False,
            "save_freq": 1,
            "eval_freq": 1,
            "num_img": 4,
            "color_mode": "grayscale",
            "ema": 0.999,
            "grad_clip": 1.0,
        }
        defaults.update(kwargs)
        for k, v in defaults.items():
            setattr(self, k, v)
    
    def get(self, key, default=None):
        return getattr(self, key, default)
    
    def update(self, d):
        for k, v in d.items():
            setattr(self, k, v)


def test_training_loop():
    """Test full training loop with synthetic data."""
    print("\n" + "=" * 60)
    print("TEST: Full Training Loop")
    print("=" * 60 + "\n")
    
    # Import training components
    from datasets import get_dataset
    from generators.models import get_model, create_optimizer
    from utils.checkpoints import ModelCheckpoint, EMAHelper
    from utils.gpu_config import get_device
    from utils.utils import set_random_seed
    
    # Check no TensorFlow
    assert "tensorflow" not in sys.modules, "TensorFlow should not be imported!"
    print("[OK] No TensorFlow imported")
    
    # Create temporary directory for data and checkpoints
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create synthetic data
        data_root = create_synthetic_data(tmpdir, n_train=16, n_val=4, shape=(3, 128, 64))
        
        # Create config
        config = SimpleConfig(
            data_root=data_root,
            log_dir=str(Path(tmpdir) / "runs"),
        )
        
        # Set device
        device = get_device(config)
        print(f"Using device: {device}")
        
        # Set random seed
        set_random_seed(config.seed)
        
        # Load dataset
        train_loader, val_loader = get_dataset(config)
        print(f"[OK] Dataset loaded: {len(train_loader)} train batches")
        
        # Get a batch to verify data loading
        batch = next(iter(train_loader))
        print(f"[OK] Batch shape: {batch.shape}")
        assert batch.shape == (4, 3, 128, 64), f"Unexpected batch shape: {batch.shape}"
        
        # Create model
        model = get_model(config, plot_summary=True)
        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[OK] Model created: {n_params:,} parameters")
        
        # Create optimizer
        optimizer = create_optimizer(model, config)
        print("[OK] Optimizer created")
        
        # Create EMA
        ema = EMAHelper(model, decay=config.ema, device=device)
        print("[OK] EMA helper created")
        
        # Create checkpoint manager
        ckpt = ModelCheckpoint(model, config, optimizer=optimizer)
        print("[OK] Checkpoint manager created")
        
        # Training loop
        print("\nRunning training loop...")
        model.train()
        
        losses = []
        for epoch in range(config.epochs):
            epoch_losses = []
            
            for batch_idx, batch in enumerate(train_loader):
                batch = batch.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                loss = model.score_loss(batch)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                
                optimizer.step()
                
                # EMA update
                ema.update(model)
                
                loss_val = loss.item()
                epoch_losses.append(loss_val)
            
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            print(f"  Epoch {epoch + 1}/{config.epochs}: loss = {avg_loss:.6f}")
        
        # Verify loss is finite and not increasing dramatically
        assert all(np.isfinite(l) for l in losses), "Loss became NaN or Inf!"
        print("[OK] All losses are finite")
        
        # Loss should generally be reasonable (not exploding)
        assert losses[-1] < 1000, f"Loss too high: {losses[-1]}"
        print(f"[OK] Final loss is reasonable: {losses[-1]:.6f}")
        
        # Save checkpoint
        ckpt.save(epoch=config.epochs - 1)
        print("[OK] Checkpoint saved")
        
        # Verify checkpoint can be loaded
        ckpt2 = ModelCheckpoint(model, config)
        ckpt2.restore()
        print("[OK] Checkpoint restored")
        
        # Test evaluation mode
        model.eval()
        with torch.no_grad():
            eval_batch = next(iter(val_loader)).to(device)
            eval_loss = model.score_loss(eval_batch)
            print(f"[OK] Eval loss: {eval_loss.item():.6f}")
        
        print("\n" + "=" * 60)
        print("ALL TRAINING LOOP TESTS PASSED!")
        print("=" * 60 + "\n")


def test_imports():
    """Test all imports work without TensorFlow."""
    print("\n" + "=" * 60)
    print("TEST: Import Chain")
    print("=" * 60 + "\n")
    
    # Core imports
    from datasets import get_dataset
    print("[OK] from datasets import get_dataset")
    
    from generators.models import get_model
    print("[OK] from generators.models import get_model")
    
    from utils.utils import load_config_from_yaml, set_random_seed
    print("[OK] from utils.utils import load_config_from_yaml, set_random_seed")
    
    from utils.checkpoints import ModelCheckpoint, EMAHelper
    print("[OK] from utils.checkpoints import ModelCheckpoint, EMAHelper")
    
    from utils.callbacks import CallbackList, EvalDataset
    print("[OK] from utils.callbacks import CallbackList, EvalDataset")
    
    from utils.gpu_config import set_gpu_usage, get_device
    print("[OK] from utils.gpu_config import set_gpu_usage, get_device")
    
    from utils.inverse import SGMDenoiser, get_denoiser
    print("[OK] from utils.inverse import SGMDenoiser, get_denoiser")
    
    # SGM components
    from generators.SGM.SGM import ScoreNet, NCSNv2
    print("[OK] from generators.SGM.SGM import ScoreNet, NCSNv2")
    
    from generators.SGM.sampling import ScoreSampler
    print("[OK] from generators.SGM.sampling import ScoreSampler")
    
    from generators.SGM.guidance import PIGDM, DPS
    print("[OK] from generators.SGM.guidance import PIGDM, DPS")
    
    from utils.corruptors import GaussianCorruptor
    print("[OK] from utils.corruptors import GaussianCorruptor")
    
    # Check NO TensorFlow
    if "tensorflow" in sys.modules:
        print("[FAIL] TensorFlow was imported!")
        sys.exit(1)
    else:
        print("\n[OK] No TensorFlow in sys.modules")
    
    print("\n" + "=" * 60)
    print("ALL IMPORT TESTS PASSED!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    test_imports()
    test_training_loop()
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED - TRAINING LOOP IS WORKING!")
    print("=" * 60 + "\n")
