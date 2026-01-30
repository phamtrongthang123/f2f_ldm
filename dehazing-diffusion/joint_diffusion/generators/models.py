"""Load and save deep learning models (PyTorch version)
Author(s): Tristan Stevens
Ported to PyTorch: Jan 2026

Note: Only 'score' (ScoreNet) and 'glow' models are supported.
Legacy TF models (GAN, NCSNv2-Keras, UNet-Keras) have been removed.
"""
import torch
from generators.SGM.SGM import ScoreNet


def get_model(config, run_eagerly=False, plot_summary=False, training=True):
    """Return model based on config parameters.

    Args:
        config (dict): dict object with model init parameters.
            requires different keys for different models.
        run_eagerly (bool, optional): Not used in PyTorch. Kept for API compatibility.
        plot_summary (bool, optional): If True, print model summary. Defaults to False.
        training (bool, optional): Training mode flag. Defaults to True.

    Returns:
        model: PyTorch model (nn.Module)
    """
    model_name = config.model_name

    supported_models = ["score", "glow"]
    assert model_name.lower() in supported_models, (
        f"Invalid model name '{model_name}' found in config file. "
        f"Supported models: {supported_models}"
    )

    print(f"\nLoading {model_name} model...")

    if model_name.lower() == "score":
        model = ScoreNet(config)
        
        if plot_summary:
            _print_model_summary(model, config)
        
        return model

    if model_name.lower() == "glow":
        # Glow is already PyTorch
        from generators.glow.glow import Glow
        import numpy as np
        
        model = Glow(
            np.array(config.image_shape)[[2, 0, 1]],
            K=config.K,
            L=config.L,
            coupling=config.coupling,
            n_bits_x=config.n_bits_x,
            nn_init_last_zeros=config.last_zeros,
            device=config.device,
        )
        
        return model

    raise ValueError(f"Model {model_name} not implemented")


def _print_model_summary(model, config):
    """Print a summary of the model architecture."""
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{'='*60}")
    print(f"Model: {model.__class__.__name__}")
    print(f"{'='*60}")
    print(f"  Input shape: {config.image_shape}")
    print(f"  Total parameters: {n_params:,}")
    print(f"  Trainable parameters: {n_trainable:,}")
    print(f"  SDE: {config.get('sde', 'N/A')}")
    print(f"  Backbone: {config.get('score_backbone', 'NCSNv2')}")
    print(f"{'='*60}\n")


def create_optimizer(model, config):
    """Create optimizer for model based on config.
    
    Args:
        model: PyTorch model
        config: Config with lr, ema, etc.
    
    Returns:
        optimizer: PyTorch optimizer
    """
    lr = config.get("lr", 1e-4)
    betas = config.get("adam_betas", (0.9, 0.999))
    weight_decay = config.get("weight_decay", 0.0)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        betas=tuple(betas),
        weight_decay=weight_decay,
    )
    
    return optimizer


def create_lr_scheduler(optimizer, config):
    """Create learning rate scheduler based on config.
    
    Args:
        optimizer: PyTorch optimizer
        config: Config with scheduler params
    
    Returns:
        scheduler or None
    """
    scheduler_type = config.get("lr_scheduler", None)
    
    if scheduler_type is None:
        return None
    
    if scheduler_type == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.get("lr_factor", 0.3),
            patience=config.get("lr_patience", 10),
            verbose=True,
        )
    
    if scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.get("epochs", 100),
            eta_min=config.get("lr_min", 1e-6),
        )
    
    if scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get("lr_step_size", 30),
            gamma=config.get("lr_gamma", 0.1),
        )
    
    return None
