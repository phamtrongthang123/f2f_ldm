"""Load / generate datasets (PyTorch version).
Author(s): Tristan Stevens
Ported to PyTorch: Jan 2026

Note: Only ZEA datasets are fully supported. Legacy TF datasets (MNIST, CelebA, etc.)
have been removed. If needed, they can be reimplemented using torchvision.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

_DATASETS = [
    "zea_tissue",
    "zea_haze",
]


def get_dataset(config):
    """
    Generate a dataset based on parameters specified in config object.

    Args:
        config (dict): config dict.

    Returns:
        tuple(DataLoader, DataLoader)
            Tuple of train-, and val-dataloader respectively
    """
    dataset_name = config["dataset_name"]

    assert (
        dataset_name.lower() in _DATASETS
    ), f"""Invalid dataset name {dataset_name.lower()} found in config file.
        Supported datasets: {_DATASETS}."""

    print(f"Loading {dataset_name} dataset...")

    if dataset_name.lower() == "zea_tissue":
        return _get_zea_dataset(config, "tissue")
    if dataset_name.lower() == "zea_haze":
        return _get_zea_dataset(config, "haze")

    raise ValueError(f"Dataset {dataset_name} not supported")


class ZeaDataset(Dataset):
    """PyTorch Dataset for ZEA synthetic RF data (tissue or haze)."""

    def __init__(self, npz_path, npz_key="rf", image_range=(0, 1), limit_n=None,
                 mu=255, training=True):
        """
        Args:
            npz_path: Path to the .npz file
            npz_key: Key in NPZ file to load data from
            image_range: Tuple (min, max) to normalize data to
            limit_n: Optional limit on number of samples
            mu: μ-law companding parameter (default 255)
            training: If True, apply data augmentation in __getitem__
        """
        data = np.load(npz_path)[npz_key].astype(np.float32)
        # Stored shape: (N, n_tx, n_ax, n_el) — use all transmits as channels
        # Already in (N, C, H, W) format where C=n_tx
        if limit_n:
            data = data[:limit_n]

        # Step 1: Normalize raw RF to [-1, 1] via min-max
        data_min, data_max = data.min(), data.max()
        if data_max > data_min:
            data = 2.0 * (data - data_min) / (data_max - data_min) - 1.0

        # Step 2: μ-law companding (logarithmic compression of dynamic range)
        data = np.sign(data) * np.log1p(mu * np.abs(data)) / np.log1p(mu)

        # Step 3: Rescale from [-1, 1] to image_range
        lo, hi = image_range
        data = (data + 1.0) / 2.0 * (hi - lo) + lo

        self.data = torch.from_numpy(data.astype(np.float32))
        self.image_range = (lo, hi)
        self.data_min = float(data_min)
        self.data_max = float(data_max)
        self.mu = mu
        self.training = training

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def _get_zea_dataset(config, kind: str):
    """Load ZEA synthetic RF dataset (tissue or haze).

    Args:
        config: Configuration dict/object with data_root, batch_size, etc.
        kind: Either "tissue" or "haze"

    Returns:
        Tuple of (train_loader, val_loader)
    """
    data_root = Path(config.data_root)
    npz_key = config.get("npz_key", "rf")
    image_range = config.get("image_range", [0, 1])
    batch_size = config.get("batch_size", 16)
    shuffle = config.get("shuffle", True)
    seed = config.get("seed", None)
    limit_n = config.get("limit_n_samples", None)
    num_workers = config.get("num_workers", 0)

    train_path = data_root / "zea_synth" / kind / "train.npz"
    val_path = data_root / "zea_synth" / kind / "val.npz"

    if not train_path.exists():
        raise FileNotFoundError(f"ZEA dataset not found: {train_path}")

    train_ds = ZeaDataset(train_path, npz_key, image_range, limit_n, training=True)

    if val_path.exists():
        val_ds = ZeaDataset(val_path, npz_key, image_range, limit_n, training=False)
    else:
        # If no val set, use a portion of training
        print(f"Validation file not found at {val_path}, using last 10% of train")
        n_val = max(1, len(train_ds) // 10)
        train_ds, val_ds = torch.utils.data.random_split(
            train_ds, [len(train_ds) - n_val, n_val]
        )

    print(f"Using {len(train_ds)} samples for training.")
    print(f"Using {len(val_ds)} samples for validation.")

    # Create data loaders
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        generator=generator if shuffle else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Set image shape in config
    sample = train_ds[0] if hasattr(train_ds, '__getitem__') else train_ds.dataset[train_ds.indices[0]]
    n_channels = sample.shape[0]
    config.image_shape = [n_channels, *config.get("image_size", [128, 64])]
    
    return train_loader, val_loader


# ===== Utility functions for data loading =====

def get_batch_from_loader(loader, num=None):
    """Get a single batch from a DataLoader.
    
    Args:
        loader: PyTorch DataLoader
        num: Optional, limit batch to first `num` samples
    
    Returns:
        Batch tensor
    """
    batch = next(iter(loader))
    if num is not None and num < len(batch):
        batch = batch[:num]
    return batch


def collate_paired(batch):
    """Collate function for paired data (noisy, clean)."""
    if isinstance(batch[0], tuple):
        noisy = torch.stack([b[0] for b in batch])
        clean = torch.stack([b[1] for b in batch])
        return noisy, clean
    else:
        return torch.stack(batch)
