"""
Dataset and DataLoader for GASF images.

Supports loading from .npz files and applying augmentations for MAE pretraining.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List, Callable
import random


class GASFDataset(Dataset):
    """
    Dataset for GASF (Gramian Angular Summation Field) images.

    Loads data from .npz file with structure:
    - 'gasf_data': (N, C, H, W) array of GASF images
    - 'file_names': (N,) array of file names
    """

    def __init__(
        self,
        npz_path: str,
        target_size: int = 256,
        transform: Optional[Callable] = None,
        augment: bool = True,
        normalize: bool = True
    ):
        """
        Args:
            npz_path: Path to .npz file containing GASF data
            target_size: Target image size (will resize if different)
            transform: Optional custom transform function
            augment: Whether to apply data augmentation
            normalize: Whether to normalize to [-1, 1] range
        """
        self.npz_path = npz_path
        self.target_size = target_size
        self.transform = transform
        self.augment = augment
        self.normalize = normalize

        # Load data
        print(f"Loading dataset from {npz_path}...")
        data = np.load(npz_path, allow_pickle=True)

        self.images = data['gasf_data'].astype(np.float32)
        self.file_names = data['file_names']

        # Get data info
        self.num_samples = len(self.images)
        self.in_channels = self.images.shape[1]
        self.original_size = self.images.shape[2]

        print(f"Loaded {self.num_samples} samples")
        print(f"Shape: {self.images.shape}")
        print(f"Value range: [{self.images.min():.4f}, {self.images.max():.4f}]")

        # Normalize to [-1, 1] if requested and data is in [0, 1] or other range
        if self.normalize:
            # Normalize each sample independently to handle varying ranges
            self.images = self._normalize_data(self.images)

    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """Normalize data to [-1, 1] range."""
        # Global normalization
        data_min = data.min()
        data_max = data.max()

        if data_max - data_min > 1e-6:
            data = 2 * (data - data_min) / (data_max - data_min) - 1
        else:
            data = np.zeros_like(data)

        return data

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a single GASF image.

        Args:
            idx: Sample index

        Returns:
            image: (C, H, W) tensor
        """
        image = self.images[idx].copy()  # (C, H, W)

        # Apply augmentation
        if self.augment:
            image = self._augment(image)

        # Apply custom transform if provided
        if self.transform is not None:
            image = self.transform(image)

        # Convert to tensor
        image = torch.from_numpy(image)

        return image

    def _augment(self, image: np.ndarray) -> np.ndarray:
        """
        Apply random augmentations suitable for GASF images.

        Args:
            image: (C, H, W) array

        Returns:
            Augmented image
        """
        # # Random rotation (0, 90, 180, 270 degrees)
        # if random.random() < 0.5:
        #     k = random.randint(0, 3)
        #     image = np.rot90(image, k, axes=(1, 2)).copy()

        # # Random horizontal flip
        # if random.random() < 0.5:
        #     image = np.flip(image, axis=2).copy()

        # # Random vertical flip
        # if random.random() < 0.5:
        #     image = np.flip(image, axis=1).copy()

        # Add small Gaussian noise
        if random.random() < 0.3:
            noise = np.random.normal(0, 0.02, image.shape).astype(np.float32)
            image = image + noise
            image = np.clip(image, -1, 1)

        # # Random scaling (brightness-like adjustment)
        # if random.random() < 0.3:
        #     scale = random.uniform(0.9, 1.1)
        #     image = image * scale
        #     image = np.clip(image, -1, 1)

        return image

    def get_file_name(self, idx: int) -> str:
        """Get file name for a given index."""
        return self.file_names[idx]


class GASFPretrainingDataset(GASFDataset):
    """
    Dataset for MAE pretraining.

    Returns images without labels (self-supervised).
    Applies stronger augmentation.
    """

    def __init__(
        self,
        npz_path: str,
        target_size: int = 256,
        transform: Optional[Callable] = None,
        augment: bool = True
    ):
        super().__init__(
            npz_path=npz_path,
            target_size=target_size,
            transform=transform,
            augment=augment,
            normalize=True
        )


def _is_genuine(file_name: str) -> bool:
    """Check if a sample is genuine based on its file name.

    Naming conventions handled:
      - DeepSignDB simple:  u0001_g_0100v00.txt (genuine), u0001_s_0100f00.txt (skilled forgery)
      - DeepSignDB complex: u0231_g_u1013s0001_sg0001.txt (genuine), u0231_s_u1013s0001_sg0003.txt (skilled)
      - Generic:            *genuine* / *forgery* / *skilled*
    The second underscore-delimited field ('g' or 's') is the authoritative marker.
    """
    name_lower = file_name.lower()
    parts = name_lower.split('_')

    # Use the second field (index 1) as the primary indicator when available
    if len(parts) >= 2:
        marker = parts[1]
        if marker == 'g':
            return True
        if marker == 's':
            return False

    # Fallback: keyword-based detection
    if 'forgery' in name_lower or 'f_' in name_lower or 'skilled' in name_lower:
        return False
    if 'genuine' in name_lower or 'g_' in name_lower:
        return True

    # Default: treat as genuine
    return True


def create_dataloaders(
    npz_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    train_ratio: float = 0.9,
    seed: int = 42,
    augment_train: bool = True,
    genuine_only: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        npz_path: Path to .npz file
        batch_size: Batch size
        num_workers: Number of data loading workers
        train_ratio: Fraction of data for training
        seed: Random seed for reproducibility
        augment_train: Whether to augment training data
        genuine_only: If True, only use genuine samples (exclude forged/skilled)

    Returns:
        train_loader, val_loader
    """
    # Load full dataset
    full_data = np.load(npz_path, allow_pickle=True)
    images = full_data['gasf_data'].astype(np.float32)
    file_names = full_data['file_names']

    # Filter to genuine samples only if requested
    if genuine_only:
        genuine_mask = np.array([
            _is_genuine(str(name)) for name in file_names
        ])
        images = images[genuine_mask]
        file_names = file_names[genuine_mask]
        print(f"Filtered to genuine samples only: {len(images)} / {genuine_mask.shape[0]} total")

    num_samples = len(images)
    num_train = int(num_samples * train_ratio)

    # Random split
    np.random.seed(seed)
    indices = np.random.permutation(num_samples)
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]

    # Create separate .npz files for train and val (in memory)
    train_images = images[train_indices]
    train_names = file_names[train_indices]

    val_images = images[val_indices]
    val_names = file_names[val_indices]

    # Create datasets
    train_dataset = _InMemoryGASFDataset(
        train_images, train_names, augment=augment_train
    )
    val_dataset = _InMemoryGASFDataset(
        val_images, val_names, augment=False
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    return train_loader, val_loader


class _InMemoryGASFDataset(Dataset):
    """Helper dataset that works with in-memory arrays."""

    def __init__(
        self,
        images: np.ndarray,
        file_names: np.ndarray,
        augment: bool = True
    ):
        self.images = images
        self.file_names = file_names
        self.augment = augment

        # Normalize
        data_min = self.images.min()
        data_max = self.images.max()
        if data_max - data_min > 1e-6:
            self.images = 2 * (self.images - data_min) / (data_max - data_min) - 1
        else:
            self.images = np.zeros_like(self.images)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image = self.images[idx].copy()

        if self.augment:
            image = self._augment(image)

        return torch.from_numpy(image)

    def _augment(self, image: np.ndarray) -> np.ndarray:
        # Random rotation
        if random.random() < 0.5:
            k = random.randint(0, 3)
            image = np.rot90(image, k, axes=(1, 2)).copy()

        # Random flips
        if random.random() < 0.5:
            image = np.flip(image, axis=2).copy()
        if random.random() < 0.5:
            image = np.flip(image, axis=1).copy()

        # Gaussian noise
        if random.random() < 0.3:
            noise = np.random.normal(0, 0.02, image.shape).astype(np.float32)
            image = np.clip(image + noise, -1, 1)

        return image


if __name__ == "__main__":
    # Test dataset
    import sys

    npz_path = r"C:\Users\Himanshu Singhal\Desktop\BTP\working\deepsigndb_asymmetric_gasf.npz"

    if os.path.exists(npz_path):
        print("Testing GASFPretrainingDataset...")
        dataset = GASFPretrainingDataset(npz_path, augment=True)

        print(f"\nDataset length: {len(dataset)}")

        # Get a sample
        sample = dataset[0]
        print(f"Sample shape: {sample.shape}")
        print(f"Sample dtype: {sample.dtype}")
        print(f"Sample range: [{sample.min():.4f}, {sample.max():.4f}]")

        # Test dataloader
        print("\nTesting create_dataloaders...")
        train_loader, val_loader = create_dataloaders(
            npz_path, batch_size=32, num_workers=0
        )

        for batch in train_loader:
            print(f"Batch shape: {batch.shape}")
            break
    else:
        print(f"Dataset not found at {npz_path}")
