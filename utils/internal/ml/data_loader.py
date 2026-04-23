try:
    from local_setup import local_setup
    local_setup()
except ModuleNotFoundError as e:
    from utils.internal.sbas.local_setup import local_setup
    local_setup()

import os
import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Dict
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.internal.log.logger import get_logger
from utils.internal.ml.data_config import MLDataConfig

log = get_logger()


class AtmosphericCorrectionDataset(Dataset):
    """
    PyTorch Dataset for atmospheric correction.

    Inputs (14 channels):
    - SEVIRI t1: WV_062, WV_073, IR_097, IR_087, IR_108, IR_120 (6 channels)
    - SEVIRI t2: WV_062, WV_073, IR_097, IR_087, IR_108, IR_120 (6 channels)
    - Coherence maps: 2 interferogram pairs (2 channels)

    Output (1 channel):
    - GACOS-corrected atmospheric phase delay
    """

    def __init__(
        self,
        seviri_path: str,
        coherence_path: str,
        phase_path: str,
        config: MLDataConfig,
        split: str = 'train',
        transform: Optional[A.Compose] = None
    ):
        """
        Initialize the dataset.

        Args:
            seviri_path: Path to SEVIRI temporal stack NetCDF
            coherence_path: Path to SAR coherence NetCDF
            phase_path: Path to GACOS-corrected phase NetCDF
            config: MLDataConfig instance
            split: 'train' or 'val'
            transform: Albumentations transform pipeline
        """
        self.config = config
        self.split = split
        self.transform = transform

        # Load data
        log.info(f"Loading {split} data...")
        self.seviri_data = xr.open_dataset(seviri_path)
        self.coherence_data = xr.open_dataset(coherence_path)
        self.phase_data = xr.open_dataset(phase_path)

        # Get number of samples
        self.n_pairs = len(self.seviri_data.pair)

        # Compute normalization statistics if training
        if split == 'train':
            self.compute_normalization_stats()
        else:
            self.load_normalization_stats()

        log.info(
            f"Dataset initialized: {self.n_pairs} samples, "
            f"split={split}"
        )

    def __len__(self) -> int:
        return self.n_pairs

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Tuple of (input [14, H, W], target [1, H, W])
        """
        # Extract SEVIRI data (t1 and t2, 6 channels each)
        # seviri_data has shape (pair, channel, y, x)
        seviri_sample = self.seviri_data['seviri_data'].isel(pair=idx).values
        # seviri_sample now has shape (channel, y, x) with 6 channels

        # We need two time steps, so let's assume the data has a time dimension
        # or we select consecutive pairs. For now, duplicate:
        # TODO: Fix this based on actual data structure
        seviri_t1 = seviri_sample  # (6, y, x)
        seviri_t2 = seviri_sample  # (6, y, x) - placeholder

        # Extract coherence maps (2 channels)
        # Assuming coherence_data has shape (pair, y, x) for 2 pairs
        coherence_sample = self.coherence_data['coherence'].isel(pair=idx).values
        # coherence_sample has shape (2, y, x)

        # Extract target phase
        phase_sample = self.phase_data['phase_corrected'].isel(pair=idx).values
        # phase_sample has shape (y, x)

        # Stack all input channels: (14, H, W)
        input_data = np.concatenate([
            seviri_t1,      # 6 channels
            seviri_t2,      # 6 channels
            coherence_sample  # 2 channels
        ], axis=0)  # Shape: (14, H, W)

        # Add channel dimension to target: (1, H, W)
        target_data = np.expand_dims(phase_sample, axis=0)

        # Normalize
        input_data = self.normalize(input_data)
        target_data = self.normalize_target(target_data)

        # Apply transformations if any
        if self.transform:
            # Albumentations expects (H, W, C) format
            input_transposed = np.transpose(input_data, (1, 2, 0))  # (H, W, 14)
            target_transposed = np.transpose(target_data, (1, 2, 0))  # (H, W, 1)

            # Combine for synchronized transforms
            combined = np.concatenate([input_transposed, target_transposed], axis=2)  # (H, W, 15)

            transformed = self.transform(image=combined)
            combined_transformed = transformed['image']

            # Split back
            input_transposed = combined_transformed[:, :, :14]
            target_transposed = combined_transformed[:, :, 14:]

            # Transpose back to (C, H, W)
            input_data = np.transpose(input_transposed, (2, 0, 1))
            target_data = np.transpose(target_transposed, (2, 0, 1))

        # Convert to tensors
        input_tensor = torch.from_numpy(input_data).float()
        target_tensor = torch.from_numpy(target_data).float()

        return input_tensor, target_tensor

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize input data using computed statistics.

        Args:
            data: Input array (C, H, W)

        Returns:
            Normalized array
        """
        if self.config.per_channel_norm:
            # Per-channel normalization
            for c in range(data.shape[0]):
                if self.config.normalization_method == 'standardize':
                    mean = self.norm_stats['input_mean'][c]
                    std = self.norm_stats['input_std'][c]
                    data[c] = (data[c] - mean) / (std + 1e-8)
                elif self.config.normalization_method == 'minmax':
                    min_val = self.norm_stats['input_min'][c]
                    max_val = self.norm_stats['input_max'][c]
                    data[c] = (data[c] - min_val) / (max_val - min_val + 1e-8)
        else:
            # Global normalization
            if self.config.normalization_method == 'standardize':
                mean = self.norm_stats['input_mean']
                std = self.norm_stats['input_std']
                data = (data - mean) / (std + 1e-8)
            elif self.config.normalization_method == 'minmax':
                min_val = self.norm_stats['input_min']
                max_val = self.norm_stats['input_max']
                data = (data - min_val) / (max_val - min_val + 1e-8)

        return data

    def normalize_target(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize target data.

        Args:
            data: Target array (1, H, W)

        Returns:
            Normalized array
        """
        if self.config.normalization_method == 'standardize':
            mean = self.norm_stats['target_mean']
            std = self.norm_stats['target_std']
            data = (data - mean) / (std + 1e-8)
        elif self.config.normalization_method == 'minmax':
            min_val = self.norm_stats['target_min']
            max_val = self.norm_stats['target_max']
            data = (data - min_val) / (max_val - min_val + 1e-8)

        return data

    def compute_normalization_stats(self) -> None:
        """
        Compute normalization statistics from training data.
        """
        log.info("Computing normalization statistics...")

        # Initialize statistics
        self.norm_stats = {}

        if self.config.per_channel_norm:
            # Per-channel statistics for input (14 channels)
            input_mean = []
            input_std = []
            input_min = []
            input_max = []

            # Compute for each channel
            for c in range(14):
                # Sample data from first few pairs to compute stats
                sample_data = []
                n_samples = min(10, self.n_pairs)

                for idx in range(n_samples):
                    seviri_sample = self.seviri_data['seviri_data'].isel(pair=idx).values
                    coherence_sample = self.coherence_data['coherence'].isel(pair=idx).values

                    # Build input stack
                    input_stack = np.concatenate([
                        seviri_sample,
                        seviri_sample,
                        coherence_sample
                    ], axis=0)

                    sample_data.append(input_stack[c])

                sample_data = np.array(sample_data)

                input_mean.append(np.nanmean(sample_data))
                input_std.append(np.nanstd(sample_data))
                input_min.append(np.nanmin(sample_data))
                input_max.append(np.nanmax(sample_data))

            self.norm_stats['input_mean'] = np.array(input_mean)
            self.norm_stats['input_std'] = np.array(input_std)
            self.norm_stats['input_min'] = np.array(input_min)
            self.norm_stats['input_max'] = np.array(input_max)

        else:
            # Global statistics
            # Sample data
            sample_data = []
            n_samples = min(10, self.n_pairs)

            for idx in range(n_samples):
                seviri_sample = self.seviri_data['seviri_data'].isel(pair=idx).values
                coherence_sample = self.coherence_data['coherence'].isel(pair=idx).values
                input_stack = np.concatenate([
                    seviri_sample,
                    seviri_sample,
                    coherence_sample
                ], axis=0)
                sample_data.append(input_stack)

            sample_data = np.array(sample_data)

            self.norm_stats['input_mean'] = np.nanmean(sample_data)
            self.norm_stats['input_std'] = np.nanstd(sample_data)
            self.norm_stats['input_min'] = np.nanmin(sample_data)
            self.norm_stats['input_max'] = np.nanmax(sample_data)

        # Target statistics
        target_sample = []
        n_samples = min(10, self.n_pairs)
        for idx in range(n_samples):
            phase_sample = self.phase_data['phase_corrected'].isel(pair=idx).values
            target_sample.append(phase_sample)

        target_sample = np.array(target_sample)

        self.norm_stats['target_mean'] = np.nanmean(target_sample)
        self.norm_stats['target_std'] = np.nanstd(target_sample)
        self.norm_stats['target_min'] = np.nanmin(target_sample)
        self.norm_stats['target_max'] = np.nanmax(target_sample)

        # Save statistics
        self.save_normalization_stats()

        log.info(f"Normalization statistics computed and saved")

    def save_normalization_stats(self) -> None:
        """Save normalization statistics to file."""
        import json

        stats_path = os.path.join(
            self.config.preprocessed_folder,
            'normalization_stats.json'
        )

        # Convert numpy arrays to lists for JSON serialization
        stats_serializable = {}
        for key, value in self.norm_stats.items():
            if isinstance(value, np.ndarray):
                stats_serializable[key] = value.tolist()
            else:
                stats_serializable[key] = value

        with open(stats_path, 'w') as f:
            json.dump(stats_serializable, f, indent=2)

        log.info(f"Saved normalization stats to {stats_path}")

    def load_normalization_stats(self) -> None:
        """Load normalization statistics from file."""
        import json

        stats_path = os.path.join(
            self.config.preprocessed_folder,
            'normalization_stats.json'
        )

        with open(stats_path, 'r') as f:
            stats_loaded = json.load(f)

        # Convert lists back to numpy arrays
        self.norm_stats = {}
        for key, value in stats_loaded.items():
            if isinstance(value, list):
                self.norm_stats[key] = np.array(value)
            else:
                self.norm_stats[key] = value

        log.info(f"Loaded normalization stats from {stats_path}")


def get_data_loaders(
    config: MLDataConfig,
    seviri_path: str,
    coherence_path: str,
    phase_path: str
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders.

    Args:
        config: MLDataConfig instance
        seviri_path: Path to SEVIRI data
        coherence_path: Path to coherence data
        phase_path: Path to phase data

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Define transforms
    train_transform = None
    if config.random_flip or config.random_rotation or config.noise_std > 0:
        transforms_list = []

        if config.random_flip:
            transforms_list.append(A.HorizontalFlip(p=0.5))
            transforms_list.append(A.VerticalFlip(p=0.5))

        if config.random_rotation > 0:
            transforms_list.append(
                A.Rotate(limit=config.random_rotation, p=0.5)
            )

        if config.noise_std > 0:
            transforms_list.append(
                A.GaussNoise(var_limit=(0, config.noise_std**2), p=0.5)
            )

        train_transform = A.Compose(transforms_list)

    # Create datasets
    train_dataset = AtmosphericCorrectionDataset(
        seviri_path, coherence_path, phase_path,
        config, split='train', transform=train_transform
    )

    val_dataset = AtmosphericCorrectionDataset(
        seviri_path, coherence_path, phase_path,
        config, split='val', transform=None
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    log.info(
        f"Created data loaders:\n"
        f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches\n"
        f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches"
    )

    return train_loader, val_loader
