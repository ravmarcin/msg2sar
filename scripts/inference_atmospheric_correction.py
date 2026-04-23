#!/usr/bin/env python3
"""
Atmospheric Correction Inference Script

Applies trained model to correct interferograms for atmospheric effects.

Usage:
    python scripts/inference_atmospheric_correction.py \
        --config data/configs/ml/atmospheric_correction/bogo_pl_master.json \
        --checkpoint checkpoints/best_model.pth \
        --input path/to/input_data.nc \
        --output path/to/corrected_output.nc
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from local_setup import local_setup
    local_setup()
except ModuleNotFoundError:
    from utils.internal.sbas.local_setup import local_setup
    local_setup()

import torch
import numpy as np
import xarray as xr
from datetime import datetime
from utils.internal.ml import (
    MLDataConfig,
    AtmosphericCorrectionUNet
)
from utils.internal.log.logger import get_logger

log = get_logger()


def load_model(checkpoint_path: str, config: MLDataConfig, device: torch.device):
    """
    Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        config: MLDataConfig instance
        device: Device to load model on

    Returns:
        Loaded model
    """
    log.info(f"Loading model from: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get model config from checkpoint
    model_config = checkpoint.get('config', {})

    # Create model
    model = AtmosphericCorrectionUNet(
        in_channels=model_config.get('in_channels', config.in_channels),
        out_channels=model_config.get('out_channels', config.out_channels),
        init_features=model_config.get('init_features', config.init_features),
        input_size=model_config.get('input_size', config.model_input_size),
        output_size=model_config.get('output_size', config.model_output_size)
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    log.info(f"✓ Model loaded (epoch {checkpoint['epoch']}, val_loss: {checkpoint['best_val_loss']:.6f})")

    return model


def normalize_input(data: np.ndarray, stats: dict, method: str = 'standardize') -> np.ndarray:
    """
    Normalize input data using saved statistics.

    Args:
        data: Input array (C, H, W)
        stats: Normalization statistics dictionary
        method: Normalization method

    Returns:
        Normalized array
    """
    if method == 'standardize':
        mean = np.array(stats['input_mean'])
        std = np.array(stats['input_std'])

        # Reshape for broadcasting
        mean = mean[:, None, None]
        std = std[:, None, None]

        normalized = (data - mean) / (std + 1e-8)

    elif method == 'minmax':
        min_val = np.array(stats['input_min'])
        max_val = np.array(stats['input_max'])

        min_val = min_val[:, None, None]
        max_val = max_val[:, None, None]

        normalized = (data - min_val) / (max_val - min_val + 1e-8)

    else:
        log.warning(f"Unknown normalization method '{method}', using raw data")
        normalized = data

    return normalized


def denormalize_output(data: np.ndarray, stats: dict, method: str = 'standardize') -> np.ndarray:
    """
    Denormalize output data using saved statistics.

    Args:
        data: Normalized output array
        stats: Normalization statistics dictionary
        method: Normalization method

    Returns:
        Denormalized array
    """
    if method == 'standardize':
        mean = stats['target_mean']
        std = stats['target_std']
        denormalized = data * std + mean

    elif method == 'minmax':
        min_val = stats['target_min']
        max_val = stats['target_max']
        denormalized = data * (max_val - min_val) + min_val

    else:
        denormalized = data

    return denormalized


def run_inference(
    config_path: str,
    checkpoint_path: str,
    input_path: str,
    output_path: str,
    device: str = None,
    batch_size: int = 8
):
    """
    Run inference on input data.

    Args:
        config_path: Path to master config
        checkpoint_path: Path to model checkpoint
        input_path: Path to input data
        output_path: Path to save output
        device: Device to use (auto-detect if None)
        batch_size: Batch size for inference
    """
    log.info("="*80)
    log.info("Atmospheric Correction Inference")
    log.info("="*80)

    # Load config
    config = MLDataConfig(config_path)

    log.info(f"Job: {config.job_name}")
    log.info(f"Input: {input_path}")
    log.info(f"Output: {output_path}")

    # Setup device
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            log.info("Using Apple Silicon (MPS)")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            log.info("Using CUDA GPU")
        else:
            device = torch.device("cpu")
            log.info("Using CPU")
    else:
        device = torch.device(device)

    # Load model
    model = load_model(checkpoint_path, config, device)

    # Load normalization statistics
    stats_path = os.path.join(config.preprocessed_folder, 'normalization_stats.json')
    if os.path.exists(stats_path):
        import json
        with open(stats_path, 'r') as f:
            norm_stats = json.load(f)
        log.info(f"✓ Loaded normalization statistics from: {stats_path}")
    else:
        log.warning("No normalization statistics found, using raw data")
        norm_stats = None

    # Load input data
    log.info("\nLoading input data...")
    try:
        input_data = xr.open_dataset(input_path)
        log.info(f"✓ Input data loaded")

        # Extract required data
        # Expecting: seviri_data (14 channels: SEVIRI t1 + t2 + coherence)
        if 'seviri_data' in input_data:
            seviri = input_data['seviri_data'].values
        else:
            log.error("Input data must contain 'seviri_data' variable")
            return

        n_samples = seviri.shape[0]
        log.info(f"  Number of samples: {n_samples}")
        log.info(f"  Data shape: {seviri.shape}")

    except Exception as e:
        log.error(f"Error loading input data: {e}")
        return

    # Run inference
    log.info("\nRunning inference...")
    predictions = []

    model.eval()
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            batch_data = seviri[i:batch_end]

            # Normalize
            if norm_stats is not None:
                batch_normalized = np.stack([
                    normalize_input(batch_data[j], norm_stats, config.normalization_method)
                    for j in range(len(batch_data))
                ])
            else:
                batch_normalized = batch_data

            # Convert to tensor
            batch_tensor = torch.from_numpy(batch_normalized).float().to(device)

            # Forward pass
            output = model(batch_tensor)

            # Move to CPU and convert to numpy
            output_np = output.cpu().numpy()

            # Denormalize
            if norm_stats is not None:
                output_denorm = np.stack([
                    denormalize_output(output_np[j], norm_stats, config.normalization_method)
                    for j in range(len(output_np))
                ])
            else:
                output_denorm = output_np

            predictions.append(output_denorm)

            log.info(f"  Processed {batch_end}/{n_samples} samples")

    # Concatenate predictions
    predictions = np.concatenate(predictions, axis=0)
    log.info(f"✓ Inference complete: {predictions.shape}")

    # Save output
    log.info(f"\nSaving output to: {output_path}")

    # Create output dataset
    output_dataset = xr.Dataset(
        {
            'atmospheric_correction': xr.DataArray(
                predictions.squeeze(),  # Remove channel dimension if output is (N, 1, H, W)
                dims=['pair', 'y', 'x'],
                attrs={
                    'units': 'radians',
                    'long_name': 'Predicted atmospheric phase correction',
                    'model_checkpoint': checkpoint_path,
                    'inference_date': datetime.now().isoformat()
                }
            )
        },
        coords=input_data.coords
    )

    output_dataset.to_netcdf(output_path)
    log.info("✓ Output saved")

    # Print statistics
    log.info("\n" + "="*80)
    log.info("Inference Statistics")
    log.info("="*80)
    log.info(f"Samples processed: {n_samples}")
    log.info(f"Output shape: {predictions.shape}")
    log.info(f"Mean correction: {np.nanmean(predictions):.6f} rad")
    log.info(f"Std correction: {np.nanstd(predictions):.6f} rad")
    log.info(f"Min correction: {np.nanmin(predictions):.6f} rad")
    log.info(f"Max correction: {np.nanmax(predictions):.6f} rad")


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with trained atmospheric correction model"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to master config file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input data (NetCDF)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to save output (NetCDF)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['mps', 'cuda', 'cpu'],
        help='Force specific device (auto-detect if not specified)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for inference (default: 8)'
    )

    args = parser.parse_args()

    # Run inference
    run_inference(
        args.config,
        args.checkpoint,
        args.input,
        args.output,
        args.device,
        args.batch_size
    )


if __name__ == "__main__":
    main()
