#!/usr/bin/env python3
"""
ML Data Preparation Script

Prepares training and validation data for atmospheric correction model.
This script:
1. Loads SEVIRI temporal stacks
2. Loads SAR coherence maps
3. Loads GACOS-corrected phase data
4. Splits into train/validation sets
5. Computes normalization statistics
6. Saves preprocessed data

Usage:
    python scripts/prepare_ml_data.py --config data/configs/ml/atmospheric_correction/bogo_pl_master.json
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

import numpy as np
import xarray as xr
from datetime import datetime
from utils.internal.ml.data_config import MLDataConfig
from utils.internal.log.logger import get_logger

log = get_logger()


def prepare_data(config_path: str):
    """
    Prepare ML training data.

    Args:
        config_path: Path to master config file
    """
    log.info("="*80)
    log.info("ML Data Preparation")
    log.info("="*80)

    # Load config
    config = MLDataConfig(config_path)

    log.info(f"Job: {config.job_name}")
    log.info(f"Description: {config.description}")

    # Define data paths (these would come from the pipeline)
    # For now, using placeholders that need to be updated based on actual data structure
    seviri_path = os.path.join(config.data_dir, "seviri_temporal_stack.nc")
    coherence_path = os.path.join(config.data_dir, "coherence_maps.nc")
    phase_path = os.path.join(config.data_dir, "gacos_corrected_phase.nc")

    # Check if files exist
    missing_files = []
    for path, name in [(seviri_path, "SEVIRI"), (coherence_path, "Coherence"), (phase_path, "Phase")]:
        if not os.path.exists(path):
            missing_files.append(f"{name}: {path}")

    if missing_files:
        log.error("Missing required data files:")
        for missing in missing_files:
            log.error(f"  - {missing}")
        log.error("\nPlease run the following steps first:")
        log.error("  1. Process SEVIRI temporal stack")
        log.error("  2. Extract SAR coherence maps")
        log.error("  3. Apply GACOS corrections")
        return

    log.info("Loading data files...")

    # Load datasets
    try:
        seviri_data = xr.open_dataset(seviri_path)
        log.info(f"✓ SEVIRI data loaded: {seviri_path}")
        log.info(f"  Shape: {seviri_data['seviri_data'].shape}")

        coherence_data = xr.open_dataset(coherence_path)
        log.info(f"✓ Coherence data loaded: {coherence_path}")
        log.info(f"  Shape: {coherence_data['coherence'].shape}")

        phase_data = xr.open_dataset(phase_path)
        log.info(f"✓ Phase data loaded: {phase_path}")
        log.info(f"  Shape: {phase_data['phase_corrected'].shape}")

    except Exception as e:
        log.error(f"Error loading data: {e}")
        return

    # Split into train/validation
    n_samples = len(seviri_data.pair)
    n_train = int(n_samples * config.train_val_split)

    # Set random seed for reproducibility
    np.random.seed(config.seed)
    indices = np.random.permutation(n_samples)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    log.info(f"\nData split:")
    log.info(f"  Total samples: {n_samples}")
    log.info(f"  Training: {len(train_indices)} ({config.train_val_split*100:.0f}%)")
    log.info(f"  Validation: {len(val_indices)} ({(1-config.train_val_split)*100:.0f}%)")

    # Create train/validation directories
    os.makedirs(config.training_folder, exist_ok=True)
    os.makedirs(config.validation_folder, exist_ok=True)

    # Save split datasets
    log.info("\nSaving split datasets...")

    # Training data
    seviri_train = seviri_data.isel(pair=train_indices)
    coherence_train = coherence_data.isel(pair=train_indices)
    phase_train = phase_data.isel(pair=train_indices)

    seviri_train.to_netcdf(os.path.join(config.training_folder, "seviri_temporal_stack.nc"))
    coherence_train.to_netcdf(os.path.join(config.training_folder, "coherence_maps.nc"))
    phase_train.to_netcdf(os.path.join(config.training_folder, "gacos_corrected_phase.nc"))

    log.info(f"✓ Training data saved to: {config.training_folder}")

    # Validation data
    seviri_val = seviri_data.isel(pair=val_indices)
    coherence_val = coherence_data.isel(pair=val_indices)
    phase_val = phase_data.isel(pair=val_indices)

    seviri_val.to_netcdf(os.path.join(config.validation_folder, "seviri_temporal_stack.nc"))
    coherence_val.to_netcdf(os.path.join(config.validation_folder, "coherence_maps.nc"))
    phase_val.to_netcdf(os.path.join(config.validation_folder, "gacos_corrected_phase.nc"))

    log.info(f"✓ Validation data saved to: {config.validation_folder}")

    # Compute and save statistics
    log.info("\nComputing normalization statistics...")

    # This will be computed by the DataLoader, but we can preview here
    seviri_sample = seviri_train['seviri_data'].isel(pair=slice(0, min(10, len(train_indices)))).values
    log.info(f"  SEVIRI mean: {np.nanmean(seviri_sample):.4f}")
    log.info(f"  SEVIRI std: {np.nanstd(seviri_sample):.4f}")

    phase_sample = phase_train['phase_corrected'].isel(pair=slice(0, min(10, len(train_indices)))).values
    log.info(f"  Phase mean: {np.nanmean(phase_sample):.4f}")
    log.info(f"  Phase std: {np.nanstd(phase_sample):.4f}")

    log.info("\n" + "="*80)
    log.info("Data preparation complete!")
    log.info("="*80)
    log.info("\nNext steps:")
    log.info("  1. Review the data splits")
    log.info("  2. Run training: python scripts/train_atmospheric_correction.py")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare ML training data for atmospheric correction"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to master config file'
    )

    args = parser.parse_args()

    # Run data preparation
    prepare_data(args.config)


if __name__ == "__main__":
    main()
