#!/usr/bin/env python3
"""
Atmospheric Correction Model Training Script

Trains a UNet model for atmospheric phase correction using SEVIRI data.

Optimized for:
- Python 3.12
- Apple Silicon (Mac M chips) with MPS backend
- CUDA GPUs (if available)
- CPU fallback

Usage:
    python scripts/train_atmospheric_correction.py --config data/configs/ml/atmospheric_correction/bogo_pl_master.json

Optional arguments:
    --resume CHECKPOINT    Resume training from checkpoint
    --device DEVICE        Force specific device (mps/cuda/cpu)
    --test-only           Only test model without training
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
import json
from datetime import datetime
from utils.internal.ml import (
    MLDataConfig,
    get_data_loaders,
    AtmosphericCorrectionUNet,
    AtmosphericCorrectionTrainer
)
from utils.internal.log.logger import get_logger

log = get_logger()


def test_model_only(config: MLDataConfig, device: str = None):
    """
    Test model architecture without training.

    Args:
        config: MLDataConfig instance
        device: Device to use
    """
    log.info("="*80)
    log.info("Model Architecture Test")
    log.info("="*80)

    # Create model
    model = AtmosphericCorrectionUNet(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        init_features=config.init_features,
        input_size=config.model_input_size,
        output_size=config.model_output_size
    )

    # Setup device
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    log.info(f"Using device: {device}")

    model = model.to(device)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"\nModel architecture:")
    log.info(f"  Total parameters: {total_params:,}")
    log.info(f"  Input: [{config.in_channels}, {config.model_input_size}, {config.model_input_size}]")
    log.info(f"  Output: [{config.out_channels}, {config.model_output_size}, {config.model_output_size}]")

    # Test forward pass
    log.info("\nTesting forward pass...")
    dummy_input = torch.randn(2, config.in_channels, config.model_input_size, config.model_input_size).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    log.info(f"✓ Forward pass successful!")
    log.info(f"  Input shape: {dummy_input.shape}")
    log.info(f"  Output shape: {output.shape}")

    # Test backward pass
    log.info("\nTesting backward pass...")
    criterion = torch.nn.MSELoss()
    target = torch.randn_like(output)
    loss = criterion(output, target)
    loss.backward()

    log.info(f"✓ Backward pass successful!")
    log.info(f"  Loss: {loss.item():.6f}")

    log.info("\n" + "="*80)
    log.info("Model test completed successfully!")
    log.info("="*80)


def train_model(config_path: str, resume_checkpoint: str = None, device: str = None):
    """
    Train atmospheric correction model.

    Args:
        config_path: Path to master config file
        resume_checkpoint: Optional checkpoint to resume from
        device: Optional device override (mps/cuda/cpu)
    """
    log.info("="*80)
    log.info("Atmospheric Correction Model Training")
    log.info("="*80)

    # Load config
    config = MLDataConfig(config_path)

    log.info(f"Job: {config.job_name}")
    log.info(f"Description: {config.description}")
    log.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Define data paths
    seviri_train_path = os.path.join(config.training_folder, "seviri_temporal_stack.nc")
    coherence_train_path = os.path.join(config.training_folder, "coherence_maps.nc")
    phase_train_path = os.path.join(config.training_folder, "gacos_corrected_phase.nc")

    seviri_val_path = os.path.join(config.validation_folder, "seviri_temporal_stack.nc")
    coherence_val_path = os.path.join(config.validation_folder, "coherence_maps.nc")
    phase_val_path = os.path.join(config.validation_folder, "gacos_corrected_phase.nc")

    # Check if preprocessed data exists
    required_files = [
        seviri_train_path, coherence_train_path, phase_train_path,
        seviri_val_path, coherence_val_path, phase_val_path
    ]

    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        log.error("Missing preprocessed data files:")
        for f in missing_files:
            log.error(f"  - {f}")
        log.error("\nPlease run data preparation first:")
        log.error("  python scripts/prepare_ml_data.py --config <config_path>")
        return

    log.info("\nLoading data...")

    # Create data loaders
    try:
        train_loader, val_loader = get_data_loaders(
            config,
            seviri_train_path,
            coherence_train_path,
            phase_train_path
        )
        log.info("✓ Data loaders created successfully")

    except Exception as e:
        log.error(f"Error creating data loaders: {e}")
        import traceback
        traceback.print_exc()
        return

    # Create model
    log.info("\nInitializing model...")
    model = AtmosphericCorrectionUNet(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        init_features=config.init_features,
        input_size=config.model_input_size,
        output_size=config.model_output_size
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"✓ Model created with {total_params:,} parameters")

    # Create trainer
    log.info("\nInitializing trainer...")
    trainer = AtmosphericCorrectionTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )

    # Resume from checkpoint if specified
    if resume_checkpoint:
        log.info(f"\nResuming from checkpoint: {resume_checkpoint}")
        trainer.load_checkpoint(resume_checkpoint)

    # Train model
    log.info("\n" + "="*80)
    log.info("Starting training...")
    log.info("="*80 + "\n")

    try:
        history = trainer.train()

        # Save training history
        history_path = os.path.join(config.checkpoint_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        log.info(f"\n✓ Training history saved to: {history_path}")

    except KeyboardInterrupt:
        log.warning("\nTraining interrupted by user")
        # Save checkpoint on interrupt
        trainer.save_checkpoint('interrupted_checkpoint.pth')
        log.info("Checkpoint saved. You can resume with --resume interrupted_checkpoint.pth")
        return

    except Exception as e:
        log.error(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        return

    # Print summary
    log.info("\n" + "="*80)
    log.info("Training Summary")
    log.info("="*80)
    log.info(f"Best validation loss: {trainer.best_val_loss:.6f}")
    log.info(f"Final epoch: {trainer.current_epoch}")
    log.info(f"Checkpoints saved to: {config.checkpoint_dir}")
    log.info(f"TensorBoard logs: {config.log_dir}")
    log.info("\nTo visualize training:")
    log.info(f"  tensorboard --logdir {config.log_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Train atmospheric correction model"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to master config file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Resume training from checkpoint'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['mps', 'cuda', 'cpu'],
        help='Force specific device (auto-detect if not specified)'
    )
    parser.add_argument(
        '--test-only',
        action='store_true',
        help='Only test model architecture without training'
    )

    args = parser.parse_args()

    if args.test_only:
        # Load config and test model
        config = MLDataConfig(args.config)
        test_model_only(config, args.device)
    else:
        # Run training
        train_model(args.config, args.resume, args.device)


if __name__ == "__main__":
    main()
