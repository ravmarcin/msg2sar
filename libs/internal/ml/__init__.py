"""
Machine Learning module for atmospheric correction.

This module provides PyTorch data loaders, models, and training infrastructure
for ML-based atmospheric phase correction in InSAR processing.
"""

from libs.internal.ml.data_config import MLDataConfig
from libs.internal.ml.data_loader import AtmosphericCorrectionDataset, get_data_loaders
from libs.internal.ml.trainer import AtmosphericCorrectionTrainer
from libs.internal.ml.models.unet import AtmosphericCorrectionUNet

__all__ = [
    'MLDataConfig',
    'AtmosphericCorrectionDataset',
    'get_data_loaders',
    'AtmosphericCorrectionTrainer',
    'AtmosphericCorrectionUNet'
]
