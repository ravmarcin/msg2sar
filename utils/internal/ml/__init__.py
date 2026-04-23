"""
Machine Learning module for atmospheric correction.

This module provides PyTorch data loaders, models, and training infrastructure
for ML-based atmospheric phase correction in InSAR processing.
"""

from utils.internal.ml.data_config import MLDataConfig
from utils.internal.ml.data_loader import AtmosphericCorrectionDataset, get_data_loaders
from utils.internal.ml.trainer import AtmosphericCorrectionTrainer
from utils.internal.ml.models.unet import AtmosphericCorrectionUNet

__all__ = [
    'MLDataConfig',
    'AtmosphericCorrectionDataset',
    'get_data_loaders',
    'AtmosphericCorrectionTrainer',
    'AtmosphericCorrectionUNet'
]
