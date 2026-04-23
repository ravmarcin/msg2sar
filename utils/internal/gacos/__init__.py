"""
GACOS atmospheric correction module.

This module provides tools to download and apply GACOS (Generic Atmospheric
Correction Online Service) tropospheric correction products for InSAR processing.
"""

from utils.internal.gacos.gacos_config import GacosConfig
from utils.internal.gacos.gacos_processor import GacosProcessor

__all__ = [
    'GacosConfig',
    'GacosProcessor'
]
