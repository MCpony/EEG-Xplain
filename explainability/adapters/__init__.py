"""
模型适配器模块
"""
from .cbramod_adapter import CBraModAdapter
from .labram_adapter import LaBraMAdapter
from .eegpt_adapter import EEGPTAdapter
from .eegmamba_adapter import EEGMambaAdapter
from .biot_adapter import BIOTAdapter

__all__ = [
    'CBraModAdapter',
    'LaBraMAdapter',
    'EEGPTAdapter',
    'EEGMambaAdapter',
    'BIOTAdapter',
]
 