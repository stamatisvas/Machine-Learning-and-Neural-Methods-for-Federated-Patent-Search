"""
Baseline methods for results merging.

This module contains implementations of baseline methods:
- CORI: Collection Inference Retrieval Network
- SSL: Semi-Supervised Learning
- SAFE: Sample-Agglomerate Fitting Estimate
- Centralized: Single centralized index approach
- Random: Random merging baseline
"""

from .cori_merging import merge_cori
from .ssl import merge_ssl
from .safe import merge_safe
from .centralized import merge_centralized
from .random_merging import merge_random

__all__ = [
    'merge_cori',
    'merge_ssl',
    'merge_safe',
    'merge_centralized',
    'merge_random',
]
