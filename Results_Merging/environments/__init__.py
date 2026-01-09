"""
Environment handlers for different federated search scenarios.

This module handles:
- Cooperative: Documents returned with scores
- Uncooperative: Documents returned as ranked lists (no scores)
- Uncooperative Weighted: Ranked lists with weighted artificial scores
"""

from .cooperative import CooperativeEnvironment
from .uncooperative import UncooperativeEnvironment
from .uncooperative_weighted import UncooperativeWeightedEnvironment

__all__ = [
    'CooperativeEnvironment',
    'UncooperativeEnvironment',
    'UncooperativeWeightedEnvironment',
]
