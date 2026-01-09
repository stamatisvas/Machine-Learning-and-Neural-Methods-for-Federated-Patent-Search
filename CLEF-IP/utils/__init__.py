"""
Utility functions for CLEF-IP data preparation.
"""

from .xml_utils import XMLCombiner, hashabledict
from .text_utils import clean_text, extract_ipc_codes

__all__ = [
    'XMLCombiner',
    'hashabledict',
    'clean_text',
    'extract_ipc_codes',
]
