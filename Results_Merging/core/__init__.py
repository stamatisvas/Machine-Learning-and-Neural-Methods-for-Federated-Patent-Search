"""
Core utilities for MLRM experiments.

This module contains core functions for:
- CORI source selection
- Data preparation and transformation
- Result merging utilities
"""

from .cori import CORI, CORI2, CORI2_for_CORI, get_avg_cw, get_number_of_collections_contain_word
from .data_preparation import (
    dataframe_for_training,
    dataframe_for_predictions,
    dataframe_for_training_SSL,
    dataframe_for_predictions_SSL,
    dataframe_for_training_polynomial_x2_SSL,
    dataframe_for_predictions_polynomial_x2_SSL,
    dataframe_for_training_polynomial_x3_SSL,
    dataframe_for_predictions_polynomial_x3_SSL,
)
from .merging_utils import (
    remove_duplicates,
    final_merged_list_for_SSL,
    final_merged_list_for_simple_machine_learning,
    final_merged_list_for_deep_learning,
)

__all__ = [
    'CORI',
    'CORI2',
    'CORI2_for_CORI',
    'get_avg_cw',
    'get_number_of_collections_contain_word',
    'dataframe_for_training',
    'dataframe_for_predictions',
    'dataframe_for_training_SSL',
    'dataframe_for_predictions_SSL',
    'dataframe_for_training_polynomial_x2_SSL',
    'dataframe_for_predictions_polynomial_x2_SSL',
    'dataframe_for_training_polynomial_x3_SSL',
    'dataframe_for_predictions_polynomial_x3_SSL',
    'remove_duplicates',
    'final_merged_list_for_SSL',
    'final_merged_list_for_simple_machine_learning',
    'final_merged_list_for_deep_learning',
]
