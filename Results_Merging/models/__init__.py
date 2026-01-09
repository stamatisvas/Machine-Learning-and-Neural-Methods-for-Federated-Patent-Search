"""
Machine Learning models for Results Merging.

This module contains implementations of ML models used in the MLRM paper:
- Multiple Models (MMs): One model per resource
- Global Models (GMs): One model for all resources
"""

from .ml_models import (
    train_ml_model,
    create_random_forest,
    create_decision_tree,
    create_svr,
    create_linear_regression,
    create_polynomial_x2,
    create_polynomial_x3,
    create_dnn,
)

__all__ = [
    'train_ml_model',
    'create_random_forest',
    'create_decision_tree',
    'create_svr',
    'create_linear_regression',
    'create_polynomial_x2',
    'create_polynomial_x3',
    'create_dnn',
]
