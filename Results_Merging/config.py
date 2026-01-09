"""
Configuration module for MLRM (Machine Learning Results Merging) experiments.

This module contains all configuration parameters needed to run the experiments
as described in the paper "Machine Learning Methods for Results Merging (MLRM) 
in Patent Retrieval".
"""

import os
from pathlib import Path

# Base paths - Update these according to your setup
BASE_DIR = Path(__file__).parent

# Data paths
COLLECTION_INDEX = os.getenv('COLLECTION_INDEX', '/path/to/split_3_index/')
REPRESENTATIONS_INDEX = os.getenv('REPRESENTATIONS_INDEX', '/path/to/Query_based_sampling_indexes/')
CENTRALIZED_INDEX = os.getenv('CENTRALIZED_INDEX', '/path/to/centralized_index/')
TOPICS_PATH = os.getenv('TOPICS_PATH', '/path/to/300_topics_set3_EN.txt')
RESULTS_PATH = os.getenv('RESULTS_PATH', BASE_DIR / 'results/')

# Experiment parameters
NUM_COLLECTIONS_TO_QUERY = 20  # Number of top collections to query (as per paper)
RESULTS_PER_COLLECTION = 100  # Number of results to retrieve per collection
CENTRALIZED_RESULTS = 1000  # Number of results from centralized index
FINAL_RESULTS = 100  # Final number of merged results to return

# CORI parameters
CORI_B = 0.4  # CORI smoothing parameter

# Artificial scoring parameters (for uncooperative environments)
ARTIFICIAL_SCORE_FIRST = 0.6  # Score for first document
ARTIFICIAL_SCORE_LAST = 0.4   # Score for last document

# ML Model parameters
RANDOM_FOREST_N_ESTIMATORS = 100  # Number of trees in random forest
DNN_LEARNING_RATE = 0.01
DNN_BATCH_SIZE = 100
DNN_EPOCHS = 50
DNN_HIDDEN_LAYERS = [632, 300, 150, 50]  # Architecture for sequential DNN

# Environment types
ENV_COOPERATIVE = 'cooperative'
ENV_UNCOOPERATIVE = 'uncooperative'
ENV_UNCOOPERATIVE_WEIGHTED = 'uncooperative_weighted'

# Scenario types
SCENARIO_REALISTIC = 'realistic'
SCENARIO_OPTIMAL = 'optimal'
SCENARIO_RANDOM = 'random'

# Model types
MODEL_TYPE_MM = 'multiple_models'  # One model per resource
MODEL_TYPE_GM = 'global_models'     # One model for all resources

# ML Models
ML_MODEL_RANDOM_FOREST = 'random_forest'
ML_MODEL_DECISION_TREE = 'decision_tree'
ML_MODEL_SVR = 'svr'
ML_MODEL_LINEAR_REGRESSION = 'linear_regression'
ML_MODEL_POLYNOMIAL_X2 = 'polynomial_x2'
ML_MODEL_POLYNOMIAL_X3 = 'polynomial_x3'
ML_MODEL_DNN = 'dnn'

# Baseline methods
BASELINE_CORI = 'cori'
BASELINE_SSL = 'ssl'
BASELINE_SAFE = 'safe'
BASELINE_CENTRALIZED = 'centralized'
BASELINE_RANDOM = 'random'

# Create results directory if it doesn't exist
RESULTS_PATH.mkdir(parents=True, exist_ok=True)
