# Machine Learning Methods for Results Merging (MLRM) in Patent Retrieval

> **Part of**: [Machine Learning and Neural Methods for Federated Patent Search](../README.md) - PhD Research by Vasileios Stamatis

This repository contains code to reproduce the results from the paper "Machine Learning Methods for Results Merging (MLRM) in Patent Retrieval".

## Overview

This codebase implements:
- **Multiple Models (MMs)**: One ML model per resource
- **Global Models (GMs)**: One ML model for all resources
- **Baseline Methods**: CORI, SSL, SAFE, Centralized, Random
- **Environments**: Cooperative, Uncooperative, Uncooperative Weighted

## Structure

```
Results_Merging/
├── config.py              # Configuration parameters
├── core/                   # Core utilities
│   ├── cori.py            # CORI source selection
│   ├── data_preparation.py # Data preparation functions
│   └── merging_utils.py    # Merging utilities
├── models/                 # ML models
│   └── ml_models.py       # Model creation and training
├── baselines/             # Baseline methods
│   ├── cori_merging.py   # CORI merging
│   ├── ssl.py            # SSL merging
│   ├── safe.py           # SAFE merging
│   ├── centralized.py    # Centralized approach
│   └── random_merging.py # Random baseline
├── environments/          # Environment handlers
│   ├── cooperative.py
│   ├── uncooperative.py
│   └── uncooperative_weighted.py
├── scripts/              # Execution scripts
│   ├── run_experiment.py # Main experiment runner
│   └── example_run.py   # Example usage
└── README.md            # This file
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Paths

Edit `config.py` and set the following paths:

```python
COLLECTION_INDEX = '/path/to/split_3_index/'
REPRESENTATIONS_INDEX = '/path/to/Query_based_sampling_indexes/'
CENTRALIZED_INDEX = '/path/to/centralized_index/'
TOPICS_PATH = '/path/to/300_topics_set3_EN.txt'
RESULTS_PATH = '/path/to/results/'
```

Alternatively, set environment variables:

```bash
export COLLECTION_INDEX=/path/to/split_3_index/
export REPRESENTATIONS_INDEX=/path/to/Query_based_sampling_indexes/
export CENTRALIZED_INDEX=/path/to/centralized_index/
export TOPICS_PATH=/path/to/300_topics_set3_EN.txt
export RESULTS_PATH=/path/to/results/
```

### 3. Prepare Data

1. Download CLEF-IP 2011 dataset from http://www.ifs.tuwien.ac.at/~clef-ip/download/2011/index.shtml
2. Run the [CLEF-IP data preparation pipeline](../CLEF-IP/README.md) to process the dataset and create distributed collections

## Usage

### Running Experiments

Use the main script to run experiments:

```bash
python scripts/run_experiment.py \
    --environment cooperative \
    --method MMs:random_forest \
    --scenario realistic \
    --output results/MMs_random_forest_cooperative.res
```

### Parameters

- `--environment`: `cooperative`, `uncooperative`, or `uncooperative_weighted`
- `--method`: 
  - Baselines: `cori`, `ssl`, `safe`, `centralized`, `random`
  - ML Models: `MMs:random_forest`, `MMs:decision_tree`, `MMs:svr`, `MMs:linear_regression`, `MMs:polynomial_x2`, `MMs:polynomial_x3`
  - Global Models: `GMs:random_forest`, `GMs:decision_tree`, `GMs:svr`, `GMs:linear_regression`, `GMs:dnn`
- `--scenario`: `realistic`, `optimal`, or `random`
- `--output`: Path to output file

### Examples

**MMs Random Forest (Best Model) - Cooperative:**
```bash
python scripts/run_experiment.py \
    --environment cooperative \
    --method MMs:random_forest \
    --output results/MMs_rf_coop.res
```

**GMs DNN - Cooperative:**
```bash
python scripts/run_experiment.py \
    --environment cooperative \
    --method GMs:dnn \
    --output results/GMs_dnn_coop.res
```

**MMs Random Forest - Uncooperative Weighted (Best from Paper):**
```bash
python scripts/run_experiment.py \
    --environment uncooperative_weighted \
    --method MMs:random_forest \
    --output results/MMs_rf_uncoop_weighted.res
```

**Baseline SSL:**
```bash
python scripts/run_experiment.py \
    --environment cooperative \
    --method ssl \
    --output results/SSL_coop.res
```

**Baseline CORI:**
```bash
python scripts/run_experiment.py \
    --environment cooperative \
    --method cori \
    --output results/CORI_coop.res
```

**Baseline SAFE:**
```bash
python scripts/run_experiment.py \
    --environment uncooperative_weighted \
    --method safe \
    --output results/SAFE_uncoop_weighted.res
```

## Methods

### Multiple Models (MMs)
- One model per resource
- Models: Random Forest, Decision Tree, SVR, Linear Regression, Polynomial (x², x³)
- Best performance: Random Forest

### Global Models (GMs)
- One model for all resources
- Models: Random Forest, Decision Tree, SVR, Linear Regression, DNN
- Best performance: DNN (MAP), Random Forest (PRES/RECALL)

### Environments

**Cooperative:**
- Documents returned with scores from remote engines

**Uncooperative:**
- Documents returned as ranked lists (no scores)
- Artificial scores assigned linearly: 0.6 (first) to 0.4 (last)

**Uncooperative Weighted:**
- Documents returned as ranked lists (no scores)
- Artificial scores weighted by CORI source selection score

## Results Format

Results are written in TREC format:
```
topic_id Q0 doc_id rank score run_id
```

Example:
```
EP-1310580-A2 Q0 EP-1310580-A2 1 0.1234 MMs:random_forest
```

## Evaluation

Use standard TREC evaluation tools (trec_eval) to evaluate results:

```bash
trec_eval -m map -m recall -m pres qrels_file results_file
```

## Configuration

Key parameters in `config.py`:

- `NUM_COLLECTIONS_TO_QUERY = 20`: Number of top collections to query
- `RESULTS_PER_COLLECTION = 100`: Results per collection
- `CENTRALIZED_RESULTS = 1000`: Results from centralized index
- `FINAL_RESULTS = 100`: Final merged results
- `RANDOM_FOREST_N_ESTIMATORS = 100`: Number of trees
- `DNN_EPOCHS = 50`: DNN training epochs

## Implementation Details

### Data Preparation
- Training dataframes use overlapped documents between resources and centralized index
- Prediction dataframes use non-overlapped documents
- For MMs, one model is trained per collection using overlapped documents
- For GMs, one model is trained for all collections using all overlapped documents

### Model Training
- MMs: Linear regression uses first 10 overlapped documents per collection
- GMs: All overlapped documents are used for training
- DNN architecture: 4 hidden layers (632 -> 300 -> 150 -> 50 -> 1)
- Random Forest uses 100 trees by default

### Score Assignment (Uncooperative Environments)
- Linear: Scores assigned from 0.6 (first document) to 0.4 (last document)
- Weighted: Linear scores multiplied by CORI source selection score

### SAFE Implementation
- Requires both sample and full index readers
- Maps document ranks to centralized scores using linear regression
- Estimates ranks for documents not in sample collections
