#!/usr/bin/env python3
"""
Step 5: Train Models
====================
Trains LambdaMART and MLP models on extracted features.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from utils import load_config, save_pickle
import lightgbm as lgb
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

config = load_config()

FEATURES_DIR = Path(config['output_dir']) / "features"
MODELS_DIR = Path(config['output_dir']) / "models"
USE_LAMBDAMART = config['use_lambdamart']
USE_MLP = config['use_mlp']

print("=" * 80)
print("Step 5: Train Models")
print("=" * 80)
print()

MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Load training data
print("Loading training data...")
train_df = pd.read_csv(FEATURES_DIR / "features_train.csv")

print(f"Training instances: {len(train_df)}")
print(f"Queries: {train_df['query_id'].nunique()}")

# Prepare features and labels
feature_cols = [col for col in train_df.columns 
                if col not in ['query_id', 'doc_id', 'label', 'max_lex', 'max_sem']]

X_train = train_df[feature_cols].values
y_train = train_df['label'].values
query_ids = train_df['query_id'].values

print(f"Features: {len(feature_cols)}")
print(f"Feature names: {feature_cols}")

# Train LambdaMART
if USE_LAMBDAMART:
    print("\n" + "-" * 80)
    print("Training LambdaMART...")
    print("-" * 80)
    
    # Create query groups (number of documents per query)
    query_groups = train_df.groupby('query_id').size().values
    
    # LambdaMART parameters from config
    lambda_params = config['lambdamart']
    
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_at': [10, 20],
        'num_leaves': lambda_params['num_leaves'],
        'learning_rate': lambda_params['learning_rate'],
        'num_trees': lambda_params['num_trees'],
        'min_data_in_leaf': lambda_params['min_data_in_leaf'],
        'verbose': 1
    }
    
    # Create dataset
    train_data = lgb.Dataset(X_train, label=y_train, group=query_groups)
    
    # Train
    lambdamart_model = lgb.train(
        params,
        train_data,
        num_boost_round=lambda_params['num_trees'],
        valid_sets=[train_data],
        valid_names=['train']
    )
    
    # Save model
    model_file = MODELS_DIR / "lambdamart.pkl"
    save_pickle(lambdamart_model, model_file)
    
    print(f"\nLambdaMART model saved to: {model_file}")

# Train MLP
if USE_MLP:
    print("\n" + "-" * 80)
    print("Training MLP...")
    print("-" * 80)
    
    # MLP parameters from config
    mlp_params = config['mlp']
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Create and train MLP
    mlp_model = MLPRegressor(
        hidden_layer_sizes=tuple(mlp_params['hidden_layers']),
        activation=mlp_params['activation'],
        learning_rate_init=mlp_params['learning_rate'],
        max_iter=mlp_params['epochs'],
        batch_size=mlp_params['batch_size'],
        verbose=True,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42
    )
    
    mlp_model.fit(X_train_scaled, y_train)
    
    # Save model and scaler
    mlp_file = MODELS_DIR / "mlp.pkl"
    scaler_file = MODELS_DIR / "scaler.pkl"
    
    save_pickle(mlp_model, mlp_file)
    save_pickle(scaler, scaler_file)
    
    print(f"\nMLP model saved to: {mlp_file}")
    print(f"Scaler saved to: {scaler_file}")

print("\n" + "=" * 80)
print("Model training complete!")
print("=" * 80)
