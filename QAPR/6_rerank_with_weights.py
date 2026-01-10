#!/usr/bin/env python3
"""
Step 6: Re-rank with Query Weights
===================================
Applies trained models and query-specific weights for final re-ranking.
Implements Section 3.3 from the paper: Query Weights and Final Score.

Final Score = CombinedScore + α * MaxLex + (1-α) * MaxSem
"""

import pandas as pd
import numpy as np
from pathlib import Path
from utils import (load_config, load_pickle, calculate_query_alpha, 
                   write_trec_run)
from tqdm import tqdm

config = load_config()

FEATURES_DIR = Path(config['output_dir']) / "features"
MODELS_DIR = Path(config['output_dir']) / "models"
SPLITS_DIR = Path(config['output_dir']) / "splits"
RESULTS_DIR = Path(config['output_dir']) / "results"
USE_LAMBDAMART = config['use_lambdamart']
USE_MLP = config['use_mlp']

print("=" * 80)
print("Step 6: Re-rank with Query Weights")
print("=" * 80)
print()

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load test data
print("Loading test data...")
test_df = pd.read_csv(FEATURES_DIR / "features_test.csv")

print(f"Test instances: {len(test_df)}")
print(f"Test queries: {test_df['query_id'].nunique()}")

# Load IDF and topic splits for alpha calculation
print("Loading IDF dictionary and topic splits...")
idf_dict = load_pickle(SPLITS_DIR / "idf_dict.pkl")
topic_splits = load_pickle(SPLITS_DIR / "topic_splits.pkl")

# Calculate average IDF for all documents (for alpha calculation)
print("Calculating document IDF statistics...")
doc_splits = load_pickle(SPLITS_DIR / "document_splits.pkl")

all_doc_idfs = []
for doc_id, doc in doc_splits.items():
    full_text = f"{doc['abstract']} {doc['description']} {doc['claims']}"
    words = full_text.lower().split()
    idfs = [idf_dict.get(word, 0) for word in words]
    avg_idf = np.mean(idfs) if idfs else 0
    all_doc_idfs.append(avg_idf)

print(f"Average document IDF: {np.mean(all_doc_idfs):.4f}")

# Prepare features
feature_cols = [col for col in test_df.columns 
                if col not in ['query_id', 'doc_id', 'label', 'max_lex', 'max_sem']]

X_test = test_df[feature_cols].values


# Re-rank with LambdaMART
if USE_LAMBDAMART:
    print("\n" + "-" * 80)
    print("Re-ranking with LambdaMART + Query Weights...")
    print("-" * 80)
    
    # Load model
    lambdamart_model = load_pickle(MODELS_DIR / "lambdamart.pkl")
    
    # Predict combined scores
    print("Predicting combined scores...")
    combined_scores = lambdamart_model.predict(X_test)
    test_df['combined_score'] = combined_scores
    
    # Calculate final scores with query weights
    print("Calculating final scores with query-specific alpha...")
    
    final_results = {}
    
    for query_id, group in tqdm(test_df.groupby('query_id'), desc="Processing queries"):
        # Get query text
        if query_id not in topic_splits:
            continue
        
        topic = topic_splits[query_id]
        query_text = f"{topic['abstract']} {topic['description']} {topic['claims']}"
        
        # Calculate alpha for this query
        alpha = calculate_query_alpha(query_text, idf_dict, all_doc_idfs)
        
        # Calculate final scores
        final_scores = []
        for _, row in group.iterrows():
            doc_id = row['doc_id']
            combined = row['combined_score']
            max_lex = row['max_lex']
            max_sem = row['max_sem']
            
            # Final Score = CombinedScore + α * MaxLex + (1-α) * MaxSem
            final_score = combined + alpha * max_lex + (1 - alpha) * max_sem
            
            final_scores.append((doc_id, final_score))
        
        final_results[query_id] = final_scores
    
    # Write TREC run file
    output_file = RESULTS_DIR / "lambdamart_ranking.txt"
    write_trec_run(final_results, output_file, run_name="LambdaMART")
    
    print(f"Results written to: {output_file}")


# Re-rank with MLP
if USE_MLP:
    print("\n" + "-" * 80)
    print("Re-ranking with MLP + Query Weights...")
    print("-" * 80)
    
    # Load model and scaler
    mlp_model = load_pickle(MODELS_DIR / "mlp.pkl")
    scaler = load_pickle(MODELS_DIR / "scaler.pkl")
    
    # Scale features
    X_test_scaled = scaler.transform(X_test)
    
    # Predict combined scores
    print("Predicting combined scores...")
    combined_scores = mlp_model.predict(X_test_scaled)
    test_df['combined_score_mlp'] = combined_scores
    
    # Calculate final scores with query weights
    print("Calculating final scores with query-specific alpha...")
    
    final_results = {}
    
    for query_id, group in tqdm(test_df.groupby('query_id'), desc="Processing queries"):
        # Get query text
        if query_id not in topic_splits:
            continue
        
        topic = topic_splits[query_id]
        query_text = f"{topic['abstract']} {topic['description']} {topic['claims']}"
        
        # Calculate alpha for this query
        alpha = calculate_query_alpha(query_text, idf_dict, all_doc_idfs)
        
        # Calculate final scores
        final_scores = []
        for _, row in group.iterrows():
            doc_id = row['doc_id']
            combined = row['combined_score_mlp']
            max_lex = row['max_lex']
            max_sem = row['max_sem']
            
            # Final Score = CombinedScore + α * MaxLex + (1-α) * MaxSem
            final_score = combined + alpha * max_lex + (1 - alpha) * max_sem
            
            final_scores.append((doc_id, final_score))
        
        final_results[query_id] = final_scores
    
    # Write TREC run file
    output_file = RESULTS_DIR / "mlp_ranking.txt"
    write_trec_run(final_results, output_file, run_name="MLP")
    
    print(f"Results written to: {output_file}")


print("\n" + "=" * 80)
print("Re-ranking complete!")
print(f"Results directory: {RESULTS_DIR}")
print("=" * 80)
