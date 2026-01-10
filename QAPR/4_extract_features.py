#!/usr/bin/env python3
"""
Step 4: Extract Features
=========================
Extracts lexical (BM25) and semantic (SBERT) features for all 9 section pairs.
Implements Section 3.2 from the paper: Interpolating Lexical and Semantic Similarity.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from utils import load_config, load_pickle, load_qrels
from tqdm import tqdm

config = load_config()

INITIAL_RANKING_FILE = Path(config['output_dir']) / "initial_ranking.tsv"
SPLITS_DIR = Path(config['output_dir']) / "splits"
OUTPUT_DIR = Path(config['output_dir']) / "features"
QRELS_FILE = config['qrels_file']
TRAIN_TEST_SPLIT = config['train_test_split']
SBERT_MODEL = config['sbert_model']

print("=" * 80)
print("Step 4: Extract Features")
print("=" * 80)
print(f"SBERT model: {SBERT_MODEL}")
print()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load splits and IDF
print("Loading document and topic splits...")
doc_splits = load_pickle(SPLITS_DIR / "document_splits.pkl")
topic_splits = load_pickle(SPLITS_DIR / "topic_splits.pkl")
idf_dict = load_pickle(SPLITS_DIR / "idf_dict.pkl")

# Load initial rankings
print("Loading initial rankings...")
initial_ranking = pd.read_csv(INITIAL_RANKING_FILE, sep='\t')

# Load qrels for labels
print("Loading qrels...")
qrels = load_qrels(QRELS_FILE)

# Load SBERT model
print(f"Loading SBERT model: {SBERT_MODEL}...")
print("(This may take a while on first run)")
sbert_model = SentenceTransformer(SBERT_MODEL)

# Section names
sections = ['abstract', 'description', 'claims']


def calculate_bm25_score(query_text: str, doc_text: str, corpus: list) -> float:
    """Calculate BM25 score between query and document."""
    if not query_text or not doc_text:
        return 0.0
    
    # Tokenize
    query_tokens = query_text.lower().split()
    doc_tokens = doc_text.lower().split()
    
    # Add document to corpus if needed
    if doc_tokens not in corpus:
        corpus.append(doc_tokens)
    
    # Calculate BM25
    bm25 = BM25Okapi(corpus)
    score = bm25.get_scores(query_tokens)[len(corpus) - 1]
    
    return float(score)


def calculate_sbert_score(query_text: str, doc_text: str) -> float:
    """Calculate SBERT cosine similarity between query and document."""
    if not query_text or not doc_text:
        return 0.0
    
    # Encode
    query_emb = sbert_model.encode([query_text])
    doc_emb = sbert_model.encode([doc_text])
    
    # Cosine similarity
    score = cosine_similarity(query_emb, doc_emb)[0][0]
    
    return float(score)


print("\nExtracting features for all query-document pairs...")
print("This will take a while...")

features_list = []

# Group by query_id for efficiency
for query_id, group in tqdm(initial_ranking.groupby('query_id'), desc="Processing queries"):
    
    if query_id not in topic_splits:
        continue
    
    topic = topic_splits[query_id]
    
    # Prepare corpus for BM25 (all candidate documents)
    corpus = []
    
    for _, row in group.iterrows():
        doc_id = row['doc_id']
        initial_score = row['score']
        
        if doc_id not in doc_splits:
            continue
        
        doc = doc_splits[doc_id]
        
        # Extract 19 features: 1 initial BM25 + 9 lexical + 9 semantic
        features = {
            'query_id': query_id,
            'doc_id': doc_id,
            'initial_bm25': initial_score
        }
        
        # Calculate 9 lexical scores (BM25 for each section pair)
        lex_scores = []
        for q_section in sections:
            for d_section in sections:
                pair_name = f"{q_section[0]}{d_section[0]}"  # e.g., "aa", "ad", "ac"
                
                score = calculate_bm25_score(
                    topic[q_section],
                    doc[d_section],
                    corpus
                )
                
                features[f'lex_{pair_name}'] = score
                lex_scores.append(score)
        
        # Calculate 9 semantic scores (SBERT for each section pair)
        sem_scores = []
        for q_section in sections:
            for d_section in sections:
                pair_name = f"{q_section[0]}{d_section[0]}"
                
                score = calculate_sbert_score(
                    topic[q_section],
                    doc[d_section]
                )
                
                features[f'sem_{pair_name}'] = score
                sem_scores.append(score)
        
        # Add max scores for later use
        features['max_lex'] = max(lex_scores) if lex_scores else 0
        features['max_sem'] = max(sem_scores) if sem_scores else 0
        
        # Add label (relevance from qrels)
        label = 0
        if query_id in qrels and doc_id in qrels[query_id]:
            label = qrels[query_id][doc_id]
        features['label'] = label
        
        features_list.append(features)

# Create DataFrame
print("\nCreating features DataFrame...")
features_df = pd.DataFrame(features_list)

print(f"Total features extracted: {len(features_df)}")
print(f"Feature columns: {len(features_df.columns) - 3}")  # Exclude query_id, doc_id, label

# Split into train/test based on query_id
print("\nSplitting into train/test sets...")

unique_queries = features_df['query_id'].unique()
n_train = int(len(unique_queries) * TRAIN_TEST_SPLIT)

train_queries = unique_queries[:n_train]
test_queries = unique_queries[n_train:]

train_df = features_df[features_df['query_id'].isin(train_queries)]
test_df = features_df[features_df['query_id'].isin(test_queries)]

# Save
train_file = OUTPUT_DIR / "features_train.csv"
test_file = OUTPUT_DIR / "features_test.csv"

train_df.to_csv(train_file, index=False)
test_df.to_csv(test_file, index=False)

print(f"\nTrain set: {len(train_df)} instances ({len(train_queries)} queries)")
print(f"Test set: {len(test_df)} instances ({len(test_queries)} queries)")

print("\n" + "=" * 80)
print("Feature extraction complete!")
print(f"Output files:")
print(f"  - {train_file}")
print(f"  - {test_file}")
print("=" * 80)
