#!/usr/bin/env python3
"""
Step 12: MSMARCO Adaptation (Optional)
=======================================
Adapts QAPR for MSMARCO dataset to test generalizability.
From Paper 1: "A Novel Re-ranking Architecture for Patent Search"

Key differences from patent version:
1. Queries are NOT split (too short)
2. Only documents are split into sections
3. Only 3 scores per component (not 9)
4. No query weights (queries too short for IDF calculation)
5. Uses Dev1 + Dev2 for training (9,254 docs to match patent training size)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from utils import save_pickle, load_pickle
from tqdm import tqdm

print("=" * 80)
print("Step 12: MSMARCO Adaptation")
print("=" * 80)
print()
print("NOTE: This adaptation shows limited performance on MSMARCO,")
print("indicating QAPR's domain-specific nature for patent documents.")
print()

# Configuration
MSMARCO_DOCS = "./data/msmarco/documents"  # User should download MSMARCO
MSMARCO_QUERIES = "./data/msmarco/queries.dev.small.tsv"
MSMARCO_QRELS = "./data/msmarco/qrels.dev.small.tsv"
MSMARCO_CANDIDATES = "./data/msmarco/top1000.dev.tsv"  # BM25 baseline candidates
OUTPUT_DIR = Path("./output/msmarco")
SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading MSMARCO data...")
print(f"Documents: {MSMARCO_DOCS}")
print(f"Queries: {MSMARCO_QUERIES}")
print(f"Candidates: {MSMARCO_CANDIDATES}")
print()

# Check if files exist
if not Path(MSMARCO_QUERIES).exists():
    print("ERROR: MSMARCO files not found!")
    print()
    print("Please download MSMARCO dataset:")
    print("  1. Download from: https://microsoft.github.io/msmarco/")
    print("  2. Place files in ./data/msmarco/")
    print("  3. Required files:")
    print("     - documents/")
    print("     - queries.dev.small.tsv")
    print("     - qrels.dev.small.tsv")
    print("     - top1000.dev.tsv (BM25 baseline)")
    exit(1)

# Load SBERT model
print("Loading SBERT model...")
sbert_model = SentenceTransformer(SBERT_MODEL)

# Load queries
print("Loading queries...")
queries = {}
with open(MSMARCO_QUERIES, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            qid = parts[0]
            query_text = parts[1]
            queries[qid] = query_text

print(f"Loaded {len(queries)} queries")

# Load qrels
print("Loading qrels...")
qrels = {}
with open(MSMARCO_QRELS, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 4:
            qid = parts[0]
            did = parts[2]
            rel = int(parts[3])
            
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][did] = rel

print(f"Loaded qrels for {len(qrels)} queries")

# Function to split document (MSMARCO docs are shorter, split into 3 parts)
def split_document_msmarco(text, max_words=500):
    """
    Split MSMARCO document into 3 roughly equal sections.
    This simulates abstract/description/claims for consistency.
    """
    words = text.split()
    total_words = len(words)
    
    if total_words <= max_words:
        # Short doc: use same text for all 3 sections
        return {
            'section1': text,
            'section2': text,
            'section3': text
        }
    
    # Split into 3 parts
    third = total_words // 3
    
    return {
        'section1': ' '.join(words[:third]),
        'section2': ' '.join(words[third:2*third]),
        'section3': ' '.join(words[2*third:])
    }


def extract_features_msmarco(query_text, doc_sections, initial_score):
    """
    Extract features for MSMARCO (adapted from patent version).
    
    Differences:
    - Query is NOT split (too short)
    - Only 3 lexical scores (query vs 3 doc sections)
    - Only 3 semantic scores (query vs 3 doc sections)
    - Total: 1 initial + 3 lex + 3 sem = 7 features (vs 19 for patents)
    """
    features = {'initial_score': initial_score}
    corpus = []
    
    # Lexical scores (3)
    lex_scores = []
    for i, section_text in enumerate(doc_sections.values(), 1):
        if not query_text or not section_text:
            score = 0.0
        else:
            query_tokens = query_text.lower().split()
            doc_tokens = section_text.lower().split()
            
            if doc_tokens not in corpus:
                corpus.append(doc_tokens)
            
            bm25 = BM25Okapi(corpus)
            score = float(bm25.get_scores(query_tokens)[-1])
        
        features[f'lex_{i}'] = score
        lex_scores.append(score)
    
    # Semantic scores (3)
    sem_scores = []
    for i, section_text in enumerate(doc_sections.values(), 1):
        if not query_text or not section_text:
            score = 0.0
        else:
            query_emb = sbert_model.encode([query_text])
            doc_emb = sbert_model.encode([section_text])
            score = float(cosine_similarity(query_emb, doc_emb)[0][0])
        
        features[f'sem_{i}'] = score
        sem_scores.append(score)
    
    features['max_lex'] = max(lex_scores) if lex_scores else 0
    features['max_sem'] = max(sem_scores) if sem_scores else 0
    
    return features


print("\nProcessing MSMARCO candidates and extracting features...")
print("This may take a while...")

# Load candidate rankings
candidates_df = pd.read_csv(MSMARCO_CANDIDATES, sep='\t', 
                            names=['qid', 'did', 'rank', 'score'])

print(f"Loaded {len(candidates_df)} candidate pairs")

# Extract features
features_list = []

for qid, group in tqdm(candidates_df.groupby('qid'), desc="Extracting features"):
    if qid not in queries:
        continue
    
    query_text = queries[qid]
    
    for _, row in group.iterrows():
        did = row['did']
        initial_score = row['score']
        
        # Load document (simplified - assumes docs are in memory or accessible)
        # In production, you'd load from disk
        doc_path = Path(MSMARCO_DOCS) / f"{did}.txt"
        
        if not doc_path.exists():
            continue
        
        with open(doc_path, 'r', encoding='utf-8') as f:
            doc_text = f.read()
        
        # Split document
        doc_sections = split_document_msmarco(doc_text)
        
        # Extract features
        features = extract_features_msmarco(query_text, doc_sections, initial_score)
        
        # Add label
        label = 0
        if qid in qrels and did in qrels[qid]:
            label = qrels[qid][did]
        
        features['qid'] = qid
        features['did'] = did
        features['label'] = label
        
        features_list.append(features)

# Create DataFrame
features_df = pd.DataFrame(features_list)

print(f"\nExtracted features for {len(features_df)} pairs")
print(f"Queries: {features_df['qid'].nunique()}")

# Split into train/test (use Dev1 + Dev2 for training as per paper)
# For this example, use first 80% for training
unique_queries = features_df['qid'].unique()
n_train = int(len(unique_queries) * 0.8)

train_queries = unique_queries[:n_train]
test_queries = unique_queries[n_train:]

train_df = features_df[features_df['qid'].isin(train_queries)]
test_df = features_df[features_df['qid'].isin(test_queries)]

print(f"\nTrain set: {len(train_df)} pairs ({len(train_queries)} queries)")
print(f"Test set: {len(test_df)} pairs ({len(test_queries)} queries)")

# Train models (MLP only, as paper shows it works better on MSMARCO)
print("\n" + "-" * 80)
print("Training MLP model...")
print("-" * 80)

feature_cols = [f'lex_{i}' for i in range(1, 4)] + \
               [f'sem_{i}' for i in range(1, 4)] + \
               ['initial_score']

X_train = train_df[feature_cols].values
y_train = train_df['label'].values

# Normalize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train MLP
mlp = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    learning_rate_init=0.001,
    max_iter=50,
    batch_size=32,
    verbose=True,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=42
)

mlp.fit(X_train_scaled, y_train)

# Save models
save_pickle(mlp, OUTPUT_DIR / "mlp_msmarco.pkl")
save_pickle(scaler, OUTPUT_DIR / "scaler_msmarco.pkl")

print(f"\nModels saved to: {OUTPUT_DIR}")

# Re-rank test set
print("\n" + "-" * 80)
print("Re-ranking test set...")
print("-" * 80)

X_test = test_df[feature_cols].values
X_test_scaled = scaler.transform(X_test)

# Predict
combined_scores = mlp.predict(X_test_scaled)
test_df['combined_score'] = combined_scores

# Note: No query weights for MSMARCO (queries too short)
# Final score = combined_score + max_lex + max_sem (equal weights)
test_df['final_score'] = (test_df['combined_score'] + 
                           test_df['max_lex'] + 
                           test_df['max_sem'])

# Write TREC format
output_file = OUTPUT_DIR / "msmarco_qapr_ranking.txt"

with open(output_file, 'w') as f:
    for qid, group in test_df.groupby('qid'):
        sorted_group = group.sort_values('final_score', ascending=False)
        
        for rank, (_, row) in enumerate(sorted_group.iterrows(), 1):
            did = row['did']
            score = row['final_score']
            f.write(f"{qid}\tQ0\t{did}\t{rank}\t{score:.6f}\tQAPR_MSMARCO\n")

print(f"\nResults saved to: {output_file}")

print("\n" + "=" * 80)
print("MSMARCO adaptation complete!")
print()
print("NOTE: As reported in the paper, QAPR shows limited performance")
print("on MSMARCO compared to patent datasets, indicating its")
print("domain-specific nature.")
print()
print("Key differences that may explain lower performance:")
print("  - Queries not split (too short)")
print("  - No query-specific weights")
print("  - Only 7 features vs 19 for patents")
print("  - Different document structure")
print("=" * 80)
