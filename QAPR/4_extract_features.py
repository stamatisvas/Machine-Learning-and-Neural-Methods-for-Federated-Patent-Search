#!/usr/bin/env python3
"""
Step 4: Extract Features
=========================
Extracts lexical (BM25) and semantic (SBERT) features for all 9 section pairs.
Implements Section 3.2 from the paper: Interpolating Lexical and Semantic Similarity.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from utils import load_config, load_pickle
from tqdm import tqdm

config = load_config()

INITIAL_RANKING_FILE = Path(config['output_dir']) / "initial_ranking.tsv"
SPLITS_DIR = Path(config['output_dir']) / "splits"
OUTPUT_DIR = Path(config['output_dir']) / "features"
TRAIN_TEST_SPLIT = config['train_test_split']
SBERT_MODEL = config['sbert_model']
CLEF_FIXED_SPLIT = config.get('clef_fixed_split', config.get('paper_clef_split', False))
CLEF_TOTAL_TOPICS = int(config.get('clef_total_topics', config.get('paper_clef_total_topics', 1351)))
CLEF_TRAIN_TOPICS = int(config.get('clef_train_topics', config.get('paper_clef_train_topics', 1051)))
CLEF_TEST_TOPICS = int(config.get('clef_test_topics', config.get('paper_clef_test_topics', 300)))
SPLIT_SEED = int(config.get('split_seed', 42))

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

# Load SBERT model
print(f"Loading SBERT model: {SBERT_MODEL}...")
print("(This may take a while on first run)")
sbert_model = SentenceTransformer(SBERT_MODEL)

# Section names
sections = ['abstract', 'description', 'claims']


def has_all_required_query_sections(topic_entry: dict) -> bool:
    """Return True when query has non-empty abstract/description/claims."""
    return all(bool(topic_entry.get(section, "").strip()) for section in sections)


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
        
        features_list.append(features)

# Create DataFrame
print("\nCreating features DataFrame...")
features_df = pd.DataFrame(features_list)

print(f"Total features extracted: {len(features_df)}")
print(f"Feature columns: {len(features_df.columns) - 3}")  # Exclude query_id, doc_id, label

# Build synthetic labels:
# For each query, min-max normalize the 18 local scores (9 lexical + 9 semantic)
# column-wise across candidate documents, then sum (and average) the normalized
# scores to obtain a continuous label in [0, 1].
print("Building per-query normalized-sum labels...")
local_score_cols = [col for col in features_df.columns
                    if col.startswith('lex_') or col.startswith('sem_')]

for query_id, group_idx in features_df.groupby('query_id').groups.items():
    group_scores = features_df.loc[group_idx, local_score_cols]
    mins = group_scores.min(axis=0)
    maxs = group_scores.max(axis=0)
    denom = (maxs - mins).replace(0, 1.0)
    normalized = (group_scores - mins) / denom
    # Average of normalized 18 scores keeps the label in [0, 1].
    features_df.loc[group_idx, 'label'] = normalized.sum(axis=1) / len(local_score_cols)

# Split into train/test based on query_id
print("\nSplitting into train/test sets...")

unique_queries = sorted(features_df['query_id'].unique())
valid_queries = [
    query_id for query_id in unique_queries
    if query_id in topic_splits and has_all_required_query_sections(topic_splits[query_id])
]

print(f"Queries with extracted features: {len(unique_queries)}")
print(f"Valid queries (EN abstract+description+claims present): {len(valid_queries)}")

if CLEF_FIXED_SPLIT:
    expected_total = CLEF_TOTAL_TOPICS
    expected_train = CLEF_TRAIN_TOPICS
    expected_test = CLEF_TEST_TOPICS
    expected_sum = expected_train + expected_test

    if expected_total != expected_sum:
        raise ValueError(
            "Invalid CLEF split configuration: "
            f"clef_total_topics={expected_total} must equal "
            f"clef_train_topics + clef_test_topics ({expected_sum})."
        )

    if len(valid_queries) != expected_total:
        raise ValueError(
            f"CLEF fixed split expects exactly {expected_total} valid queries, "
            f"but found {len(valid_queries)}."
        )

    rng = np.random.default_rng(SPLIT_SEED)
    shuffled_queries = valid_queries.copy()
    rng.shuffle(shuffled_queries)

    train_queries = shuffled_queries[:expected_train]
    test_queries = shuffled_queries[expected_train:expected_train + expected_test]

    print(
        "CLEF fixed split enabled: "
        f"{len(train_queries)} train / {len(test_queries)} test "
        f"(seed={SPLIT_SEED})"
    )
else:
    n_train = int(len(valid_queries) * TRAIN_TEST_SPLIT)
    train_queries = valid_queries[:n_train]
    test_queries = valid_queries[n_train:]
    print(
        "Standard split enabled: "
        f"{len(train_queries)} train / {len(test_queries)} test "
        f"(train_test_split={TRAIN_TEST_SPLIT})"
    )

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
