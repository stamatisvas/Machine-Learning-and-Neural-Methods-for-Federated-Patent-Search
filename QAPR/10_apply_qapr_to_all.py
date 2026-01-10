#!/usr/bin/env python3
"""
Step 10: Apply QAPR to All First-Stage Results
===============================================
Applies QAPR re-ranking to BM25, MiniLM, and ColBERT results.
From paper: "Beyond BM25: Strengthening First-Stage Patent Retrieval"
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from utils import (load_config, load_pickle, calculate_query_alpha,
                   write_trec_run, save_pickle)
from tqdm import tqdm

config = load_config()

FIRST_STAGE_DIR = Path(config['output_dir']) / "first_stage_rankings"
SPLITS_DIR = Path(config['output_dir']) / "splits"
MODELS_DIR = Path(config['output_dir']) / "models"
OUTPUT_DIR = Path(config['output_dir']) / "qapr_results"
SBERT_MODEL = config['sbert_model']

print("=" * 80)
print("Step 10: Apply QAPR to All First-Stage Results")
print("=" * 80)
print()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load necessary data
print("Loading splits and models...")
doc_splits = load_pickle(SPLITS_DIR / "document_splits.pkl")
topic_splits = load_pickle(SPLITS_DIR / "topic_splits.pkl")
idf_dict = load_pickle(SPLITS_DIR / "idf_dict.pkl")
mlp_model = load_pickle(MODELS_DIR / "mlp.pkl")
scaler = load_pickle(MODELS_DIR / "scaler.pkl")

# Load SBERT model
print(f"Loading SBERT model: {SBERT_MODEL}...")
sbert_model = SentenceTransformer(SBERT_MODEL)

# Calculate doc IDF statistics for alpha
print("Calculating document IDF statistics...")
all_doc_idfs = []
for doc_id, doc in doc_splits.items():
    full_text = f"{doc['abstract']} {doc['description']} {doc['claims']}"
    words = full_text.lower().split()
    idfs = [idf_dict.get(word, 0) for word in words]
    avg_idf = np.mean(idfs) if idfs else 0
    all_doc_idfs.append(avg_idf)

sections = ['abstract', 'description', 'claims']


def extract_features_for_pair(topic, doc):
    """Extract 19 features for a query-document pair."""
    features = {}
    corpus = []
    
    # 9 lexical scores
    lex_scores = []
    for q_section in sections:
        for d_section in sections:
            # BM25 score
            query_text = topic[q_section]
            doc_text = doc[d_section]
            
            if not query_text or not doc_text:
                score = 0.0
            else:
                query_tokens = query_text.lower().split()
                doc_tokens = doc_text.lower().split()
                
                if doc_tokens not in corpus:
                    corpus.append(doc_tokens)
                
                bm25 = BM25Okapi(corpus)
                score = float(bm25.get_scores(query_tokens)[-1])
            
            pair_name = f"{q_section[0]}{d_section[0]}"
            features[f'lex_{pair_name}'] = score
            lex_scores.append(score)
    
    # 9 semantic scores
    sem_scores = []
    for q_section in sections:
        for d_section in sections:
            query_text = topic[q_section]
            doc_text = doc[d_section]
            
            if not query_text or not doc_text:
                score = 0.0
            else:
                query_emb = sbert_model.encode([query_text])
                doc_emb = sbert_model.encode([doc_text])
                score = float(cosine_similarity(query_emb, doc_emb)[0][0])
            
            pair_name = f"{q_section[0]}{d_section[0]}"
            features[f'sem_{pair_name}'] = score
            sem_scores.append(score)
    
    features['max_lex'] = max(lex_scores) if lex_scores else 0
    features['max_sem'] = max(sem_scores) if sem_scores else 0
    
    return features


def apply_qapr(ranking_file, method_name):
    """Apply QAPR to a first-stage ranking."""
    print(f"\n{'-' * 80}")
    print(f"Applying QAPR to {method_name}")
    print(f"{'-' * 80}")
    
    # Load ranking
    ranking_df = pd.read_csv(ranking_file, sep='\t')
    
    print(f"Loaded {len(ranking_df)} query-document pairs")
    print(f"Queries: {ranking_df['query_id'].nunique()}")
    
    # Process each query
    final_results = {}
    
    for query_id, group in tqdm(ranking_df.groupby('query_id'), desc=f"Processing {method_name}"):
        if query_id not in topic_splits:
            continue
        
        topic = topic_splits[query_id]
        query_text = f"{topic['abstract']} {topic['description']} {topic['claims']}"
        
        # Calculate alpha for this query
        alpha = calculate_query_alpha(query_text, idf_dict, all_doc_idfs)
        
        final_scores = []
        
        for _, row in group.iterrows():
            doc_id = row['doc_id']
            initial_score = row['score']
            
            if doc_id not in doc_splits:
                # Keep original score if document not in splits
                final_scores.append((doc_id, initial_score))
                continue
            
            doc = doc_splits[doc_id]
            
            # Extract features
            features = extract_features_for_pair(topic, doc)
            
            # Prepare feature vector (19 features + initial score)
            feature_vector = [initial_score]
            for q_sec in sections:
                for d_sec in sections:
                    pair = f"{q_sec[0]}{d_sec[0]}"
                    feature_vector.append(features[f'lex_{pair}'])
            for q_sec in sections:
                for d_sec in sections:
                    pair = f"{q_sec[0]}{d_sec[0]}"
                    feature_vector.append(features[f'sem_{pair}'])
            
            feature_vector = np.array(feature_vector).reshape(1, -1)
            
            # Scale and predict
            feature_vector_scaled = scaler.transform(feature_vector)
            combined_score = mlp_model.predict(feature_vector_scaled)[0]
            
            # Final score with query-specific weights
            max_lex = features['max_lex']
            max_sem = features['max_sem']
            
            final_score = combined_score + alpha * max_lex + (1 - alpha) * max_sem
            
            final_scores.append((doc_id, final_score))
        
        final_results[query_id] = final_scores
    
    # Write results
    output_file = OUTPUT_DIR / f"qapr_{method_name}_ranking.txt"
    write_trec_run(final_results, output_file, run_name=f"QAPR_{method_name}")
    
    print(f"Results saved to: {output_file}")


# Apply QAPR to all three methods
apply_qapr(FIRST_STAGE_DIR / "bm25_ranking.tsv", "BM25")
apply_qapr(FIRST_STAGE_DIR / "minilm_ranking.tsv", "MiniLM")
apply_qapr(FIRST_STAGE_DIR / "colbert_ranking.tsv", "ColBERT")

print("\n" + "=" * 80)
print("QAPR application complete!")
print(f"Output directory: {OUTPUT_DIR}")
print("Files created:")
print("  - qapr_BM25_ranking.txt")
print("  - qapr_MiniLM_ranking.txt")
print("  - qapr_ColBERT_ranking.txt")
print("=" * 80)
