#!/usr/bin/env python3
"""
Step 9: Retrieve with All Methods (BM25, MiniLM, ColBERT)
==========================================================
Performs first-stage retrieval with all baseline methods.
From paper: "Beyond BM25: Strengthening First-Stage Patent Retrieval"
"""

import os
import numpy as np
from pathlib import Path
from pyserini.search.lucene import LuceneSearcher
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from utils import load_config, parse_sgml_document, load_pickle
from tqdm import tqdm

config = load_config()

BM25_INDEX_DIR = Path(config['index_dir']) / "lucene_index"
DENSE_INDEX_DIR = Path(config['output_dir']) / "dense_indexes"
TOPICS_DIR = config['topics_dir']
OUTPUT_DIR = Path(config['output_dir']) / "first_stage_rankings"
TOP_K = 1000  # Paper uses 1000 candidates
BM25_K1 = config['bm25_k1']
BM25_B = config['bm25_b']

print("=" * 80)
print("Step 9: First-Stage Retrieval with All Methods")
print("=" * 80)
print(f"Top-k: {TOP_K}")
print()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load topics
print("Loading topics...")
topics = []

for file in os.listdir(TOPICS_DIR):
    if not file.endswith('.xml'):
        continue
    
    file_path = os.path.join(TOPICS_DIR, file)
    try:
        topic_data = parse_sgml_document(file_path)
        
        topics.append({
            'topic_id': topic_data['doc_id'],
            'abstract': topic_data['abstract'],
            'description': topic_data['description'],
            'claims': topic_data['claims']
        })
    except Exception as e:
        print(f"Error loading topic {file}: {e}")

print(f"Loaded {len(topics)} topics\n")

# === Method 1: BM25 ===
print("-" * 80)
print("Method 1: BM25 Retrieval")
print("-" * 80)

searcher = LuceneSearcher(str(BM25_INDEX_DIR))
searcher.set_bm25(BM25_K1, BM25_B)

bm25_file = OUTPUT_DIR / "bm25_ranking.tsv"

with open(bm25_file, 'w', encoding='utf-8') as out:
    out.write("query_id\tdoc_id\trank\tscore\n")
    
    for topic in tqdm(topics, desc="BM25"):
        topic_id = topic['topic_id']
        query = f"{topic['abstract']} {topic['description']} {topic['claims']}"
        
        hits = searcher.search(query, k=TOP_K)
        
        for rank, hit in enumerate(hits, 1):
            out.write(f"{topic_id}\t{hit.docid}\t{rank}\t{hit.score}\n")

print(f"BM25 results saved to: {bm25_file}")

# === Method 2: MiniLM (Dense Retrieval) ===
print("\n" + "-" * 80)
print("Method 2: MiniLM Dense Retrieval")
print("-" * 80)

model_minilm = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Load index
minilm_index = load_pickle(DENSE_INDEX_DIR / "minilm_index.pkl")
doc_ids = minilm_index['doc_ids']
doc_embeddings = minilm_index['embeddings']

print(f"Index size: {len(doc_ids)} documents")

minilm_file = OUTPUT_DIR / "minilm_ranking.tsv"

with open(minilm_file, 'w', encoding='utf-8') as out:
    out.write("query_id\tdoc_id\trank\tscore\n")
    
    for topic in tqdm(topics, desc="MiniLM"):
        topic_id = topic['topic_id']
        
        # Encode query (same way as documents: 3 sections averaged)
        abstract_text = ' '.join(topic['abstract'].split()[:512])
        description_text = ' '.join(topic['description'].split()[:512])
        claims_text = ' '.join(topic['claims'].split()[:512])
        
        emb_abs = model_minilm.encode(abstract_text, convert_to_numpy=True, show_progress_bar=False)
        emb_desc = model_minilm.encode(description_text, convert_to_numpy=True, show_progress_bar=False)
        emb_clm = model_minilm.encode(claims_text, convert_to_numpy=True, show_progress_bar=False)
        
        query_embedding = (emb_abs + emb_desc + emb_clm) / 3
        query_embedding = query_embedding.reshape(1, -1)
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
        
        # Get top-k
        top_k_indices = np.argsort(similarities)[::-1][:TOP_K]
        
        for rank, idx in enumerate(top_k_indices, 1):
            doc_id = doc_ids[idx]
            score = similarities[idx]
            out.write(f"{topic_id}\t{doc_id}\t{rank}\t{score}\n")

print(f"MiniLM results saved to: {minilm_file}")

# === Method 3: ColBERT (Simplified) ===
print("\n" + "-" * 80)
print("Method 3: ColBERT Retrieval (simplified)")
print("-" * 80)

# Load index
colbert_index = load_pickle(DENSE_INDEX_DIR / "colbert_index.pkl")
doc_ids_colbert = colbert_index['doc_ids']
doc_embeddings_colbert = colbert_index['embeddings']

print(f"Index size: {len(doc_ids_colbert)} documents")

colbert_file = OUTPUT_DIR / "colbert_ranking.tsv"

with open(colbert_file, 'w', encoding='utf-8') as out:
    out.write("query_id\tdoc_id\trank\tscore\n")
    
    for topic in tqdm(topics, desc="ColBERT"):
        topic_id = topic['topic_id']
        
        # Encode query (256 tokens from abstract, as per paper)
        text = ' '.join(topic['abstract'].split()[:256])
        query_embedding = model_minilm.encode(text, convert_to_numpy=True, show_progress_bar=False)
        query_embedding = query_embedding.reshape(1, -1)
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, doc_embeddings_colbert)[0]
        
        # Get top-k
        top_k_indices = np.argsort(similarities)[::-1][:TOP_K]
        
        for rank, idx in enumerate(top_k_indices, 1):
            doc_id = doc_ids_colbert[idx]
            score = similarities[idx]
            out.write(f"{topic_id}\t{doc_id}\t{rank}\t{score}\n")

print(f"ColBERT results saved to: {colbert_file}")

print("\n" + "=" * 80)
print("First-stage retrieval complete!")
print(f"Output directory: {OUTPUT_DIR}")
print("Files created:")
print("  - bm25_ranking.tsv")
print("  - minilm_ranking.tsv")
print("  - colbert_ranking.tsv")
print("=" * 80)
