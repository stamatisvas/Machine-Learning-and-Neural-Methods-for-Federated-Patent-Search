#!/usr/bin/env python3
"""
Step 2: First-Stage Retrieval with BM25
========================================
Retrieves top-k candidates using BM25 for each query.
"""

import os
from pathlib import Path
from pyserini.search.lucene import LuceneSearcher
from utils import load_config, parse_sgml_document
from tqdm import tqdm

config = load_config()

INDEX_DIR = Path(config['index_dir']) / "lucene_index"
TOPICS_DIR = config['topics_dir']
OUTPUT_FILE = Path(config['output_dir']) / "initial_ranking.tsv"
TOP_K = config['top_k']
BM25_K1 = config['bm25_k1']
BM25_B = config['bm25_b']

print("=" * 80)
print("Step 2: First-Stage Retrieval with BM25")
print("=" * 80)
print(f"Index: {INDEX_DIR}")
print(f"Topics: {TOPICS_DIR}")
print(f"Top-k: {TOP_K}")
print(f"BM25 parameters: k1={BM25_K1}, b={BM25_B}")
print()

# Initialize searcher
print("Loading BM25 index...")
searcher = LuceneSearcher(str(INDEX_DIR))
searcher.set_bm25(BM25_K1, BM25_B)

# Load topics
print("Loading topics...")
topics = []

for file in os.listdir(TOPICS_DIR):
    if not file.endswith('.xml'):
        continue
    
    file_path = os.path.join(TOPICS_DIR, file)
    try:
        topic_data = parse_sgml_document(file_path)
        
        # Combine sections as query
        query_text = f"{topic_data['abstract']} {topic_data['description']} {topic_data['claims']}"
        
        topics.append({
            'topic_id': topic_data['doc_id'],
            'query': query_text.strip()
        })
    except Exception as e:
        print(f"Error loading topic {file}: {e}")

print(f"Loaded {len(topics)} topics")

# Perform retrieval
print(f"\nRetrieving top-{TOP_K} candidates for each topic...")

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with open(OUTPUT_FILE, 'w', encoding='utf-8') as out:
    out.write("query_id\tdoc_id\trank\tscore\n")
    
    for topic in tqdm(topics):
        topic_id = topic['topic_id']
        query = topic['query']
        
        # Search
        hits = searcher.search(query, k=TOP_K)
        
        for rank, hit in enumerate(hits, 1):
            out.write(f"{topic_id}\t{hit.docid}\t{rank}\t{hit.score}\n")

print("\n" + "=" * 80)
print("First-stage retrieval complete!")
print(f"Output: {OUTPUT_FILE}")
print("=" * 80)
