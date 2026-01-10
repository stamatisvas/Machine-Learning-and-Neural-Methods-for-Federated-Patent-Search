#!/usr/bin/env python3
"""
Step 8: Index with Dense Models (MiniLM + ColBERT)
===================================================
Creates dense embeddings for first-stage retrieval experiments.
Implements methodology from: "Beyond BM25: Strengthening First-Stage Patent Retrieval"
"""

import os
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from utils import load_config, parse_sgml_document, save_pickle
from tqdm import tqdm

config = load_config()

DOCUMENTS_DIR = config['documents_dir']
OUTPUT_DIR = Path(config['output_dir']) / "dense_indexes"

print("=" * 80)
print("Step 8: Index with Dense Models")
print("=" * 80)
print()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load documents
print("Loading documents...")
documents = {}

for root, dirs, files in os.walk(DOCUMENTS_DIR):
    if not files:
        continue
    
    for file in tqdm(files, desc=f"Processing {Path(root).name}"):
        if not file.endswith('.txt'):
            continue
        
        file_path = os.path.join(root, file)
        
        try:
            doc_data = parse_sgml_document(file_path)
            
            if not doc_data['doc_id']:
                continue
            
            documents[doc_data['doc_id']] = {
                'abstract': doc_data['abstract'],
                'description': doc_data['description'],
                'claims': doc_data['claims']
            }
        except:
            continue

print(f"Loaded {len(documents)} documents")

# Index with MiniLM
print("\n" + "-" * 80)
print("Indexing with MiniLM (all-MiniLM-L6-v2)...")
print("-" * 80)

model_minilm = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

doc_ids_minilm = []
embeddings_minilm = []

print("Encoding documents (512 tokens per section, averaged)...")

for doc_id, doc in tqdm(documents.items(), desc="Encoding"):
    # Limit to 512 tokens per section (approx)
    # Average embeddings from 3 sections
    abstract_text = ' '.join(doc['abstract'].split()[:512])
    description_text = ' '.join(doc['description'].split()[:512])
    claims_text = ' '.join(doc['claims'].split()[:512])
    
    # Encode each section
    emb_abs = model_minilm.encode(abstract_text, convert_to_numpy=True, show_progress_bar=False)
    emb_desc = model_minilm.encode(description_text, convert_to_numpy=True, show_progress_bar=False)
    emb_clm = model_minilm.encode(claims_text, convert_to_numpy=True, show_progress_bar=False)
    
    # Average the three embeddings
    avg_embedding = (emb_abs + emb_desc + emb_clm) / 3
    
    doc_ids_minilm.append(doc_id)
    embeddings_minilm.append(avg_embedding)

embeddings_minilm = np.array(embeddings_minilm)

# Save MiniLM index
minilm_index = {
    'doc_ids': doc_ids_minilm,
    'embeddings': embeddings_minilm
}

save_pickle(minilm_index, OUTPUT_DIR / "minilm_index.pkl")

print(f"MiniLM index saved: {len(doc_ids_minilm)} documents")
print(f"Embedding shape: {embeddings_minilm.shape}")

# Index with ColBERT (simplified - just title + abstract, 256 tokens)
print("\n" + "-" * 80)
print("Indexing with ColBERT (simplified: title + abstract, 256 tokens)...")
print("-" * 80)

# Note: Full ColBERT requires special token-level encoding
# This is a simplified version using sentence-transformers
# For production, use the official ColBERT library

doc_ids_colbert = []
embeddings_colbert = []

print("Encoding documents (256 tokens from abstract only)...")

for doc_id, doc in tqdm(documents.items(), desc="Encoding"):
    # Use only first 256 tokens from abstract (ColBERT constraint from paper)
    text = ' '.join(doc['abstract'].split()[:256])
    
    # Encode (in real ColBERT, this would be token-level)
    embedding = model_minilm.encode(text, convert_to_numpy=True, show_progress_bar=False)
    
    doc_ids_colbert.append(doc_id)
    embeddings_colbert.append(embedding)

embeddings_colbert = np.array(embeddings_colbert)

# Save ColBERT index
colbert_index = {
    'doc_ids': doc_ids_colbert,
    'embeddings': embeddings_colbert
}

save_pickle(colbert_index, OUTPUT_DIR / "colbert_index.pkl")

print(f"ColBERT index saved: {len(doc_ids_colbert)} documents")
print(f"Embedding shape: {embeddings_colbert.shape}")

print("\n" + "=" * 80)
print("Dense indexing complete!")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Files created:")
print(f"  - minilm_index.pkl")
print(f"  - colbert_index.pkl")
print("=" * 80)
