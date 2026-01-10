#!/usr/bin/env python3
"""
Step 1: Index Documents with BM25
==================================
Creates BM25 index using pyserini for first-stage retrieval.
"""

import os
import json
from pathlib import Path
from utils import load_config, parse_sgml_document
from tqdm import tqdm

config = load_config()

DOCUMENTS_DIR = config['documents_dir']
INDEX_DIR = config['index_dir']
BM25_K1 = config['bm25_k1']
BM25_B = config['bm25_b']

print("="* 80)
print("Step 1: Indexing Documents with BM25")
print("=" * 80)
print(f"Documents directory: {DOCUMENTS_DIR}")
print(f"Index directory: {INDEX_DIR}")
print(f"BM25 parameters: k1={BM25_K1}, b={BM25_B}")
print()

# Create JSON collection for pyserini
json_dir = Path(INDEX_DIR) / "json_collection"
json_dir.mkdir(parents=True, exist_ok=True)

print("Converting SGML documents to JSON format...")

json_docs = []
doc_count = 0

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
            
            # Combine all sections for BM25 index
            full_text = f"{doc_data['abstract']} {doc_data['description']} {doc_data['claims']}"
            
            json_doc = {
                "id": doc_data['doc_id'],
                "contents": full_text.strip()
            }
            
            json_docs.append(json_doc)
            doc_count += 1
            
            # Write in batches to avoid memory issues
            if len(json_docs) >= 10000:
                batch_file = json_dir / f"docs_{doc_count // 10000}.json"
                with open(batch_file, 'w', encoding='utf-8') as f:
                    for doc in json_docs:
                        f.write(json.dumps(doc) + '\n')
                json_docs = []
        
        except Exception as e:
            print(f"Error processing {file}: {e}")

# Write remaining documents
if json_docs:
    batch_file = json_dir / f"docs_{(doc_count // 10000) + 1}.json"
    with open(batch_file, 'w', encoding='utf-8') as f:
        for doc in json_docs:
            f.write(json.dumps(doc) + '\n')

print(f"\nTotal documents indexed: {doc_count}")

# Build index using pyserini
print("\nBuilding BM25 index with pyserini...")
print("This may take a while...")

index_output = Path(INDEX_DIR) / "lucene_index"

os.system(f"python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input {json_dir} \
  --index {index_output} \
  --generator DefaultLuceneDocumentGenerator \
  --threads 4 \
  --storePositions --storeDocvectors --storeRaw")

print("\n" + "=" * 80)
print("Indexing complete!")
print(f"Index location: {index_output}")
print("=" * 80)
