#!/usr/bin/env python3
"""
Step 3: Split Documents into Sections
======================================
Splits patents and topics into Abstract, Description, Claims (max 500 words each).
Implements the document splitting strategy from Section 3.1 of the paper.
"""

import os
import pandas as pd
from pathlib import Path
from utils import (load_config, parse_sgml_document, split_into_words,
                   select_best_passage, calculate_idf, save_pickle)
from tqdm import tqdm

config = load_config()

DOCUMENTS_DIR = config['documents_dir']
TOPICS_DIR = config['topics_dir']
OUTPUT_DIR = Path(config['output_dir']) / "splits"
MAX_WORDS = config['max_section_words']

print("=" * 80)
print("Step 3: Split Documents into Sections")
print("=" * 80)
print(f"Documents: {DOCUMENTS_DIR}")
print(f"Topics: {TOPICS_DIR}")
print(f"Max words per section: {MAX_WORDS}")
print()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Step 1: Calculate IDF from all documents
print("Step 1/3: Calculating IDF from collection...")

all_texts = []
doc_count = 0

for root, dirs, files in os.walk(DOCUMENTS_DIR):
    if not files:
        continue
    
    for file in tqdm(files, desc="Loading documents"):
        if not file.endswith('.txt'):
            continue
        
        file_path = os.path.join(root, file)
        
        try:
            doc_data = parse_sgml_document(file_path)
            full_text = f"{doc_data['abstract']} {doc_data['description']} {doc_data['claims']}"
            all_texts.append(full_text)
            doc_count += 1
        except:
            continue

print(f"Calculating IDF from {doc_count} documents...")
idf_dict = calculate_idf(all_texts)

print(f"IDF dictionary size: {len(idf_dict)} terms")

# Save IDF
save_pickle(idf_dict, OUTPUT_DIR / "idf_dict.pkl")

# Step 2: Process documents
print("\nStep 2/3: Processing document sections...")

doc_splits = {}

for root, dirs, files in os.walk(DOCUMENTS_DIR):
    if not files:
        continue
    
    for file in tqdm(files, desc=f"Processing {Path(root).name}"):
        if not file.endswith('.txt'):
            continue
        
        file_path = os.path.join(root, file)
        
        try:
            doc_data = parse_sgml_document(file_path)
            doc_id = doc_data['doc_id']
            
            if not doc_id:
                continue
            
            # Split each section (max 500 words)
            # If > 500 words, select passage with highest avg IDF
            abstract = doc_data['abstract']
            if len(abstract.split()) > MAX_WORDS:
                abstract = select_best_passage(abstract, MAX_WORDS, idf_dict)
            else:
                abstract = split_into_words(abstract, MAX_WORDS)
            
            description = doc_data['description']
            if len(description.split()) > MAX_WORDS:
                description = select_best_passage(description, MAX_WORDS, idf_dict)
            else:
                description = split_into_words(description, MAX_WORDS)
            
            claims = doc_data['claims']
            if len(claims.split()) > MAX_WORDS:
                claims = select_best_passage(claims, MAX_WORDS, idf_dict)
            else:
                claims = split_into_words(claims, MAX_WORDS)
            
            doc_splits[doc_id] = {
                'abstract': abstract,
                'description': description,
                'claims': claims
            }
        
        except Exception as e:
            continue

print(f"Processed {len(doc_splits)} documents")

# Save document splits
save_pickle(doc_splits, OUTPUT_DIR / "document_splits.pkl")

# Step 3: Process topics (queries)
print("\nStep 3/3: Processing topic sections...")

topic_splits = {}

for file in os.listdir(TOPICS_DIR):
    if not file.endswith('.xml'):
        continue
    
    file_path = os.path.join(TOPICS_DIR, file)
    
    try:
        topic_data = parse_sgml_document(file_path)
        topic_id = topic_data['doc_id']
        
        if not topic_id:
            continue
        
        # Split each section (max 500 words)
        abstract = split_into_words(topic_data['abstract'], MAX_WORDS)
        description = split_into_words(topic_data['description'], MAX_WORDS)
        claims = split_into_words(topic_data['claims'], MAX_WORDS)
        
        topic_splits[topic_id] = {
            'abstract': abstract,
            'description': description,
            'claims': claims
        }
    
    except Exception as e:
        print(f"Error processing topic {file}: {e}")

print(f"Processed {len(topic_splits)} topics")

# Save topic splits
save_pickle(topic_splits, OUTPUT_DIR / "topic_splits.pkl")

print("\n" + "=" * 80)
print("Document splitting complete!")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Files created:")
print(f"  - idf_dict.pkl")
print(f"  - document_splits.pkl ({len(doc_splits)} documents)")
print(f"  - topic_splits.pkl ({len(topic_splits)} topics)")
print("=" * 80)
