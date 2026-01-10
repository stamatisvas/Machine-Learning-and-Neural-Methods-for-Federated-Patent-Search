#!/usr/bin/env python3
"""
Step 3: Generate Ground Truth
Extracts patent citations to create ground truth data
"""

import os
import yaml
from bs4 import BeautifulSoup

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

INPUT_DIR = os.path.join(config['output_dir'], 'wpi_valid')
OUTPUT_FILE = 'wpi_ground_truths.txt'
MIN_CITATIONS = config['min_citations']

print(f"Input: {INPUT_DIR}")
print(f"Output: {OUTPUT_FILE}")
print(f"Minimum citations: {MIN_CITATIONS}")
print("-" * 80)

# Extract citations from all patents
citation_dict = {}
processed_count = 0

print("Extracting citations...")
for root, dirs, files in os.walk(INPUT_DIR):
    if not files:
        continue
    
    for file in files:
        if not file.endswith('.xml'):
            continue
        
        processed_count += 1
        if processed_count % 1000 == 0:
            print(f"Processed: {processed_count}")
        
        file_path = os.path.join(root, file)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            soup = BeautifulSoup(content, 'xml')
            citations = soup.find_all('patcit')
            
            if len(citations) == 0:
                continue
            
            citation_list = []
            for cit in citations:
                try:
                    ucid = cit.get('ucid')
                    if ucid:
                        citation_list.append(ucid)
                except:
                    continue
            
            if citation_list:
                patent_id = file.replace('.xml', '')
                citation_dict[patent_id] = citation_list
        
        except Exception as e:
            continue

print(f"Found {len(citation_dict)} patents with citations")

# Filter to keep only valid citations
print("Filtering valid citations...")
available_patents = set(citation_dict.keys())
valid_citations = {}

for patent_id, citations in citation_dict.items():
    # Skip if too few citations
    if len(citations) < MIN_CITATIONS:
        continue
    
    # Check if all cited patents are available
    all_available = all(cit in available_patents for cit in citations)
    
    if all_available:
        valid_citations[patent_id] = citations

print(f"Found {len(valid_citations)} valid queries")

# Write ground truth file
print(f"Writing ground truth file...")
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for patent_id, citations in valid_citations.items():
        citations_str = ' '.join(citations)
        f.write(f"{patent_id}<sep>{citations_str}\n")

total_relevant = sum(len(cits) for cits in valid_citations.values())
avg_relevant = total_relevant / len(valid_citations) if valid_citations else 0

print("-" * 80)
print(f"Complete!")
print(f"Queries: {len(valid_citations)}")
print(f"Total relevant documents: {total_relevant}")
print(f"Average relevant per query: {avg_relevant:.2f}")
print(f"Output: {OUTPUT_FILE}")
