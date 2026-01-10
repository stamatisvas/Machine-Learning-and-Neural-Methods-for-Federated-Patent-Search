#!/usr/bin/env python3
"""
Step 5: Split by Sections (Optional)
Creates separate collections for abstracts, descriptions, and claims
"""

import os
import yaml
from bs4 import BeautifulSoup

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

INPUT_DIR = os.path.join(config['output_dir'], 'wpi_sgml', 'data')
OUTPUT_BASE = os.path.join(config['output_dir'], 'wpi_sgml')

ABSTRACT_DIR = os.path.join(OUTPUT_BASE, 'abstracts')
DESCRIPTION_DIR = os.path.join(OUTPUT_BASE, 'descriptions')
CLAIMS_DIR = os.path.join(OUTPUT_BASE, 'claims')

print(f"Input: {INPUT_DIR}")
print(f"Output (abstracts): {ABSTRACT_DIR}")
print(f"Output (descriptions): {DESCRIPTION_DIR}")
print(f"Output (claims): {CLAIMS_DIR}")
print("-" * 80)

processed = 0

for root, dirs, files in os.walk(INPUT_DIR):
    if not files:
        continue
    
    # Get folder name
    folder_name = os.path.basename(root)
    
    # Create output folders
    os.makedirs(os.path.join(ABSTRACT_DIR, folder_name), exist_ok=True)
    os.makedirs(os.path.join(DESCRIPTION_DIR, folder_name), exist_ok=True)
    os.makedirs(os.path.join(CLAIMS_DIR, folder_name), exist_ok=True)
    
    for file in files:
        if not file.endswith('.txt'):
            continue
        
        file_path = os.path.join(root, file)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            soup = BeautifulSoup(content, 'html.parser')
            
            # Extract document ID
            docno = soup.find('docno')
            if not docno:
                continue
            doc_id = docno.text.strip()
            
            # Extract sections
            title_elem = soup.find('title')
            abstract_elem = soup.find('abstract')
            description_elem = soup.find('description')
            claims_elem = soup.find('claims')
            
            title = title_elem.text.strip() if title_elem else ""
            abstract = abstract_elem.text.strip() if abstract_elem else ""
            description = description_elem.text.strip() if description_elem else ""
            claims = claims_elem.text.strip() if claims_elem else ""
            
            # Write abstract file (title + abstract)
            abstract_path = os.path.join(ABSTRACT_DIR, folder_name, file)
            with open(abstract_path, 'w', encoding='utf-8') as f:
                f.write("<DOC>\n")
                f.write(f"<DOCNO>\n{doc_id}\n</DOCNO>\n")
                f.write("<TEXT>\n")
                f.write("<ABSTRACT>\n")
                f.write(f"{title} {abstract}\n")
                f.write("</ABSTRACT>\n")
                f.write("</TEXT>\n")
                f.write("</DOC>\n")
            
            # Write description file
            description_path = os.path.join(DESCRIPTION_DIR, folder_name, file)
            with open(description_path, 'w', encoding='utf-8') as f:
                f.write("<DOC>\n")
                f.write(f"<DOCNO>\n{doc_id}\n</DOCNO>\n")
                f.write("<TEXT>\n")
                f.write("<DESCRIPTION>\n")
                f.write(f"{description}\n")
                f.write("</DESCRIPTION>\n")
                f.write("</TEXT>\n")
                f.write("</DOC>\n")
            
            # Write claims file
            claims_path = os.path.join(CLAIMS_DIR, folder_name, file)
            with open(claims_path, 'w', encoding='utf-8') as f:
                f.write("<DOC>\n")
                f.write(f"<DOCNO>\n{doc_id}\n</DOCNO>\n")
                f.write("<TEXT>\n")
                f.write("<CLAIMS>\n")
                f.write(f"{claims}\n")
                f.write("</CLAIMS>\n")
                f.write("</TEXT>\n")
                f.write("</DOC>\n")
            
            processed += 1
            if processed % 1000 == 0:
                print(f"Processed: {processed}")
        
        except Exception as e:
            print(f"Error processing {file}: {e}")

print("-" * 80)
print(f"Complete! Processed {processed} documents")
print(f"Output directories:")
print(f"  - {ABSTRACT_DIR}")
print(f"  - {DESCRIPTION_DIR}")
print(f"  - {CLAIMS_DIR}")
