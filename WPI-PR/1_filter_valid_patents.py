#!/usr/bin/env python3
"""
Step 1: Filter Valid Patents
Keeps only patents with all required English sections
"""

import os
import shutil
import yaml
from bs4 import BeautifulSoup

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

INPUT_DIR = config['input_dir']
OUTPUT_DIR = os.path.join(config['output_dir'], 'wpi_valid')
DOCS_PER_FOLDER = config['docs_per_folder']
LANGUAGE = config['language']

print(f"Input: {INPUT_DIR}")
print(f"Output: {OUTPUT_DIR}")
print("-" * 80)

# Create output directory
folder_counter = 0
os.makedirs(os.path.join(OUTPUT_DIR, str(folder_counter)), exist_ok=True)

valid_count = 0
total_count = 0

for root, dirs, files in os.walk(INPUT_DIR):
    if not files:
        continue
    
    for file in files:
        if not file.endswith('.xml'):
            continue
        
        total_count += 1
        if total_count % 1000 == 0:
            print(f"Processed: {total_count}, Valid: {valid_count}")
        
        file_path = os.path.join(root, file)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            soup = BeautifulSoup(content, 'xml')
            
            # Check for all required English sections
            required_sections = 0
            
            # Title
            titles = soup.find_all('invention-title')
            for title in titles:
                if title.get('lang') == LANGUAGE and len(title.text.strip()) > 0:
                    required_sections += 1
                    break
            
            # Abstract
            abstracts = soup.find_all('abstract')
            for abstract in abstracts:
                if abstract.get('lang') == LANGUAGE and len(abstract.text.strip()) > 2:
                    required_sections += 1
                    break
            
            # Description
            descriptions = soup.find_all('description')
            for desc in descriptions:
                if desc.get('lang') == LANGUAGE and len(desc.text.strip()) > 0:
                    required_sections += 1
                    break
            
            # Claims
            claims_list = soup.find_all('claims')
            for claims in claims_list:
                if claims.get('lang') == LANGUAGE and len(claims.text.strip()) > 0:
                    required_sections += 1
                    break
            
            # Keep patent if all 4 sections are present
            if required_sections == 4:
                valid_count += 1
                output_path = os.path.join(OUTPUT_DIR, str(folder_counter), file)
                shutil.copyfile(file_path, output_path)
                
                # Create new folder if needed
                if valid_count % DOCS_PER_FOLDER == 0:
                    folder_counter += 1
                    os.makedirs(os.path.join(OUTPUT_DIR, str(folder_counter)), exist_ok=True)
                    print(f"Created folder {folder_counter}")
        
        except Exception as e:
            print(f"Error processing {file}: {e}")

print("-" * 80)
print(f"Complete! Processed: {total_count}, Valid: {valid_count}")
print(f"Output: {OUTPUT_DIR}")
