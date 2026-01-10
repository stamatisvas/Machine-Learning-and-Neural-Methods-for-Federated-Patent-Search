#!/usr/bin/env python3
"""
Step 2: Convert XML to SGML Format
Converts filtered patents to SGML format with text cleaning
"""

import os
import re
import shutil
import yaml
from bs4 import BeautifulSoup

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

INPUT_DIR = os.path.join(config['output_dir'], 'wpi_valid')
OUTPUT_DIR = os.path.join(config['output_dir'], 'wpi_sgml', 'data')
TOPICS_DIR = os.path.join(config['output_dir'], 'wpi_sgml', 'topics')
DOCS_PER_FOLDER = 20000

# Create output directories
os.makedirs(TOPICS_DIR, exist_ok=True)

# Load queries if qrels exists
queries = set()
if os.path.exists('wpi_qrels.txt'):
    with open('wpi_qrels.txt', 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                queries.add(parts[0])
    print(f"Loaded {len(queries)} query IDs")

def clean_text(text):
    """Clean and normalize text"""
    text = text.lower()
    
    # Remove special characters
    chars_to_remove = [
        "\n", "\t", "'", "-", ".", "!", "@", "#", "$", "%", "&", "*", "=", "+",
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "(", ")", "{", "}", "[", "]", ",", '"', "?", "<", ">", ";", "¬", "°", ":"
    ]
    for char in chars_to_remove:
        text = text.replace(char, ' ')
    
    # Normalize whitespace
    text = re.sub(r' +', ' ', text)
    return text.strip()

print(f"Input: {INPUT_DIR}")
print(f"Output: {OUTPUT_DIR}")
print(f"Topics: {TOPICS_DIR}")
print("-" * 80)

folder_num = 0
os.makedirs(os.path.join(OUTPUT_DIR, str(folder_num)), exist_ok=True)

doc_count = 0
topic_count = 0

for root, dirs, files in os.walk(INPUT_DIR):
    if not files:
        continue
    
    for file in files:
        if not file.endswith('.xml'):
            continue
        
        file_path = os.path.join(root, file)
        patent_id = file.replace('.xml', '')
        
        # Check if this is a topic document
        if patent_id in queries:
            shutil.copy(file_path, TOPICS_DIR)
            topic_count += 1
            continue
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            soup = BeautifulSoup(content, 'xml')
            patent_doc = soup.find('patent-document')
            
            if not patent_doc:
                continue
            
            # Create new folder if needed
            if doc_count >= DOCS_PER_FOLDER:
                folder_num += 1
                doc_count = 0
                os.makedirs(os.path.join(OUTPUT_DIR, str(folder_num)), exist_ok=True)
                print(f"Created folder {folder_num}")
            
            doc_count += 1
            
            # Extract data
            doc_id = patent_doc.get('ucid', '')
            date = patent_doc.get('date', '')
            
            output_file = os.path.join(OUTPUT_DIR, str(folder_num), f"{doc_id}.txt")
            
            with open(output_file, 'w', encoding='utf-8') as out:
                out.write("<DOC>\n")
                out.write(f"<DOCNO>\n{doc_id}\n</DOCNO>\n")
                out.write("<TEXT>\n")
                out.write(f"<DATE>\n{date}\n</DATE>\n")
                
                # IPCR classifications
                ipcr = soup.find_all('classification-ipcr')
                if ipcr:
                    out.write("<IPCR-CLASSIFICATIONS>\n")
                    for code in ipcr:
                        out.write(f"{code.text}\n")
                    out.write("</IPCR-CLASSIFICATIONS>\n")
                
                # CPC classifications
                cpc = soup.find_all('classification-cpc')
                if cpc:
                    out.write("<CPC-CLASSIFICATIONS>\n")
                    for code in cpc:
                        out.write(f"{code.text}\n")
                    out.write("</CPC-CLASSIFICATIONS>\n")
                
                # Title
                titles = soup.find_all('invention-title')
                for title in titles:
                    if title.get('lang') == 'EN':
                        out.write("<TITLE>\n")
                        out.write(clean_text(title.text) + "\n")
                        out.write("</TITLE>\n")
                        break
                
                # Applicants
                applicants = soup.find_all('applicant')
                if applicants:
                    out.write("<APPLICANT>\n")
                    for app in applicants:
                        out.write(clean_text(app.text) + "\n")
                    out.write("</APPLICANT>\n")
                
                # Inventors
                inventors = soup.find_all('inventor')
                if inventors:
                    out.write("<INVENTOR>\n")
                    for inv in inventors:
                        out.write(clean_text(inv.text) + "\n")
                    out.write("</INVENTOR>\n")
                
                # Abstract
                abstracts = soup.find_all('abstract')
                for abstract in abstracts:
                    if abstract.get('lang') == 'EN':
                        out.write("<ABSTRACT>\n")
                        out.write(clean_text(abstract.text) + "\n")
                        out.write("</ABSTRACT>\n")
                        break
                
                # Description
                descriptions = soup.find_all('description')
                for desc in descriptions:
                    if desc.get('lang') == 'EN':
                        out.write("<DESCRIPTION>\n")
                        out.write(clean_text(desc.text) + "\n")
                        out.write("</DESCRIPTION>\n")
                        break
                
                # Claims
                claims_list = soup.find_all('claims')
                for claims in claims_list:
                    if claims.get('lang') == 'EN':
                        out.write("<CLAIMS>\n")
                        out.write(clean_text(claims.text) + "\n")
                        out.write("</CLAIMS>\n")
                        break
                
                out.write("</TEXT>\n")
                out.write("</DOC>\n")
            
            if (doc_count + topic_count) % 1000 == 0:
                print(f"Processed: {doc_count + topic_count} (Docs: {doc_count}, Topics: {topic_count})")
        
        except Exception as e:
            print(f"Error processing {file}: {e}")

print("-" * 80)
print(f"Complete! Documents: {doc_count}, Topics: {topic_count}")
print(f"Output: {OUTPUT_DIR}")
print(f"Topics: {TOPICS_DIR}")
