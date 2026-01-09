"""
Step 2b (Optional): Convert merged XML patents to SGML without IPC classifications.

This script creates SGML files excluding IPC classifications.
"""

import os
import sys
from pathlib import Path
from bs4 import BeautifulSoup
from googletrans import Translator

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.text_utils import clean_text
import config

translator = Translator()


def convert_without_ipc():
    """Convert XML patents to SGML format without IPC classifications."""
    output_path = config.SGML_PATH / 'without_ipc'
    output_path.mkdir(parents=True, exist_ok=True)
    
    folders = os.listdir(config.MERGED_XML_PATH)
    
    for folder in folders:
        folder_path = config.MERGED_XML_PATH / folder
        output_folder = output_path / folder
        
        if not folder_path.is_dir():
            continue
        
        output_folder.mkdir(parents=True, exist_ok=True)
        
        print(f'Processing folder: {folder}')
        patents = os.listdir(folder_path)
        
        for file in patents:
            try:
                with open(folder_path / file, 'r', encoding='utf-8') as f:
                    patent_content = f.read()
                
                root = BeautifulSoup(patent_content, 'xml')
                patent_doc = root.find('patent-document')
                
                if not patent_doc:
                    continue
                
                patent_id = patent_doc.get('ucid', '')[:-3]
                output_file = output_folder / f"{patent_id}.txt"
                
                with open(output_file, 'w', encoding='utf-8') as writer:
                    writer.write("<DOC>\n")
                    writer.write("<DOCNO>\n")
                    writer.write(f"{patent_id}\n")
                    writer.write("</DOCNO>\n")
                    writer.write("<TEXT>\n")
                    writer.write("<DATE>\n")
                    writer.write(f"{patent_doc.get('date', '')}\n")
                    writer.write("</DATE>\n")
                    
                    # Write title (no IPC)
                    title = extract_title(root)
                    if title:
                        writer.write("<TITLE>\n")
                        writer.write(f"{clean_text(title).lower()}\n")
                        writer.write("</TITLE>\n")
                    
                    # Write applicant
                    applicant = extract_applicant(root)
                    if applicant:
                        writer.write("<APPLICANT>\n")
                        writer.write(f"{applicant}\n")
                        writer.write("</APPLICANT>\n")
                    
                    # Write inventor
                    inventor = extract_inventor(root)
                    if inventor:
                        writer.write("<INVENTOR>\n")
                        writer.write(f"{inventor}\n")
                        writer.write("</INVENTOR>\n")
                    
                    # Write abstract
                    abstract = extract_abstract(root)
                    if abstract:
                        writer.write("<ABSTRACT>\n")
                        writer.write(f"{clean_text(abstract).lower()}\n")
                        writer.write("</ABSTRACT>\n")
                    
                    # Write description
                    description = extract_description(root)
                    if description:
                        words = description.split()[:config.DESCRIPTION_MAX_WORDS]
                        writer.write("<DESCRIPTION>\n")
                        writer.write(f"{' '.join(words)}\n")
                        writer.write("</DESCRIPTION>\n")
                    
                    # Write claims
                    claims = extract_claims(root)
                    if claims:
                        writer.write("<CLAIMS>\n")
                        writer.write(f"{clean_text(claims).lower()}\n")
                        writer.write("</CLAIMS>\n")
                    
                    writer.write("</TEXT>\n")
                    writer.write("</DOC>\n")
                    
            except Exception as e:
                print(f"Error processing {file}: {e}")
                continue
        
        print(f"Completed folder: {folder}")


def extract_title(root):
    """Extract title from XML."""
    titles = root.find_all('invention-title')
    for title in titles:
        if title.get('lang') == 'EN' and title.text:
            return title.text
    return None


def extract_applicant(root):
    """Extract applicant information."""
    applicants = root.find_all('applicant')
    if not applicants:
        return None
    
    applicant_text = []
    for appl in applicants:
        if appl.text:
            applicant_text.append(clean_text(appl.text).lower())
    
    return '\n'.join(applicant_text) if applicant_text else None


def extract_inventor(root):
    """Extract inventor information."""
    inventors = root.find_all('inventor')
    if not inventors:
        return None
    
    inventor_text = []
    for inv in inventors:
        if inv.text:
            inventor_text.append(clean_text(inv.text).lower())
    
    return '\n'.join(inventor_text) if inventor_text else None


def extract_abstract(root):
    """Extract abstract from XML, preferring English."""
    abstracts = root.find_all('abstract')
    
    for abstr in abstracts:
        if abstr.get('lang') == 'EN' and abstr.text and len(abstr.text) >= 2:
            return abstr.text
    
    for abstr in abstracts:
        if abstr.text and len(abstr.text) >= 2:
            try:
                translated = translator.translate(abstr.text)
                return translated.text
            except:
                continue
    
    return None


def extract_description(root):
    """Extract description from XML, preferring English."""
    descriptions = root.find_all('description')
    
    for desc in descriptions:
        if desc.get('lang') == 'EN' and desc.text:
            return desc.text
    
    return None


def extract_claims(root):
    """Extract claims from XML, preferring English."""
    claims = root.find_all('claims')
    
    for clm in claims:
        if clm.get('lang') == 'EN' and clm.text:
            return clm.text
    
    return None


if __name__ == '__main__':
    convert_without_ipc()
