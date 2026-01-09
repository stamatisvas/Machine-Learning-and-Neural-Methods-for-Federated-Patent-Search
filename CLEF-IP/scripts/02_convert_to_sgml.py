"""
Step 2: Convert merged XML patents to SGML format (TREC format).

This script converts merged XML patent documents to SGML format suitable
for indexing with Anserini.
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


def convert_to_sgml():
    """Convert XML patents to SGML format."""
    # Create output directory structure
    for folder in os.listdir(config.MERGED_XML_PATH):
        output_folder = config.SGML_PATH / folder
        output_folder.mkdir(parents=True, exist_ok=True)
    
    folders = os.listdir(config.MERGED_XML_PATH)
    
    for folder in folders:
        folder_path = config.MERGED_XML_PATH / folder
        output_folder = config.SGML_PATH / folder
        
        if not folder_path.is_dir():
            continue
        
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
                
                patent_id = patent_doc.get('ucid', '')[:-3]  # Remove kind code
                
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
                    
                    # Extract and write IPC classifications
                    ipc_codes = extract_ipc_classifications(root)
                    if ipc_codes:
                        writer.write("<IPC-CLASSIFICATIONS>\n")
                        for ipc in ipc_codes:
                            writer.write(f"{ipc} ")
                        writer.write("\n</IPC-CLASSIFICATIONS>\n")
                    
                    # Extract and write title
                    title = extract_title(root)
                    if title:
                        writer.write("<TITLE>\n")
                        writer.write(f"{clean_text(title).lower()}\n")
                        writer.write("</TITLE>\n")
                    
                    # Extract and write applicant
                    applicant = extract_applicant(root)
                    if applicant:
                        writer.write("<APPLICANT>\n")
                        writer.write(f"{applicant}")
                        writer.write("\n</APPLICANT>\n")
                    
                    # Extract and write inventor
                    inventor = extract_inventor(root)
                    if inventor:
                        writer.write("<INVENTOR>\n")
                        writer.write(f"{inventor}")
                        writer.write("\n</INVENTOR>\n")
                    
                    # Extract and write abstract
                    abstract = extract_abstract(root)
                    if abstract:
                        writer.write("<ABSTRACT>\n")
                        writer.write(f"{clean_text(abstract).lower()}\n")
                        writer.write("</ABSTRACT>\n")
                    
                    # Extract and write description (first 500 words)
                    description = extract_description(root)
                    if description:
                        words = description.split()[:config.DESCRIPTION_MAX_WORDS]
                        writer.write("<DESCRIPTION>\n")
                        writer.write(f"{' '.join(words)}\n")
                        writer.write("</DESCRIPTION>\n")
                    
                    # Extract and write claims
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


def extract_ipc_classifications(root):
    """Extract IPC classifications from XML."""
    ipc_list = []
    
    # Get main classifications
    main_classifications = root.find_all('main-classification')
    for mc in main_classifications:
        if mc.text:
            code = clean_text(mc.text).split()[0]
            ipc_list.append(f'<main>{code}</main>')
    
    # Get further classifications
    further_classifications = root.find_all('further-classification')
    for fc in further_classifications:
        if fc.text:
            code = clean_text(fc.text).split()[0]
            if code not in ipc_list:
                ipc_list.append(code)
    
    # Get IPCR classifications
    ipcr = root.find_all('classification-ipcr')
    for ipc in ipcr:
        if ipc.text:
            code = clean_text(ipc.text).split()[0]
            if code not in ipc_list:
                ipc_list.append(code)
    
    return list(set(ipc_list))


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
    
    return '<sep>'.join(applicant_text) if applicant_text else None


def extract_inventor(root):
    """Extract inventor information."""
    inventors = root.find_all('inventor')
    if not inventors:
        return None
    
    inventor_text = []
    for inv in inventors:
        if inv.text:
            inventor_text.append(clean_text(inv.text).lower())
    
    return '<sep>'.join(inventor_text) if inventor_text else None


def extract_abstract(root):
    """Extract abstract from XML, preferring English."""
    abstracts = root.find_all('abstract')
    
    # First try English
    for abstr in abstracts:
        if abstr.get('lang') == 'EN' and abstr.text and len(abstr.text) >= 2:
            return abstr.text
    
    # If no English, try to translate first available
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
    convert_to_sgml()
