"""
Step 2a (Optional): Convert merged XML patents to SGML format with variants.

This script creates multiple SGML variants:
- Everything (full document)
- Titles and abstracts only
- Descriptions only
- Claims only

Also includes CPC classifications in addition to IPC/IPCR.
"""

import os
import sys
from pathlib import Path
from bs4 import BeautifulSoup
from googletrans import Translator

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.text_utils import clean_text, clean_text_aggressive
import config

translator = Translator()


def convert_to_sgml_variants():
    """Convert XML patents to SGML format with multiple variants."""
    # Create output directory structure for variants
    sgml_everything = config.SGML_PATH / 'everything'
    sgml_abstract = config.SGML_PATH / 'titles_abstracts_only'
    sgml_description = config.SGML_PATH / 'descriptions_only'
    sgml_claims = config.SGML_PATH / 'claims_only'
    
    for base_path in [sgml_everything, sgml_abstract, sgml_description, sgml_claims]:
        base_path.mkdir(parents=True, exist_ok=True)
    
    folder = 0
    file_count = 0
    
    for subdir, dirs, files in os.walk(config.MERGED_XML_PATH):
        if files:
            if file_count > 19999:
                file_count = 0
                folder += 1
                for base_path in [sgml_everything, sgml_abstract, sgml_description, sgml_claims]:
                    (base_path / str(folder)).mkdir(parents=True, exist_ok=True)
            
            for file in files:
                file_count += 1
                try:
                    with open(os.path.join(subdir, file), 'r', encoding='utf-8') as f:
                        patent_content = f.read()
                    
                    root = BeautifulSoup(patent_content, 'xml')
                    patent_doc = root.find('patent-document')
                    
                    if not patent_doc:
                        continue
                    
                    patent_id = patent_doc.get('ucid', '')[:-3]  # Remove kind code
                    date = patent_doc.get('date', '')
                    
                    # Create all variant files
                    everything_file = sgml_everything / str(folder) / f"{patent_id}.txt"
                    abstract_file = sgml_abstract / str(folder) / f"{patent_id}.txt"
                    description_file = sgml_description / str(folder) / f"{patent_id}.txt"
                    claims_file = sgml_claims / str(folder) / f"{patent_id}.txt"
                    
                    with open(everything_file, 'w', encoding='utf-8') as everything_writer, \
                         open(abstract_file, 'w', encoding='utf-8') as abstract_writer, \
                         open(description_file, 'w', encoding='utf-8') as description_writer, \
                         open(claims_file, 'w', encoding='utf-8') as claims_writer:
                        
                        # Write DOC headers
                        for writer in [everything_writer, abstract_writer, description_writer, claims_writer]:
                            writer.write("<DOC>\n")
                            writer.write("<DOCNO>\n")
                            writer.write(f"{patent_id}\n")
                            writer.write("</DOCNO>\n")
                            writer.write("<TEXT>\n")
                        
                        # Write date to everything only
                        everything_writer.write("<DATE>\n")
                        everything_writer.write(f"{date}\n")
                        everything_writer.write("</DATE>\n")
                        
                        # Extract and write IPCR classifications
                        ipcr = root.find_all('classification-ipcr')
                        if ipcr:
                            everything_writer.write("<IPCR-CLASSIFICATIONS>\n")
                            for i in ipcr:
                                if i.text:
                                    everything_writer.write(f"{i.text}\n")
                            everything_writer.write("</IPCR-CLASSIFICATIONS>\n")
                        
                        # Extract and write CPC classifications
                        cpc = root.find_all('classification-cpc')
                        if cpc:
                            everything_writer.write("<CPC-CLASSIFICATIONS>\n")
                            for i in cpc:
                                if i.text:
                                    everything_writer.write(f"{i.text}\n")
                            everything_writer.write("</CPC-CLASSIFICATIONS>\n")
                        
                        # Extract and write title
                        title = root.find_all('invention-title')
                        title_text = None
                        for title_it in title:
                            if title_it.get('lang') == 'EN' and title_it.text:
                                title_text = clean_text_aggressive(title_it.text)
                                everything_writer.write("<TITLE>\n")
                                everything_writer.write(f"{title_text}\n")
                                everything_writer.write("</TITLE>\n")
                                break
                        
                        # Extract and write applicant
                        applicant = root.find_all('applicant')
                        if applicant:
                            everything_writer.write("<APPLICANT>\n")
                            for appl in applicant:
                                if appl.text:
                                    appl_text = clean_text_aggressive(appl.text)
                                    everything_writer.write(f"{appl_text}\n")
                            everything_writer.write("</APPLICANT>\n")
                        
                        # Extract and write inventor
                        inventors = root.find_all('inventor')
                        if inventors:
                            everything_writer.write("<INVENTOR>\n")
                            for inv in inventors:
                                if inv.text:
                                    inv_text = clean_text_aggressive(inv.text)
                                    everything_writer.write(f"{inv_text}\n")
                            everything_writer.write("</INVENTOR>\n")
                        
                        # Extract and write abstract
                        abstr = root.find_all('abstract')
                        abstract_text = None
                        for abstr_it in abstr:
                            if abstr_it.get('lang') == 'EN' and abstr_it.text:
                                abstract_text = clean_text_aggressive(abstr_it.text)
                                everything_writer.write("<ABSTRACT>\n")
                                everything_writer.write(f"{abstract_text}\n")
                                everything_writer.write("</ABSTRACT>\n")
                                
                                # Write to abstract variant (with title if available)
                                abstract_writer.write("<ABSTRACT>\n")
                                if title_text:
                                    abstract_writer.write(f"{title_text} {abstract_text}\n")
                                else:
                                    abstract_writer.write(f"{abstract_text}\n")
                                abstract_writer.write("</ABSTRACT>\n")
                                break
                        
                        if title_text and not abstract_text:
                            # Only title available
                            abstract_writer.write("<ABSTRACT>\n")
                            abstract_writer.write(f"{title_text}\n")
                            abstract_writer.write("</ABSTRACT>\n")
                        
                        # Extract and write description
                        desc = root.find_all('description')
                        for desc_it in desc:
                            if desc_it.get('lang') == 'EN' and desc_it.text:
                                desc_text = clean_text_aggressive(desc_it.text)
                                everything_writer.write("<DESCRIPTION>\n")
                                everything_writer.write(f"{desc_text}\n")
                                everything_writer.write("</DESCRIPTION>\n")
                                
                                # Write to description variant
                                description_writer.write("<DESCRIPTION>\n")
                                description_writer.write(f"{desc_text}\n")
                                description_writer.write("</DESCRIPTION>\n")
                                break
                        
                        # Extract and write claims
                        clm = root.find_all('claims')
                        for clm_it in clm:
                            if clm_it.get('lang') == 'EN' and clm_it.text:
                                clm_text = clean_text_aggressive(clm_it.text)
                                everything_writer.write("<CLAIMS>\n")
                                everything_writer.write(f"{clm_text}\n")
                                everything_writer.write("</CLAIMS>\n")
                                
                                # Write to claims variant
                                claims_writer.write("<CLAIMS>\n")
                                claims_writer.write(f"{clm_text}\n")
                                claims_writer.write("</CLAIMS>\n")
                                break
                        
                        # Close all files
                        for writer in [everything_writer, abstract_writer, description_writer, claims_writer]:
                            writer.write("</TEXT>\n")
                            writer.write("</DOC>\n")
                            
                except Exception as e:
                    print(f"Error processing {file}: {e}")
                    continue
            
            if file_count % 1000 == 0:
                print(f"Processed {file_count} files in folder {folder}")
    
    print("Variant conversion complete!")


if __name__ == '__main__':
    convert_to_sgml_variants()
