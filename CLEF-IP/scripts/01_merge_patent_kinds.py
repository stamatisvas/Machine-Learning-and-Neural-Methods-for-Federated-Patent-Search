"""
Step 1: Merge different patent document kinds into single patent documents.

This script combines different patent document kinds (e.g., A1, B2, etc.)
that belong to the same patent into a single merged XML document.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.xml_utils import XMLCombiner
import config


def merge_patent_kinds():
    """Merge different patent document kinds."""
    # Create output directory structure
    for i in range(10):
        for j in range(10):
            folder_path = config.MERGED_XML_PATH / f"{i}{j}"
            folder_path.mkdir(parents=True, exist_ok=True)
    
    i = 0
    j = 0
    k = 0
    
    error_log = config.BASE_DIR / 'data' / 'merge_errors.txt'
    
    with open(error_log, 'w', encoding='utf-8') as error_writer:
        for subdir, dirs, files in os.walk(config.CLEF_IP_RAW_PATH):
            if files:
                file_list = [
                    os.path.join(subdir, file) for file in files
                ]
                
                try:
                    # Combine all XML files in the directory
                    new_xml = XMLCombiner(file_list).combine()
                    root = new_xml.getroot()
                    
                    # Get patent ID (remove last 3 characters which indicate kind)
                    patent_id = root.attrib["ucid"][:-3]
                    
                    # Write merged XML
                    output_file = config.MERGED_XML_PATH / f"{i}{j}" / f"{patent_id}.xml"
                    new_xml.write(str(output_file), encoding="utf-8")
                    
                    # Remove DTD declaration if present
                    with open(output_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    dtd_pattern = '<!DOCTYPE patent-document'
                    if dtd_pattern in content:
                        content = content.split(dtd_pattern, 1)[0] + content.split(dtd_pattern, 1)[1].split('>', 1)[1]
                    
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    k += 1
                    if k > 17687:  # Approximate number per folder
                        j += 1
                        k = 0
                        if j == 10:
                            j = 0
                            i += 1
                            
                    if k % 1000 == 0:
                        print(f"Processed {k} patents in folder {i}{j}")
                        
                except Exception as e:
                    error_writer.write(f"{subdir}: {str(e)}\n")
                    continue
    
    print(f"Merge complete. Errors logged to: {error_log}")


if __name__ == '__main__':
    merge_patent_kinds()
