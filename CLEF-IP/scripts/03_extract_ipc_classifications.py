"""
Step 3: Extract IPC classifications at level 3 (subclass level).

This script extracts IPC codes from merged XML patents and writes them
to a classifications file for splitting.
"""

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.text_utils import extract_ipc_codes_level3
import config


def extract_ipc_classifications():
    """Extract IPC classifications from merged XML files."""
    classifications_file = config.CLASSIFICATIONS_FILE
    
    with open(classifications_file, 'w', encoding='utf-8') as writer:
        for subdir, dirs, files in os.walk(config.MERGED_XML_PATH):
            if files:
                for file in files:
                    try:
                        patent_path = os.path.join(subdir, file)
                        patent = ET.parse(patent_path)
                        root = patent.getroot()
                        
                        patent_id = root.attrib["ucid"][:-3]  # Remove kind code
                        
                        # Extract IPC codes at level 3 (subclass level)
                        ipc_codes = extract_ipc_codes_level3(root)
                        
                        if ipc_codes:
                            writer.write(patent_id + '\t')
                            for ipc in ipc_codes:
                                writer.write(ipc + '\t')
                            writer.write('\n')
                    except Exception as e:
                        print(f"Error processing {file}: {e}")
                        continue
    
    print(f"Classifications extracted to: {classifications_file}")


if __name__ == '__main__':
    extract_ipc_classifications()
