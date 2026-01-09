"""
Step 4: Split patents by IPC codes at level 3 (subclass level).

This script creates separate directories for each IPC subclass code
and copies patents to the appropriate directories based on their
IPC classifications.
"""

import os
import sys
from pathlib import Path
from shutil import copy

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config


def load_classifications():
    """Load IPC classifications from file."""
    classifications = {}
    
    with open(config.CLASSIFICATIONS_FILE, 'r', encoding='utf-8') as reader:
        for line in reader:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                patent_id = parts[0]
                ipc_codes = parts[1:-1]  # Exclude empty last element
                classifications[patent_id] = ipc_codes
    
    return classifications


def split_by_ipc_level3():
    """Split patents by IPC codes at level 3."""
    classifications = load_classifications()
    
    # Create output directories for each IPC code
    for patent_id, ipc_codes in classifications.items():
        for ipc_code in ipc_codes:
            ipc_dir = config.SPLIT3_DATA_PATH / ipc_code
            ipc_dir.mkdir(parents=True, exist_ok=True)
    
    # Process all SGML files
    for subdir, dirs, files in os.walk(config.SGML_PATH):
        if files:
            for file in files:
                patent_id = file.replace('.txt', '').replace('\n', '')
                
                if patent_id in classifications:
                    classification_list = classifications[patent_id]
                    
                    for ipc_code in classification_list:
                        source_file = os.path.join(subdir, file)
                        target_dir = config.SPLIT3_DATA_PATH / ipc_code
                        target_file = target_dir / file
                        
                        # Copy file to IPC code directory
                        copy(source_file, target_file)
    
    print(f"Patents split by IPC codes to: {config.SPLIT3_DATA_PATH}")


if __name__ == '__main__':
    split_by_ipc_level3()
