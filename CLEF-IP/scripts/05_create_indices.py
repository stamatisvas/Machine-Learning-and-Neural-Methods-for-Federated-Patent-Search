"""
Step 5: Create Anserini indices for split collections.

This script creates Anserini indices for each IPC code collection
in the split3_data directory.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config


def create_indices():
    """Create Anserini indices for all collections."""
    collections = os.listdir(config.SPLIT3_DATA_PATH)
    
    for collection in collections:
        collection_path = config.SPLIT3_DATA_PATH / collection
        index_path = config.SPLIT3_INDEX_PATH / collection
        
        if not collection_path.is_dir():
            continue
        
        print(f"Indexing collection: {collection}")
        
        # Create Anserini index command
        cmd = [
            'sh',
            config.INDEX_COLLECTION,
            '-collection', 'TrecCollection',
            '-input', str(collection_path),
            '-index', str(index_path),
            '-generator', 'DefaultLuceneDocumentGenerator',
            '-threads', '8',
            '-storePositions',
            '-storeDocvectors',
            '-storeContents',
            '-quiet'
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"Completed indexing: {collection}")
        except subprocess.CalledProcessError as e:
            print(f"Error indexing {collection}: {e}")
            continue
    
    print(f"All indices created in: {config.SPLIT3_INDEX_PATH}")


if __name__ == '__main__':
    create_indices()
