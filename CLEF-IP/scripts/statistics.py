"""
Statistics script: Compute statistics about the CLEF-IP dataset.

This script analyzes the prepared SGML files and computes statistics
about the presence of title, abstract, description, and claims fields.
"""

import os
import sys
from pathlib import Path
from bs4 import BeautifulSoup

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config


def compute_statistics():
    """Compute statistics about the dataset."""
    stats = {
        'total_patents': 0,
        'with_title': 0,
        'with_abstract': 0,
        'with_description': 0,
        'with_claims': 0,
        'with_all_fields': 0
    }
    
    sgml_path = config.SGML_PATH
    
    if not sgml_path.exists():
        print(f"SGML path does not exist: {sgml_path}")
        return
    
    folders = os.listdir(sgml_path)
    
    for folder in folders:
        folder_path = sgml_path / folder
        
        if not folder_path.is_dir():
            continue
        
        print(f"Processing folder: {folder}")
        
        files = os.listdir(folder_path)
        
        for file in files:
            try:
                with open(folder_path / file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                root = BeautifulSoup(content, 'html.parser')
                
                has_title = root.find('title') is not None
                has_abstract = root.find('abstract') is not None
                has_description = root.find('description') is not None
                has_claims = root.find('claims') is not None
                
                stats['total_patents'] += 1
                
                if has_title:
                    stats['with_title'] += 1
                if has_abstract:
                    stats['with_abstract'] += 1
                if has_description:
                    stats['with_description'] += 1
                if has_claims:
                    stats['with_claims'] += 1
                
                if has_title and has_abstract and has_description and has_claims:
                    stats['with_all_fields'] += 1
                    
            except Exception as e:
                print(f"Error processing {file}: {e}")
                continue
    
    # Print statistics
    print("\n" + "="*50)
    print("Dataset Statistics")
    print("="*50)
    print(f"Total patents: {stats['total_patents']}")
    print(f"With title: {stats['with_title']}")
    print(f"With abstract: {stats['with_abstract']}")
    print(f"With description: {stats['with_description']}")
    print(f"With claims: {stats['with_claims']}")
    print(f"With all fields: {stats['with_all_fields']}")
    print("="*50)
    
    # Calculate percentages
    if stats['total_patents'] > 0:
        print("\nPercentages:")
        print(f"Title: {stats['with_title']/stats['total_patents']*100:.2f}%")
        print(f"Abstract: {stats['with_abstract']/stats['total_patents']*100:.2f}%")
        print(f"Description: {stats['with_description']/stats['total_patents']*100:.2f}%")
        print(f"Claims: {stats['with_claims']/stats['total_patents']*100:.2f}%")
        print(f"All fields: {stats['with_all_fields']/stats['total_patents']*100:.2f}%")


if __name__ == '__main__':
    compute_statistics()
