"""
Step 6: Create representation sets using query-based sampling.

This script creates representation sets for each collection using
query-based sampling and creates a centralized index from all samples.
"""

import os
import sys
import subprocess
import itertools
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from pyserini.search import SimpleSearcher
from pyserini import index as pyserini_index
import config


def choose_random_word(collection_name):
    """
    Choose a random word from the collection index.
    
    This function selects a random word that:
    - Starts with a random letter
    - Has length > 3
    - Has collection frequency > 20
    - Contains only allowed characters
    
    If a word is found and returns results, it extracts a word from
    the document vector of the first result for better word selection.
    
    Args:
        collection_name: Name of the collection
        
    Returns:
        Random word suitable for querying
    """
    allowed = list('abcdefghijklmnopqrstuvwxyz')
    not_allowed = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
                   "'", '.', '_', ':']
    
    collection_path = config.SPLIT3_INDEX_PATH / collection_name
    index_reader = pyserini_index.IndexReader(str(collection_path))
    searcher = SimpleSearcher(str(collection_path))
    
    random_letter = allowed[np.random.randint(0, len(allowed))]
    used_words = []
    
    # Find initial suitable word
    word = None
    for term in itertools.islice(index_reader.terms(), 10000000):
        if (len(term.term) > 3 and term.cf > 20 and 
            term.term[0] == random_letter and term.term not in used_words):
            # Check if word contains disallowed characters
            not_suitable = False
            for char in term.term:
                if char in not_allowed:
                    not_suitable = True
                    break
            if not_suitable:
                continue
            word = term.term
            break
    
    # Search with initial word
    results = searcher.search(word, config.SAMPLING_RESULTS_PER_QUERY) if word else []
    
    # If no results, try another random letter
    while len(results) < 1:
        used_words.append(word) if word else None
        random_letter = allowed[np.random.randint(0, len(allowed))]
        word = None
        
        for term in itertools.islice(index_reader.terms(), 10000000):
            if (len(term.term) > 3 and term.cf > 20 and 
                term.term[0] == random_letter and term.term not in used_words):
                not_suitable = False
                for char in term.term:
                    if char in not_allowed:
                        not_suitable = True
                        break
                if not_suitable:
                    continue
                word = term.term
                break
        
        if word:
            results = searcher.search(word, config.SAMPLING_RESULTS_PER_QUERY)
        else:
            break
    
    if len(results) < 1:
        return None
    
    # Get better word from document vector of first result
    doc = searcher.doc(results[0].docid)
    doc_vector = index_reader.get_document_vector(doc.docid())
    
    # Find word with high term frequency
    selected_word = None
    for term, count in doc_vector.items():
        if count > 10 and len(term) > 3:
            selected_word = term
            break
        elif count > 5 and len(term) > 3 and not selected_word:
            selected_word = term
    
    return selected_word if selected_word else word


def query_based_sampling():
    """Perform query-based sampling for all collections."""
    # Create output directories
    config.REPRESENTATIONS_INDEX_PATH.mkdir(parents=True, exist_ok=True)
    config.CENTRALIZED_INDEX_PATH.mkdir(parents=True, exist_ok=True)
    temp_dir = config.BASE_DIR / 'data' / 'temporary'
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    collections = os.listdir(config.SPLIT3_INDEX_PATH)
    total = len(collections)
    
    for idx, collection in enumerate(collections):
        print(f"Processing collection {idx+1}/{total}: {collection}")
        
        try:
            collection_path = config.SPLIT3_INDEX_PATH / collection
            if not collection_path.is_dir():
                continue
            
            # Create collection representation directory
            rep_dir = config.REPRESENTATIONS_INDEX_PATH / collection
            rep_dir.mkdir(parents=True, exist_ok=True)
            temp_collection_dir = temp_dir / collection
            temp_collection_dir.mkdir(parents=True, exist_ok=True)
            
            index_reader = pyserini_index.IndexReader(str(collection_path))
            searcher = SimpleSearcher(str(collection_path))
            
            # Initial random word query
            random_word = choose_random_word(collection)
            if not random_word:
                print(f"  Skipping {collection}: No suitable words found")
                continue
            
            # Get initial results
            first_four_results = searcher.search(random_word, config.SAMPLING_RESULTS_PER_QUERY)
            length = len(first_four_results)
            
            while length < 1:
                random_word = choose_random_word(collection)
                if not random_word:
                    print(f"  Skipping {collection}: No suitable words found")
                    break
                first_four_results = searcher.search(random_word, config.SAMPLING_RESULTS_PER_QUERY)
                length = len(first_four_results)
            
            if length < 1:
                continue
            
            # Collect documents for sampling
            sampled_doc_ids = []
            for result in first_four_results:
                sampled_doc_ids.append(result.docid)
            
            # Query-based sampling iterations
            counter = config.SAMPLING_ITERATIONS
            while counter != 0:
                # Find words from document vector
                words = []
                random_idx = np.random.randint(0, length)
                doc_vector = index_reader.get_document_vector(first_four_results[random_idx].docid)
                
                for term, count in doc_vector.items():
                    if count > 10 and len(term) > 3:
                        words.append(term)
                
                # If no words found, try another document
                while len(words) < 1:
                    random_idx = np.random.randint(0, length)
                    doc_vector = index_reader.get_document_vector(first_four_results[random_idx].docid)
                    for term, count in doc_vector.items():
                        if count > 10 and len(term) > 3:
                            words.append(term)
                
                if not words:
                    counter -= 1
                    continue
                
                # Choose random word and query
                random_word = words[np.random.randint(0, len(words))]
                first_four_results = searcher.search(random_word, config.SAMPLING_RESULTS_PER_QUERY)
                length = len(first_four_results)
                
                # If no results, try another word
                while length < 1 and len(words) > 1:
                    random_word = words[np.random.randint(0, len(words))]
                    first_four_results = searcher.search(random_word, config.SAMPLING_RESULTS_PER_QUERY)
                    length = len(first_four_results)
                
                if length < 1:
                    counter -= 1
                    continue
                
                # Add results to sampled documents
                for result in first_four_results:
                    sampled_doc_ids.append(result.docid)
                
                counter -= 1
            
            # Create set of unique document IDs
            final_set_of_patents_for_sampling = set(sampled_doc_ids)
            
            # Extract sampled documents and write to temporary SGML
            for doc_id in final_set_of_patents_for_sampling:
                try:
                    document = searcher.doc(doc_id)
                    content = document.contents()
                    content = content.replace('\n', ' ').replace('\t', ' ')
                    
                    output_file = temp_collection_dir / f"{doc_id}.txt"
                    with open(output_file, 'w', encoding='utf-8') as writer:
                        writer.write('<DOC>\n')
                        writer.write('<DOCNO>\n')
                        writer.write(f'{doc_id}\n')
                        writer.write('</DOCNO>\n')
                        writer.write('<TEXT>\n')
                        writer.write(content)
                        writer.write('\n</TEXT>\n')
                        writer.write('</DOC>\n')
                except Exception as e:
                    print(f"  Error extracting document {doc_id}: {e}")
                    continue
            
            # Create index for collection representation
            if len(list(temp_collection_dir.glob('*.txt'))) > 0:
                cmd = [
                    'sh', config.INDEX_COLLECTION,
                    '-collection', 'TrecCollection',
                    '-input', str(temp_collection_dir),
                    '-index', str(rep_dir),
                    '-generator', 'DefaultLuceneDocumentGenerator',
                    '-threads', '8',
                    '-storeContents',
                    '-storeDocvectors',
                    '-quiet'
                ]
                subprocess.run(cmd, check=True)
            
        except Exception as e:
            print(f"  Error processing {collection}: {e}")
            continue
    
    # Create centralized index from all samples
    print("Creating centralized index...")
    cmd = [
        'sh', config.INDEX_COLLECTION,
        '-collection', 'TrecCollection',
        '-input', str(temp_dir),
        '-index', str(config.CENTRALIZED_INDEX_PATH),
        '-generator', 'DefaultLuceneDocumentGenerator',
        '-threads', '8',
        '-storeContents',
        '-storeDocvectors',
        '-quiet'
    ]
    subprocess.run(cmd, check=True)
    
    # Clean up temporary directory
    import shutil
    shutil.rmtree(temp_dir)
    
    print(f"Query-based sampling complete!")
    print(f"Representation indices: {config.REPRESENTATIONS_INDEX_PATH}")
    print(f"Centralized index: {config.CENTRALIZED_INDEX_PATH}")


if __name__ == '__main__':
    query_based_sampling()
