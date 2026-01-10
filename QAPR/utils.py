"""
Utility Functions for QAPR
===========================
Helper functions for document processing, feature extraction, and evaluation.
"""

import os
import re
import yaml
import pickle
import numpy as np
from pathlib import Path
from bs4 import BeautifulSoup
from typing import Dict, List, Tuple
from collections import Counter
import math


def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_sgml_document(file_path: str) -> Dict[str, str]:
    """
    Parse SGML patent document and extract sections.
    
    Args:
        file_path: Path to SGML file
        
    Returns:
        Dictionary with doc_id and sections (abstract, description, claims)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    soup = BeautifulSoup(content, 'html.parser')
    
    doc_id_elem = soup.find('docno')
    doc_id = doc_id_elem.text.strip() if doc_id_elem else None
    
    abstract_elem = soup.find('abstract')
    description_elem = soup.find('description')
    claims_elem = soup.find('claims')
    
    return {
        'doc_id': doc_id,
        'abstract': abstract_elem.text.strip() if abstract_elem else "",
        'description': description_elem.text.strip() if description_elem else "",
        'claims': claims_elem.text.strip() if claims_elem else ""
    }


def split_into_words(text: str, max_words: int = 500) -> str:
    """
    Limit text to maximum number of words.
    
    Args:
        text: Input text
        max_words: Maximum number of words
        
    Returns:
        Text limited to max_words
    """
    words = text.split()
    if len(words) <= max_words:
        return text
    return ' '.join(words[:max_words])


def select_best_passage(text: str, passage_length: int, idf_dict: Dict[str, float]) -> str:
    """
    Split text into passages and select the one with highest average IDF.
    
    Args:
        text: Input text
        passage_length: Number of words per passage
        idf_dict: Dictionary of IDF values
        
    Returns:
        Best passage based on average IDF
    """
    words = text.split()
    
    if len(words) <= passage_length:
        return text
    
    # Split into passages
    passages = []
    for i in range(0, len(words), passage_length):
        passage = ' '.join(words[i:i + passage_length])
        passages.append(passage)
    
    # Calculate average IDF for each passage
    best_passage = passages[0]
    best_idf = 0
    
    for passage in passages:
        passage_words = passage.lower().split()
        idfs = [idf_dict.get(word, 0) for word in passage_words]
        avg_idf = np.mean(idfs) if idfs else 0
        
        if avg_idf > best_idf:
            best_idf = avg_idf
            best_passage = passage
    
    return best_passage


def calculate_idf(documents: List[str]) -> Dict[str, float]:
    """
    Calculate IDF for all terms in a collection.
    
    Args:
        documents: List of document texts
        
    Returns:
        Dictionary mapping terms to IDF values
    """
    N = len(documents)
    df = Counter()
    
    # Count document frequency
    for doc in documents:
        words = set(doc.lower().split())
        df.update(words)
    
    # Calculate IDF
    idf = {}
    for term, freq in df.items():
        idf[term] = math.log((N + 1) / (freq + 1)) + 1
    
    return idf


def calculate_query_alpha(query_text: str, idf_dict: Dict[str, float], 
                         all_doc_idfs: List[float]) -> float:
    """
    Calculate query-specific alpha based on IDF (from paper Section 3.3).
    
    α = percentage of documents with avg IDF < avg query IDF
    
    Args:
        query_text: Query text
        idf_dict: Dictionary of IDF values
        all_doc_idfs: List of average IDF values for all documents
        
    Returns:
        Alpha value between 0 and 1
    """
    query_words = query_text.lower().split()
    query_idfs = [idf_dict.get(word, 0) for word in query_words]
    
    if not query_idfs:
        return 0.5  # Default if no terms found
    
    avg_query_idf = np.mean(query_idfs)
    
    # Count documents with lower average IDF
    count_lower = sum(1 for doc_idf in all_doc_idfs if doc_idf < avg_query_idf)
    
    alpha = count_lower / len(all_doc_idfs) if all_doc_idfs else 0.5
    
    return alpha


def load_qrels(qrels_file: str) -> Dict[str, Dict[str, int]]:
    """
    Load TREC qrels file.
    
    Args:
        qrels_file: Path to qrels file
        
    Returns:
        Dictionary {query_id: {doc_id: relevance}}
    """
    qrels = {}
    
    with open(qrels_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                query_id = parts[0]
                doc_id = parts[2]
                relevance = int(parts[3])
                
                if query_id not in qrels:
                    qrels[query_id] = {}
                qrels[query_id][doc_id] = relevance
    
    return qrels


def write_trec_run(results: Dict[str, List[Tuple[str, float]]], 
                   output_file: str, run_name: str = "QAPR"):
    """
    Write results in TREC run format.
    
    Format: query_id Q0 doc_id rank score run_name
    
    Args:
        results: Dictionary {query_id: [(doc_id, score), ...]}
        output_file: Output file path
        run_name: Name of the run
    """
    with open(output_file, 'w') as f:
        for query_id, doc_scores in results.items():
            # Sort by score descending
            sorted_docs = sorted(doc_scores, key=lambda x: x[1], reverse=True)
            
            for rank, (doc_id, score) in enumerate(sorted_docs, 1):
                f.write(f"{query_id}\tQ0\t{doc_id}\t{rank}\t{score:.6f}\t{run_name}\n")


def save_pickle(obj, file_path: str):
    """Save object to pickle file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(file_path: str):
    """Load object from pickle file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Normalize scores to [0, 1] range."""
    if len(scores) == 0:
        return scores
    
    min_score = scores.min()
    max_score = scores.max()
    
    if max_score == min_score:
        return np.ones_like(scores)
    
    return (scores - min_score) / (max_score - min_score)


def print_results(metrics: Dict[str, float], model_name: str):
    """Print evaluation metrics in a formatted way."""
    print(f"\n{'=' * 60}")
    print(f"Results for {model_name}")
    print(f"{'=' * 60}")
    
    for metric, value in metrics.items():
        print(f"{metric:20s}: {value:.4f}")
    
    print(f"{'=' * 60}\n")
