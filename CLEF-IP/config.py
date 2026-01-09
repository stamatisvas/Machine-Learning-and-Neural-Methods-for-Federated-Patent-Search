"""
Configuration for CLEF-IP data preparation pipeline.

This module contains all configuration parameters needed to run the
data preparation scripts for the CLEF-IP dataset.
"""

import os
from pathlib import Path

# Base paths - Update these according to your setup
BASE_DIR = Path(__file__).parent

# Input paths
CLEF_IP_RAW_PATH = os.getenv('CLEF_IP_RAW_PATH', '/path/to/CLEF-IP-2011/')
TOPICS_PATH = os.getenv('TOPICS_PATH', '/path/to/PAC_topics/files/')

# Output paths
MERGED_XML_PATH = os.getenv('MERGED_XML_PATH', BASE_DIR / 'data/merged_xml/')
SGML_PATH = os.getenv('SGML_PATH', BASE_DIR / 'data/sgml/')
SPLIT3_DATA_PATH = os.getenv('SPLIT3_DATA_PATH', BASE_DIR / 'data/split3_data/')
SPLIT3_INDEX_PATH = os.getenv('SPLIT3_INDEX_PATH', BASE_DIR / 'data/split_3_index/')
REPRESENTATIONS_INDEX_PATH = os.getenv('REPRESENTATIONS_INDEX_PATH', BASE_DIR / 'data/Query_based_sampling_indexes/')
CENTRALIZED_INDEX_PATH = os.getenv('CENTRALIZED_INDEX_PATH', BASE_DIR / 'data/centralized_index/')
CLASSIFICATIONS_FILE = os.getenv('CLASSIFICATIONS_FILE', BASE_DIR / 'data/classifications_split3.txt')
QUERIES_FILE = os.getenv('QUERIES_FILE', BASE_DIR / 'data/300_topics_set3_EN.txt')

# Anserini paths
ANSERINI_HOME = os.getenv('ANSERINI_HOME', '/path/to/anserini/')
INDEX_COLLECTION = f'{ANSERINI_HOME}/target/appassembler/bin/IndexCollection'

# Processing parameters
NUM_QUERIES = 300  # Number of queries to generate
SAMPLING_ITERATIONS = 150  # Number of iterations for query-based sampling
SAMPLING_RESULTS_PER_QUERY = 4  # Number of results per query in sampling
TARGET_SAMPLES_PER_COLLECTION = 300  # Target number of samples per collection
DESCRIPTION_MAX_WORDS = 500  # Maximum words in description for queries

# Create output directories
for path in [MERGED_XML_PATH, SGML_PATH, SPLIT3_DATA_PATH, SPLIT3_INDEX_PATH, 
             REPRESENTATIONS_INDEX_PATH, CENTRALIZED_INDEX_PATH]:
    Path(path).mkdir(parents=True, exist_ok=True)
