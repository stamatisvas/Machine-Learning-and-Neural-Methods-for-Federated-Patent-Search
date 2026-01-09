# CLEF-IP Data Preparation Pipeline

This repository contains scripts to prepare the CLEF-IP 2011 dataset for results merging experiments.

## Overview

The data preparation pipeline consists of 7 steps:

1. **Merge Patent Kinds**: Combine different patent document kinds (A1, B2, etc.) into single documents
2. **Convert to SGML**: Convert merged XML to SGML format (TREC format)
3. **Extract IPC Classifications**: Extract IPC codes at level 3 (subclass level)
4. **Split by IPC Level 3**: Split patents into collections based on IPC subclass codes
5. **Create Indices**: Create Anserini indices for each collection
6. **Query-Based Sampling**: Create representation sets for each collection
7. **Create Queries**: Extract queries from CLEF-IP topic files

## Prerequisites

1. Download CLEF-IP 2011 dataset from http://www.ifs.tuwien.ac.at/~clef-ip/download/2011/index.shtml
2. Install Anserini (https://github.com/castorini/anserini)
3. Install Python dependencies: `pip install -r requirements.txt`

Required Python packages:
- beautifulsoup4
- googletrans
- pyserini
- numpy

## Setup

### 1. Configure Paths

Edit `config.py` and set the following paths:

```python
CLEF_IP_RAW_PATH = '/path/to/CLEF-IP-2011/'
TOPICS_PATH = '/path/to/PAC_topics/files/'
ANSERINI_HOME = '/path/to/anserini/'
```

Alternatively, set environment variables:

```bash
export CLEF_IP_RAW_PATH=/path/to/CLEF-IP-2011/
export TOPICS_PATH=/path/to/PAC_topics/files/
export ANSERINI_HOME=/path/to/anserini/
```

### 2. Run Pipeline

Execute scripts in order:

```bash
# Step 1: Merge patent kinds
python scripts/01_merge_patent_kinds.py

# Step 2: Convert to SGML
python scripts/02_convert_to_sgml.py

# Step 3: Extract IPC classifications
python scripts/03_extract_ipc_classifications.py

# Step 4: Split by IPC level 3
python scripts/04_split_by_ipc_level3.py

# Step 5: Create indices
python scripts/05_create_indices.py

# Step 6: Query-based sampling
python scripts/06_query_based_sampling.py

# Step 7: Create queries
python scripts/07_create_queries.py
```

## Output Structure

After running the pipeline, the following structure will be created:

```
CLEF-IP/
├── data/
│   ├── merged_xml/          # Merged XML patents
│   ├── sgml/                # SGML formatted patents
│   ├── classifications_split3.txt  # IPC classifications
│   ├── split3_data/         # Patents split by IPC codes
│   ├── split_3_index/       # Anserini indices per collection
│   ├── Query_based_sampling_indexes/  # Representation sets
│   ├── centralized_index/   # Centralized index
│   └── 300_topics_set3_EN.txt  # Generated queries
```

## Configuration

Key parameters in `config.py`:

- `NUM_QUERIES = 300`: Number of queries to generate
- `SAMPLING_ITERATIONS = 150`: Iterations for query-based sampling
- `SAMPLING_RESULTS_PER_QUERY = 4`: Results per query in sampling
- `TARGET_SAMPLES_PER_COLLECTION = 300`: Target samples per collection
- `DESCRIPTION_MAX_WORDS = 500`: Maximum words in description

## Notes

- The pipeline processes all patents from the CLEF-IP dataset
- Each step depends on the output of the previous step
- Ensure sufficient disk space for intermediate and final outputs
- Query-based sampling (Step 6) may take significant time for large collections
- Anserini indexing (Steps 5 and 6) requires Java and the Anserini toolkit

## Optional Scripts

Additional scripts for experimental variants:

### SGML Variants

**02a_convert_to_sgml_variants.py** - Creates multiple SGML variants:
- Full documents (everything)
- Titles and abstracts only
- Descriptions only
- Claims only
- Includes CPC classifications in addition to IPC/IPCR
- Uses aggressive text cleaning

**02b_convert_without_ipc.py** - Creates SGML files without IPC classifications

### Statistics

**statistics.py** - Computes statistics about the dataset:
- Total number of patents
- Number of patents with title, abstract, description, claims
- Percentage of patents with all fields

## Structure

```
CLEF-IP/
├── config.py                # Configuration parameters
├── utils/                   # Utility functions
│   ├── xml_utils.py        # XML processing utilities
│   └── text_utils.py       # Text processing utilities
├── scripts/                 # Pipeline scripts
│   ├── 01_merge_patent_kinds.py
│   ├── 02_convert_to_sgml.py
│   ├── 02a_convert_to_sgml_variants.py  # Optional: Multiple variants
│   ├── 02b_convert_without_ipc.py       # Optional: Without IPC
│   ├── 03_extract_ipc_classifications.py
│   ├── 04_split_by_ipc_level3.py
│   ├── 05_create_indices.py
│   ├── 06_query_based_sampling.py
│   ├── 07_create_queries.py
│   └── statistics.py                    # Optional: Dataset statistics
└── README.md               # This file
```
