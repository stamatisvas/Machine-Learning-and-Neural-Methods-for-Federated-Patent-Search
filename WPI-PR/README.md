# WPI-PR: Patent Collection Processing

> **Part of**: [Machine Learning and Neural Methods for Federated Patent Search](../README.md) - PhD Research by Vasileios Stamatis

Simple, reproducible scripts for processing the WPI patent collection for PhD research.

## Overview

This collection processes WPI XML patents into SGML format and generates ground truth data from patent citations for retrieval experiments.

## Requirements

```bash
pip install beautifulsoup4 lxml pyyaml
```

## Directory Structure

```
WPI-PR/
├── README.md                    # This file
├── config.yaml                  # Configuration file
├── 1_filter_valid_patents.py    # Step 1: Filter valid English patents
├── 2_convert_to_sgml.py         # Step 2: Convert XML to SGML format
├── 3_generate_ground_truth.py   # Step 3: Extract citations for ground truth
├── 4_convert_qrels_format.py    # Step 4: Convert to TREC qrels format
└── 5_split_by_sections.py       # Step 5 (optional): Split into abstracts/descriptions/claims
```

## Configuration

Edit `config.yaml` and update paths to match your system:

```yaml
input_dir: "/path/to/your/WPI-XML-files"
output_dir: "/path/to/output"
```

## Usage

### Step 1: Filter Valid Patents

Keeps only patents with all English sections (title, abstract, description, claims):

```bash
python 1_filter_valid_patents.py
```

**Input**: Raw WPI XML files  
**Output**: `{output_dir}/wpi_valid/` - Filtered XML patents

---

### Step 2: Convert to SGML Format

Converts XML to SGML/TREC format with text cleaning:

```bash
python 2_convert_to_sgml.py
```

**Input**: `{output_dir}/wpi_valid/`  
**Output**: 
- `{output_dir}/wpi_sgml/data/` - SGML documents
- `{output_dir}/wpi_sgml/topics/` - Topic (query) documents

---

### Step 3: Generate Ground Truth

Extracts patent citations to create ground truth:

```bash
python 3_generate_ground_truth.py
```

**Input**: `{output_dir}/wpi_valid/`  
**Output**: `wpi_ground_truths.txt` - Citation relationships

Format: `query_patent<sep>cited_patent1 cited_patent2 ...`

---

### Step 4: Convert to Qrels Format

Converts ground truth to TREC qrels format:

```bash
python 4_convert_qrels_format.py
```

**Input**: `wpi_ground_truths.txt`  
**Output**: `wpi_qrels.txt` - TREC qrels format

Format: `query_id 0 doc_id 1`

---

### Step 5: Split by Sections (Optional)

Creates separate collections for abstracts, descriptions, and claims:

```bash
python 5_split_by_sections.py
```

**Input**: `{output_dir}/wpi_sgml/data/`  
**Output**: 
- `{output_dir}/wpi_sgml/abstracts/`
- `{output_dir}/wpi_sgml/descriptions/`
- `{output_dir}/wpi_sgml/claims/`

---

## Output Formats

### SGML Document Format

```
<DOC>
<DOCNO>
US-12345678-A1
</DOCNO>
<TEXT>
<DATE>2023-01-01</DATE>
<IPCR-CLASSIFICATIONS>
H04N 21/00
</IPCR-CLASSIFICATIONS>
<TITLE>
invention title
</TITLE>
<ABSTRACT>
abstract text
</ABSTRACT>
<DESCRIPTION>
description text
</DESCRIPTION>
<CLAIMS>
claims text
</CLAIMS>
</TEXT>
</DOC>
```

### Ground Truth Format

**wpi_ground_truths.txt**:
```
US-12345678-A1<sep>US-87654321-B2 US-11223344-A1
```

**wpi_qrels.txt** (TREC format):
```
US-12345678-A1    0    US-87654321-B2    1
US-12345678-A1    0    US-11223344-A1    1
```

## Processing Details

### Text Cleaning

The scripts apply the following text normalization:
- Convert to lowercase
- Remove special characters: `( ) { } [ ] , . ! ? @ # $ % & * = + - " ' : ; < >`
- Remove digits
- Normalize whitespace

### Ground Truth Logic

- Only patents with ≥2 citations are included as queries
- All cited patents must exist in the collection
- Citations represent relevance judgments (cited = relevant)

## Typical Workflow

```bash
# 1. Edit config.yaml with your paths
vim config.yaml

# 2. Run all steps in sequence
python 1_filter_valid_patents.py
python 2_convert_to_sgml.py
python 3_generate_ground_truth.py
python 4_convert_qrels_format.py

# 3. Optional: Split by sections
python 5_split_by_sections.py
```

## Expected Processing Time

For ~500,000 patents:
- Step 1: 2-3 hours
- Step 2: 3-4 hours
- Step 3: 30 minutes
- Step 4: <1 minute
- Step 5: 1-2 hours

## Disk Space Requirements

- Input (XML): ~200 GB
- Output (SGML): ~100 GB
- Temporary: ~50 GB
- **Total**: ~350 GB

## Citation

If you use this in your research, please cite:

```
Vasilis Stamatis, PhD Thesis, International Hellenic University, 2026
```

## Notes

- All paths are configured in `config.yaml`
- Scripts process files in-place and log progress to console
- Processing can be interrupted and resumed (idempotent operations)
- Check logs for errors if processing fails
