# QAPR: Query-Adaptive Patent Re-ranking & Retrieval

> **Part of**: [Machine Learning and Neural Methods for Federated Patent Search](../README.md) - PhD Research by Vasileios Stamatis

Reproducible implementation of **two papers**:

1. **"A Novel Re-ranking Architecture for Patent Search"** (QAPR for re-ranking) By Vasileios Stamatis, Michail Salampasis, Konstantinos Diamantaras.

2. **"Beyond BM25: Strengthening First-Stage Patent Retrieval with Query-Aware Hybridization"** (QAPR for first-stage retrieval) By Vasileios Stamatis and Michail Salampasis



## Overview

QAPR combines lexical (BM25) and semantic (SBERT) signals with AI models using query-adaptive weights. It works for:
- **Re-ranking**: Refine top-k candidates (Paper 1, Steps 1-7)
- **First-stage retrieval**: Enhance initial retrieval from BM25, MiniLM, or ColBERT (Paper 2, Steps 8-11)

Key features:
- **Document segmentation** into Abstract, Description, Claims (500 words max each)
- **9-way relevance scoring** between query and document sections
- **AI-powered combination** using LambdaMART or MLP
- **Query-adaptive weights** (α) based on IDF

## Architecture

```
Query → Split (A,D,C) ──┐
                         ├─→ 9 Lexical Scores (BM25)  ─┐
Candidates → Split (A,D,C)─┘                           ├─→ AI Model → Combined Score ─┐
                         ├─→ 9 Semantic Scores (SBERT)─┘                              ├─→ Final Score
                         └────────────────────────────────→ Query Weights (α) ────────┘
```

**Final Score** = `CombinedScore + α * MaxLex + (1-α) * MaxSem`

Where α is calculated per-query based on IDF.

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- `pyserini` (BM25 indexing and retrieval)
- `sentence-transformers` (SBERT for semantic scoring)
- `lightgbm` (LambdaMART)
- `scikit-learn` (MLP, evaluation)
- `torch` (neural models)

## Dataset

**CLEF-IP 2011**: The scripts expect **SGML-formatted patents** (processed output).

- **Preprocessing required**: Run the [CLEF-IP pipeline](../CLEF-IP/README.md) first (Steps 1-2) to convert raw XML to SGML format
- **Input for QAPR**: Use the SGML files from `CLEF-IP/data/sgml/`

## Usage

### Paper 1: QAPR for Re-ranking (CLEF-IP)

**Prerequisites**: Complete [CLEF-IP pipeline](../CLEF-IP/README.md) Steps 1-2 first.

Run scripts 1-7 in order:

### Step 1: Index Documents

```bash
python 1_index_documents.py
```

Creates BM25 index for first-stage retrieval.

**Output**: `./index/lucene_index/`

---

### Step 2: First-Stage Retrieval

```bash
python 2_first_stage_retrieval.py
```

Retrieves top-1000 candidates per query using BM25.

**Output**: `./output/initial_ranking.tsv`

---

### Step 3: Split Documents into Sections

```bash
python 3_split_sections.py
```

Splits documents into Abstract, Description, Claims (max 500 words each).

**Output**: `./output/splits/` (document_splits.pkl, topic_splits.pkl, idf_dict.pkl)

---

### Step 4: Extract Features

```bash
python 4_extract_features.py
```

Extracts 19 features per query-document pair:
- 1 initial BM25 score
- 9 lexical scores (BM25 for each section pair)
- 9 semantic scores (SBERT for each section pair)

**Output**: `./output/features/` (features_train.csv, features_test.csv)

---

### Step 5: Train Models

```bash
python 5_train_models.py
```

Trains LambdaMART and MLP on training features.

**Output**: `./output/models/` (lambdamart.pkl, mlp.pkl, scaler.pkl)

---

### Step 6: Re-rank with Query Weights

```bash
python 6_rerank_with_weights.py
```

Applies query-specific weights: `Final Score = CombinedScore + α * MaxLex + (1-α) * MaxSem`

**Output**: `./output/results/` (lambdamart_ranking.txt, mlp_ranking.txt)

---

### Step 7: Evaluate

```bash
python 7_evaluate.py
```

Evaluates results using MAP, RECALL, P@10, P@20, NDCG.

**Output**: `./output/results/evaluation_summary.txt`

---

### Paper 2: QAPR for First-Stage Retrieval (WPI-PR)

**Prerequisites**: Complete [WPI-PR pipeline](../WPI-PR/README.md) (Steps 1-4) to process WPI dataset and generate SGML files and qrels.

After completing QAPR steps 1-7 with WPI-PR data, run additional scripts for first-stage retrieval experiments:

### Step 8: Index Dense Models

```bash
python 8_index_dense_models.py
```

Creates dense embeddings for MiniLM and ColBERT retrievers.

**Output**: `./output/dense_indexes/` (minilm_index.pkl, colbert_index.pkl)

---

### Step 9: Retrieve with All Methods

```bash
python 9_retrieve_all_methods.py
```

Performs first-stage retrieval with:
- **BM25** (sparse): Full documents
- **MiniLM** (dense): 512 tokens per section, averaged
- **ColBERT** (late-interaction): 256 tokens from abstract

**Output**: `./output/first_stage_rankings/` (bm25_ranking.tsv, minilm_ranking.tsv, colbert_ranking.tsv)

---

### Step 10: Apply QAPR to All Methods

```bash
python 10_apply_qapr_to_all.py
```

Applies QAPR re-ranking to all three first-stage retrieval results.

**Output**: `./output/qapr_results/` (qapr_BM25_ranking.txt, qapr_MiniLM_ranking.txt, qapr_ColBERT_ranking.txt)

---

### Step 11: Evaluate All Methods

```bash
python 11_evaluate_all.py
```

Evaluates baseline methods vs QAPR-enhanced versions.

Metrics: Recall@10/100/1000, NDCG@10/100/1000, MAP@10/100/1000

**Output**: `./output/evaluation/detailed_results.txt`

---

### Step 12: MSMARCO Adaptation (Optional)

```bash
python 12_msmarco_adaptation.py
```

Tests QAPR generalizability on MSMARCO dataset (from Paper 1).

**Key differences**:
- Queries NOT split (too short)
- Only 7 features (vs 19 for patents)
- No query-specific weights
- Equal weighting for final score

**Result**: Limited performance (domain-specific method)

**Output**: `./output/msmarco/msmarco_qapr_ranking.txt`

---

## Configuration

Edit `config.yaml` based on which experiment you're running:

### For Paper 1 (CLEF-IP Re-ranking)

```yaml
# Paths
index_dir: "./index"  # Output directory (created automatically, no need to change)
documents_dir: "/path/to/CLEF-IP/data/sgml/"  # UPDATE: From CLEF-IP pipeline Step 2
topics_file: "/path/to/CLEF-IP/data/300_topics_set3_EN.txt"  # UPDATE: From CLEF-IP pipeline Step 7
qrels_file: "/path/to/CLEF-IP/qrels.txt"  # UPDATE: CLEF-IP qrels file
```

### For Paper 2 (WPI-PR First-Stage Retrieval)

```yaml
# Paths
index_dir: "./index"  # Output directory (created automatically, no need to change)
documents_dir: "/path/to/wpi_sgml/data/"  # UPDATE: From WPI-PR pipeline Step 2
topics_dir: "/path/to/wpi_sgml/topics/"  # UPDATE: From WPI-PR pipeline Step 2
qrels_file: "/path/to/wpi_qrels.txt"  # UPDATE: From WPI-PR pipeline Step 4
```

### Common Parameters

```yaml
# Output directory (created automatically)
output_dir: "./output"

# Parameters
top_k: 1000              # Number of candidates for re-ranking
max_section_words: 500   # Maximum words per section
train_test_split: 0.8    # 80% train, 20% test

# BM25 parameters
bm25_k1: 0.9
bm25_b: 0.4

# Models
sbert_model: "google/bert-for-patents"
use_lambdamart: true
use_mlp: true

# LambdaMART parameters
lambdamart_trees: 100
lambdamart_leaves: 10
lambdamart_lr: 0.1

# MLP parameters
mlp_hidden_layers: [64, 32]
mlp_activation: "relu"
mlp_epochs: 50
```

## Reproducing Paper Results

### Paper 1: Re-ranking (CLEF-IP)

1. Prepare CLEF-IP data
2. Edit `config.yaml` with your paths
3. Run steps 1-7 in sequence

**Expected Results**:

| Model | MAP | RECALL | PRES |
|-------|-----|--------|------|
| BM25 (baseline) | baseline | baseline | baseline |
| ProposedLambdaMART | +30% | +28% | +27% |
| ProposedMLP | +28% | +26% | +25% |

(Statistical significance: ‡ p<0.01, † p<0.05, ⁕ p<0.10)

### Paper 2: First-Stage Retrieval (WPI-PR)

1. Prepare WPI-PR data (use [WPI-PR pipeline](../WPI-PR/README.md))
2. Complete steps 1-7 first
3. Run steps 8-11 for first-stage experiments

**Expected Results** (Proportional Improvements with QAPR):

| Baseline | Recall@1000 | NDCG@10 | MAP@10 |
|----------|-------------|---------|--------|
| BM25 → QAPR(BM25) | +7-10% | +8-10% | +10-12% |
| MiniLM → QAPR(MiniLM) | +8-12% | +8-10% | +10-11% |
| ColBERT → QAPR(ColBERT) | +10-14% | +15-17% | +18-21% |

**Key Findings**:
- MiniLM achieves highest absolute performance
- ColBERT shows largest relative improvements (token constraints)
- QAPR consistently improves all retrievers

### MSMARCO Generalization Test (Optional)

Run step 12 to test on MSMARCO:

```bash
python 12_msmarco_adaptation.py
```

**Expected Result**: Limited performance on MSMARCO, demonstrating domain specificity.

The paper notes: *"while our proposed methods were able to surpass the results of BERT-base re-ranking, they fell short of outperforming the other two baseline methods, namely BM25 and CombBert&BM25... suggesting that our method exhibits a certain degree of domain specificity."*

## File Structure

```
qapr_first_new/
├── README.md                    # This file
├── config.yaml                  # Configuration
├── requirements.txt             # Dependencies
├── utils.py                     # Helper functions
│
├── # Paper 1: Re-ranking (Steps 1-7)
├── 1_index_documents.py         # BM25 indexing
├── 2_first_stage_retrieval.py   # Initial ranking
├── 3_split_sections.py          # Document segmentation
├── 4_extract_features.py        # Feature extraction (19 features)
├── 5_train_models.py            # Train LambdaMART/MLP
├── 6_rerank_with_weights.py     # Re-ranking with query weights
├── 7_evaluate.py                # Evaluation (MAP, RECALL, PRES)
│
├── # Paper 2: First-Stage Retrieval (Steps 8-11)
├── 8_index_dense_models.py      # MiniLM + ColBERT indexing
├── 9_retrieve_all_methods.py    # Retrieve with BM25/MiniLM/ColBERT
├── 10_apply_qapr_to_all.py      # Apply QAPR to all methods
├── 11_evaluate_all.py           # Evaluate all (Recall/NDCG/MAP @10/100/1000)
│
└── # Optional: Generalization Test
    └── 12_msmarco_adaptation.py     # Test on MSMARCO (limited performance)
```

## Key Implementation Details

### Document Splitting (Section 3.1)

- Each document split into Abstract, Description, Claims
- Maximum 500 words per section
- If section > 500 words: split into passages, select highest average IDF passage
- Maximum 1500 words total per document (500 × 3)

### Feature Extraction (Section 3.2)

**9 Section Pairs**:
```
QAbstr × CAbstr, QAbstr × CDesc, QAbstr × CClm
QDesc  × CAbstr, QDesc  × CDesc, QDesc  × CClm
QClm   × CAbstr, QClm   × CDesc, QClm   × CClm
```

**For each pair**: Calculate BM25 score and SBERT cosine similarity

### Query Weights (Section 3.3)

α = percentage of documents with avg IDF < avg query IDF

- High α → query has rare terms → prioritize lexical matching
- Low α → query has common terms → prioritize semantic matching

### AI Models

**LambdaMART**: Gradient boosting with multiple decision trees (state-of-the-art LTR)

**MLP**: 2 hidden layers, ReLU activation

Both trained on 19 features to predict relevance.

## Baselines

### Paper 1 Baselines (Re-ranking)
- **BM25**: Anserini with k1=0.9, b=0.4
- **BERTmax**: Max SBERT score across sections
- **BERTsum**: Sum of SBERT scores across sections
- **CombBERT&BM25**: `BM25 + 0.25 * BM25 * BERT`

### Paper 2 Baselines (First-Stage)
- **BM25**: Sparse retrieval (Anserini)
- **MiniLM**: Dense retrieval (all-MiniLM-L6-v2)
  - 512 tokens per section, embeddings averaged
- **ColBERT**: Late-interaction retrieval
  - 256 tokens from abstract only (computational constraints)

## Notes

- Processing large patent collections is time-intensive
- SBERT embedding generation can be GPU-accelerated
- Consider using smaller subsets for testing/debugging
- The method is domain-specific (works best on patents)

## Troubleshooting

**Issue**: Out of memory during SBERT encoding  
**Solution**: Process in smaller batches, reduce `max_section_words`

**Issue**: Slow BM25 indexing  
**Solution**: Use pyserini with proper JVM memory settings

**Issue**: Poor results  
**Solution**: Verify train/test split matches paper (80/20), check feature normalization

## License

Research code for PhD thesis reproduction.
