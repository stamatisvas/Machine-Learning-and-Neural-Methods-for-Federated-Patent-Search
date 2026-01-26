# Experimental Results

This directory contains experimental runs from the PhD research "Machine Learning and Neural Methods for Federated Patent Search".

## Directory Structure

```
experimental_results/
├── qapr/                    # All QAPR experiments
│   ├── msmarco_adaptation/  # MS MARCO generalization tests
│   └── *.txt                # Main QAPR runs
└── results_merging/         # Results Merging experiments
    ├── *.res                # Results Merging runs
    └── *.csv                # Detailed analysis files
```

---

## QAPR Results

**Location**: `qapr/`

Contains experimental runs for query-adaptive patent re-ranking and first-stage retrieval enhancement.

### Baseline Results

| File | Description | Method |
|------|-------------|--------|
| `BM25_baseline.txt` | BM25 baseline | Anserini BM25 (k1=0.9, b=0.4) |
| `BERT_baseline.txt` | BERT baseline | BERT sum aggregation |
| CombBERT&BM25 | Combination baseline | BM25 + 0.25×BM25×BERT (see MS MARCO) |

### Re-ranking Results

| File | Description | Improvements |
|------|-------------|--------------|
| `qapr_LambdaMART_reranking.txt` | LambdaMART with query-adaptive weights | +30% MAP, +28% RECALL |
| `qapr_DNN_reranking.txt` | DNN with query-adaptive weights | +28% MAP, +26% RECALL |

**Key Features**:
- Document segmentation (Abstract, Description, Claims)
- 9-way relevance scoring (3 query sections × 3 document sections)
- Query-adaptive weighting based on IDF statistics
- 19 features per query-document pair

### First-Stage Retrieval Results

| File | Description | Baseline → QAPR Improvement |
|------|-------------|------------------------------|
| `qapr_firstStage_BM25.txt` | BM25 + QAPR | +7-10% Recall@1000 |
| `qapr_firstStage_MiniLM.txt` | MiniLM + QAPR | +8-12% Recall@1000 |
| `qapr_firstStage_ColBERT.txt` | ColBERT + QAPR | +10-14% Recall@1000, +15-17% NDCG@10 |
| `qapr_firstStage_alternative.txt` | Alternative configuration | Experimental variant |

**Key Findings**:
- QAPR consistently improves all baseline retrievers
- ColBERT shows largest relative improvements
- MiniLM achieves highest absolute performance

### MS MARCO Adaptation

**Location**: `qapr/msmarco_adaptation/`

Generalization test on MS MARCO passage ranking dataset.

| File | Description |
|------|-------------|
| `msmarco_bm25.res` | BM25 baseline on MS MARCO |
| `msmarco_bert.res` | BERT baseline |
| `msmarco_bert_max.res` | BERT max aggregation |
| `msmarco_bert_sum.res` | BERT sum aggregation |
| `msmarco_bert_avg.res` | BERT average aggregation |
| `msmarco_bert_max_norm.res` | BERT max (normalized) |
| `msmarco_bert_sum_norm.res` | BERT sum (normalized) |
| `msmarco_comb_bert_bm25.res` | **CombBERT&BM25 baseline**: BM25 + 0.25×BM25×BERT |
| `msmarco_lambdamart.res` | LambdaMART adaptation |
| `msmarco_pytorch.txt` | PyTorch implementation |

**Key Differences from Patent Experiments**:
- Queries NOT split (too short for segmentation)
- Only 7 features (vs 19 for patents)
- No query-specific weights
- Limited performance (demonstrates domain specificity)

**Result**: QAPR shows domain specificity - optimized for patent search, not general passage retrieval.

---

## Results Merging

**Location**: `results_merging/`

Contains experimental runs for machine learning methods for results merging in federated patent retrieval.

### Baseline Methods

| File | Method | Description |
|------|--------|-------------|
| `CORI.res` | CORI | Collection selection + merging baseline |
| `SAFE*.res` | SAFE | Sample-Aggregate-FEw baseline (3 variants) |
| `SSL` | SSL | Semi-Supervised Learning baseline (analysis files only) |
| `random_merging*.res` | Random | Random merging baseline (3 variants) |
| `upper_baseline_SSL.res` | Upper Baseline | Oracle upper bound for all methods |

### Multiple Models (MMs) - One Model Per Resource

| File | Model | Performance |
|------|-------|-------------|
| `MMs_random_forest.res` | Random Forest | **Best Overall** |
| `MMs_decision_tree.res` | Decision Tree | Good |
| `MMs_svr.res` | SVR | Moderate |
| `MMs_polynomial_x2.res` | Polynomial (x²) | Polynomial regression |
| `MMs_polynomial_x3.res` | Polynomial (x³) | Higher-order polynomial |
| `MMs_random_forest_*.res` | Random Forest variants | Alternative configurations |
| `upper_MMs_*.res` | MMs Upper Bounds | Oracle bounds for MMs |

### Global Models (GMs) - One Model For All Resources

| File | Model | Performance |
|------|-------|-------------|
| `GMs_dnn.res` | DNN | Best MAP |
| `GMs_random_forest.res` | Random Forest | Best PRES/RECALL |
| `GMs_decision_tree.res` | Decision Tree | Good |
| `GMs_linear_regression.res` | Linear Regression | Baseline |
| `GMs_svr.res` | SVR | Moderate |
| `upper_GMs_*.res` | GMs Upper Bounds | Oracle bounds for GMs |

### Detailed Analysis Files (CSV)

All `detailed_*.csv` and `outfile_*.csv` files contain per-query analysis including:
- Per-collection scores
- Feature importance
- Model predictions
- Comparison with centralized baseline

**CSV Files**:
- `detailed_MMs_*.csv`: Detailed per-query results for Multiple Models
- `detailed_GMs_*.csv`: Detailed per-query results for Global Models
- `detailed_cori.csv`: Detailed CORI baseline results
- `detailed_SAFE.csv`: Detailed SAFE baseline results
- `detailed_baseline_ssl.csv`: Detailed SSL baseline analysis
- `outfile_MMs_*.csv`: Model training outputs for Multiple Models
- `outfile_GMs_*.csv`: Model training outputs for Global Models
- `outfile_CORI.csv`: CORI baseline outputs
- `outfile_baseline_ssl.csv`: SSL baseline outputs

### Environments Tested

All methods evaluated in three environments:
1. **Cooperative**: Documents returned with scores
2. **Uncooperative**: Ranked lists only (no scores)
3. **Uncooperative Weighted**: Ranked lists with CORI weights

---

## File Formats

### TREC Format (.res, .txt)
```
query_id Q0 doc_id rank score run_name
```

Example:
```
EP-1310580-A2 Q0 EP-0123456-A1 1 15.234 MMs:random_forest
```

### CSV Format
Detailed analysis files with columns:
- query_id
- doc_id
- features (various)
- predictions
- relevance labels
- evaluation metrics

---

## Evaluation Metrics

### QAPR
- **MAP** (Mean Average Precision)
- **RECALL** (Recall at various cutoffs)
- **PRES** (Precision)
- **NDCG** (Normalized Discounted Cumulative Gain)
- **P@k** (Precision at k: k=10, 20)

### Results Merging
- **MAP** (Mean Average Precision)
- **RECALL** (Total recall)
- **PRES** (Precision at specific cutoff)

---

## Reproducing These Results

### For QAPR Results
1. Follow [QAPR/README.md](../QAPR/README.md) instructions
2. Use CLEF-IP 2011 or WPI-PR datasets
3. Run complete pipeline (Steps 1-11)
4. Results will be generated in `QAPR/output/`

### For Results Merging
1. Follow [Results_Merging/README.md](../Results_Merging/README.md) instructions
2. Prepare CLEF-IP data using [CLEF-IP pipeline](../CLEF-IP/README.md)
3. Run experiments with different methods and environments
4. Results will be generated in `Results_Merging/results/`

---

## Statistical Significance

Results include statistical significance testing:
- ‡ p<0.01 (highly significant)
- † p<0.05 (significant)
- ⁕ p<0.10 (marginally significant)

All improvements over baselines were tested using paired t-tests or Wilcoxon signed-rank tests.

---

## Summary Statistics

### QAPR Experiments
- **Total runs**: ~20+ experimental configurations
- **Datasets**: CLEF-IP 2011, WPI-PR, MS MARCO
- **Queries**: 300 (CLEF-IP), variable (WPI-PR), 6,980 (MS MARCO)
- **Baselines**: BM25, BERT variants
- **Proposed methods**: LambdaMART, DNN with query-adaptive weights

### Results Merging Experiments
- **Total runs**: 60+ experimental configurations
- **Dataset**: CLEF-IP 2011 (split by IPC codes)
- **Collections**: 20+ distributed collections
- **Queries**: 300
- **Methods**: 5 baselines + 11 ML approaches
- **Environments**: 3 (Cooperative, Uncooperative, Uncooperative Weighted)

---

## Notes

- All results are from PhD research experiments
- Some files may have variant names due to experimental iterations
- `upper_*.res` files represent theoretical upper bounds (oracle runs)
- `test_*.res` files may be preliminary or diagnostic runs
- CSV files provide detailed breakdowns for in-depth analysis

---

**Last Updated**: January 2026  
**Total Experimental Runs**: 80+  
**Total Evaluations**: Thousands of query-document pairs
