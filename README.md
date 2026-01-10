# Machine Learning and Neural Methods for Federated Patent Search

This repository contains the complete implementation code for the PhD thesis "Machine Learning and Neural Methods for Federated Patent Search" by Vasileios Stamatis, under the supervision of Prof. Michail Salampasis and Prof. Konstantinos Diamantaras at the International Hellenic University.

## Overview

This research explores machine learning and neural methods to improve patent search in federated environments. The work addresses three main challenges:

1. **Query-Adaptive Patent Re-ranking**: Enhancing patent retrieval through query-aware combination of lexical and semantic signals
2. **First-Stage Retrieval Enhancement**: Improving baseline retrieval methods (BM25, MiniLM, ColBERT) with query-adaptive hybridization
3. **Results Merging in Federated Search**: Developing machine learning methods to merge results from distributed patent collections

## Repository Structure

This repository contains four main sub-projects:

```
Machine-Learning-and-Neural-Methods-for-Federated-Patent-Search/
├── CLEF-IP/              # Patent dataset preparation pipeline
├── QAPR/                 # Query-Adaptive Patent Re-ranking & Retrieval
├── WPI-PR/              # WPI patent collection processing
├── Results_Merging/     # ML-based results merging methods
└── README.md            # This file
```

### 1. CLEF-IP Data Preparation Pipeline

**Location**: `CLEF-IP/`

**Purpose**: Preprocesses the CLEF-IP 2011 dataset for use in federated search experiments.

**Key Features**:
- Merges different patent document kinds (A1, B2, etc.)
- Converts XML to SGML/TREC format
- Splits patents by IPC (International Patent Classification) codes
- Creates distributed collections and centralized indices
- Generates query-based representation sets

**Used By**: Results_Merging, QAPR

**Dataset**: CLEF-IP 2011 (3.2M patent documents from EPO)

**Instructions**: See [CLEF-IP/README.md](CLEF-IP/README.md)

---

### 2. WPI-PR: Patent Collection Processing

**Location**: `WPI-PR/`

**Purpose**: Processes WPI (World Patents Index) XML patents into SGML format and generates ground truth from patent citations.

**Key Features**:
- Filters valid English patents
- Converts XML to SGML/TREC format
- Extracts patent citations for ground truth
- Splits documents into sections (abstract, description, claims)

**Used By**: QAPR (for first-stage retrieval experiments)

**Dataset**: WPI patent collection (XML format)

**Instructions**: See [WPI-PR/README.md](WPI-PR/README.md)

---

### 3. QAPR: Query-Adaptive Patent Re-ranking & Retrieval

**Location**: `QAPR/`

**Purpose**: Implementation of two research papers on query-adaptive patent retrieval.

**Papers Implemented**:
1. **"A Novel Re-ranking Architecture for Patent Search"** - QAPR for re-ranking top-k candidates
2. **"Beyond BM25: Strengthening First-Stage Patent Retrieval with Query-Aware Hybridization"** - QAPR for enhancing first-stage retrieval

**Key Innovations**:
- Document segmentation into Abstract, Description, Claims (500 words max each)
- 9-way relevance scoring between query and document sections
- Query-adaptive weighting (α) based on IDF statistics
- AI-powered combination using LambdaMART and MLP models

**Datasets**:
- **CLEF-IP 2011**: For re-ranking experiments (Paper 1)
- **WPI-PR**: For first-stage retrieval experiments (Paper 2)
- **MS MARCO**: For generalization testing (limited performance, showing domain specificity)

**Instructions**: See [QAPR/README.md](QAPR/README.md)

---

### 4. Results_Merging: ML Methods for Federated Patent Search

**Location**: `Results_Merging/`

**Purpose**: Machine learning methods for merging results from distributed patent collections.

**Key Features**:
- **Multiple Models (MMs)**: One ML model per resource (best: Random Forest)
- **Global Models (GMs)**: One ML model for all resources (best: DNN for MAP, Random Forest for PRES/RECALL)
- **Baseline Methods**: CORI, SSL, SAFE, Centralized, Random
- **Three Environments**: Cooperative, Uncooperative, Uncooperative Weighted

**Models Implemented**:
- Random Forest (best performer)
- Decision Trees
- Support Vector Regression (SVR)
- Linear Regression
- Polynomial Regression (x², x³)
- Deep Neural Networks (DNN)

**Dataset**: CLEF-IP 2011 (split by IPC codes into distributed collections)

**Instructions**: See [Results_Merging/README.md](Results_Merging/README.md)

---

## Datasets

This research uses three main datasets:

### 1. CLEF-IP 2011

**Description**: Large-scale patent collection from the European Patent Office (EPO)

**Size**: 2.5+ million patent documents (1M+ individual patents)

**Languages**: Mixed (English, German, French)

**Download**: [https://www.ifs.tuwien.ac.at/~clef-ip/download-central.shtml](https://www.ifs.tuwien.ac.at/~clef-ip/download-central.shtml)

**License**: Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License

**Used In**: CLEF-IP pipeline, Results_Merging, QAPR (re-ranking experiments)

### 2. WPI Patent Collection

**Description**: World Patents Index XML patent collection

**Size**: ~500,000 patents

**Format**: XML with structured fields (title, abstract, description, claims, citations)

**Used In**: WPI-PR processing, QAPR (first-stage retrieval experiments)

### 3. MS MARCO

**Description**: Microsoft MAchine Reading COmprehension dataset for passage/document ranking

**Download**: [https://microsoft.github.io/msmarco/Datasets.html](https://microsoft.github.io/msmarco/Datasets.html)

**Used In**: QAPR (generalization testing - Step 12)

**Note**: QAPR shows limited performance on MS MARCO, demonstrating domain specificity of the method

---

## Getting Started

### Prerequisites

- Python 3.8+
- Java 11+ (for Anserini/Pyserini)
- GPU (optional but recommended for SBERT embedding generation)
- 50-200GB disk space (depending on datasets)

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/Machine-Learning-and-Neural-Methods-for-Federated-Patent-Search.git
cd Machine-Learning-and-Neural-Methods-for-Federated-Patent-Search
```

2. **Download datasets** (see Datasets section below)

3. **Configure each sub-project** - Edit config files with your paths:
```bash
# Edit these files with your dataset paths
vim CLEF-IP/config.py
vim WPI-PR/config.yaml
vim QAPR/config.yaml
vim Results_Merging/config.py
```

4. **Install dependencies**:
```bash
# Install for each sub-project
cd CLEF-IP && pip install -r requirements.txt && cd ..
cd WPI-PR && pip install -r requirements.txt && cd ..
cd QAPR && pip install -r requirements.txt && cd ..
cd Results_Merging && pip install -r requirements.txt && cd ..
```

5. **Install Anserini** (required for indexing and retrieval):
```bash
git clone https://github.com/castorini/anserini.git
cd anserini && mvn clean package appassembler:assemble && cd ..
```

### Workflow

The typical workflow for reproducing the PhD research:

#### 1. Prepare CLEF-IP Dataset (for Results_Merging and QAPR re-ranking)
```bash
cd CLEF-IP
# Configure paths in config.py
python scripts/01_merge_patent_kinds.py
python scripts/02_convert_to_sgml.py
python scripts/03_extract_ipc_classifications.py
python scripts/04_split_by_ipc_level3.py
python scripts/05_create_indices.py
python scripts/06_query_based_sampling.py
python scripts/07_create_queries.py
```

#### 2. Results Merging Experiments
```bash
cd Results_Merging
# Configure paths in config.py
python scripts/run_experiment.py --environment cooperative --method MMs:random_forest --output results/MMs_rf_coop.res
```

#### 3. Prepare WPI-PR Dataset (for QAPR first-stage retrieval)
```bash
cd WPI-PR
# Configure paths in config.yaml
python 1_filter_valid_patents.py
python 2_convert_to_sgml.py
python 3_generate_ground_truth.py
python 4_convert_qrels_format.py
```

#### 4. QAPR Re-ranking Experiments (Paper 1)
```bash
cd QAPR
# Configure paths in config.yaml
python 1_index_documents.py
python 2_first_stage_retrieval.py
python 3_split_sections.py
python 4_extract_features.py
python 5_train_models.py
python 6_rerank_with_weights.py
python 7_evaluate.py
```

#### 5. QAPR First-Stage Retrieval Experiments (Paper 2)
```bash
cd QAPR
python 8_index_dense_models.py
python 9_retrieve_all_methods.py
python 10_apply_qapr_to_all.py
python 11_evaluate_all.py
```

---

## Key Contributions

### 1. Query-Adaptive Re-ranking (QAPR)
- Novel architecture combining lexical (BM25) and semantic (SBERT) signals
- Query-adaptive weighting based on IDF statistics
- 9-way section-level scoring (Abstract, Description, Claims)
- Significant improvements: +30% MAP, +28% RECALL, +27% PRES

### 2. First-Stage Retrieval Enhancement
- QAPR applied to enhance BM25, MiniLM, and ColBERT
- Consistent improvements across all baseline methods
- Largest gains with ColBERT: +14% Recall@1000, +17% NDCG@10, +21% MAP@10

### 3. Machine Learning for Results Merging
- Multiple Models (MMs) approach: One model per resource
- Global Models (GMs) approach: One model for all resources
- Random Forest achieves best overall performance
- DNN achieves best MAP performance in Global Models

---

## Publications

```bibtex
@phdthesis{stamatis2026federated,
  title={Machine Learning and Neural Methods for Federated Patent Search},
  author={Stamatis, Vasileios},
  year={2026},
  school={International Hellenic University},
  supervisor={Salampasis, Michail and Diamantaras, Konstantinos}
}

@article{stamatis2024patent,
  title={A Novel Re-ranking Architecture for Patent Search},
  author={Stamatis, Vasileios and Salampasis, Michail and Diamantaras, Konstantinos},
  journal={...},
  year={2024}
}

@article{stamatis2024beyond,
  title={Beyond BM25: Strengthening First-Stage Patent Retrieval with Query-Aware Hybridization},
  author={Stamatis, Vasileios and Salampasis, Michail and Diamantaras, Konstantinos},
  journal={...},
  year={2024}
}

@article{stamatis2024mlrm,
  title={Machine Learning Methods for Results Merging (MLRM) in Patent Retrieval},
  author={Stamatis, Vasileios and Salampasis, Michail and Diamantaras, Konstantinos},
  journal={...},
  year={2024}
}
```

---

## Citation

If you use this code or data in your research, please cite:

```bibtex
@misc{stamatis2026phd_repo,
  author = {Stamatis, Vasileios},
  title = {Machine Learning and Neural Methods for Federated Patent Search - PhD Repository},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/Machine-Learning-and-Neural-Methods-for-Federated-Patent-Search}}
}
```

---

## License

This research code is provided for academic and research purposes. Each dataset used has its own license:

- **CLEF-IP**: Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License
- **MS MARCO**: Non-commercial research purposes only (see MS MARCO Terms and Conditions)
- **WPI**: Subject to WPI licensing terms

---

## Contact

**Vasileios Stamatis**  
PhD Candidate  
International Hellenic University  
Email: [your.email@ihu.gr]

**Supervisors**:
- Prof. Michail Salampasis
- Prof. Konstantinos Diamantaras

---

## Acknowledgments

- CLEF-IP organizers and IRF for the MAREC patent collection
- Microsoft for the MS MARCO dataset
- Anserini team for the retrieval toolkit
- Sentence-Transformers team for SBERT models
- International Hellenic University for supporting this research

---

## Project Structure Summary

```
Machine-Learning-and-Neural-Methods-for-Federated-Patent-Search/
│
├── CLEF-IP/                    # CLEF-IP data preparation
│   ├── scripts/               # Pipeline scripts (01-07)
│   ├── utils/                 # XML/text processing utilities
│   ├── config.py              # Configuration
│   └── README.md              # Detailed instructions
│
├── WPI-PR/                     # WPI patent processing
│   ├── 1_filter_valid_patents.py
│   ├── 2_convert_to_sgml.py
│   ├── 3_generate_ground_truth.py
│   ├── 4_convert_qrels_format.py
│   ├── 5_split_by_sections.py
│   ├── config.yaml            # Configuration
│   └── README.md              # Detailed instructions
│
├── QAPR/                       # Query-Adaptive Patent Re-ranking
│   ├── 1_index_documents.py   # Paper 1: Re-ranking
│   ├── 2_first_stage_retrieval.py
│   ├── 3_split_sections.py
│   ├── 4_extract_features.py
│   ├── 5_train_models.py
│   ├── 6_rerank_with_weights.py
│   ├── 7_evaluate.py
│   ├── 8_index_dense_models.py # Paper 2: First-stage retrieval
│   ├── 9_retrieve_all_methods.py
│   ├── 10_apply_qapr_to_all.py
│   ├── 11_evaluate_all.py
│   ├── 12_msmarco_adaptation.py # Generalization test
│   ├── config.yaml            # Configuration
│   ├── utils.py               # Helper functions
│   └── README.md              # Detailed instructions
│
├── Results_Merging/           # ML-based results merging
│   ├── baselines/            # Baseline methods (CORI, SSL, SAFE)
│   ├── core/                 # Core utilities
│   ├── environments/         # Cooperative/Uncooperative environments
│   ├── models/               # ML models (MMs, GMs)
│   ├── scripts/              # Experiment runners
│   ├── config.py             # Configuration
│   └── README.md             # Detailed instructions
│
├── phd.docx                   # PhD thesis document
└── README.md                  # This file
```

---

## Troubleshooting

### Common Issues

**Issue**: Out of memory during SBERT encoding  
**Solution**: Process in smaller batches, reduce `max_section_words`, or use GPU

**Issue**: Anserini indexing fails  
**Solution**: Ensure Java 11+ is installed, set proper JVM memory settings

**Issue**: CLEF-IP download issues  
**Solution**: Check TU Wien repository availability, ensure proper licensing agreement

**Issue**: Poor reproduction of results  
**Solution**: Verify dataset versions, check train/test splits, ensure proper configuration

### Getting Help

For issues or questions:
1. Check the individual README files in each sub-project
2. Review the PhD thesis document for theoretical details
3. Open an issue on GitHub
4. Contact: [your.email@ihu.gr]

---

## Future Work

Potential extensions of this research:
- Application to other technical domains (biomedical, legal)
- Integration with large language models (LLMs)
- Cross-lingual patent retrieval
- Real-time federated patent search systems
- Privacy-preserving federated learning for patent search

---

## Version History

- **v1.0.0** (2026): Initial release with complete PhD implementation
- All sub-projects are independently versioned

---

**Last Updated**: January 2026
