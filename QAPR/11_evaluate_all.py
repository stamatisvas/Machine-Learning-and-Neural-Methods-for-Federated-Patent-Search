#!/usr/bin/env python3
"""
Step 11: Evaluate All Methods (Baseline + QAPR)
================================================
Evaluates all retrieval methods including QAPR enhancements.
Metrics: Recall@10/100/1000, NDCG@10/100/1000, MAP@10/100/1000
From paper: "Beyond BM25: Strengthening First-Stage Patent Retrieval"
"""

import subprocess
from pathlib import Path
from utils import load_config
import pytrec_eval

config = load_config()

FIRST_STAGE_DIR = Path(config['output_dir']) / "first_stage_rankings"
QAPR_DIR = Path(config['output_dir']) / "qapr_results"
QRELS_FILE = config['qrels_file']
OUTPUT_DIR = Path(config['output_dir']) / "evaluation"

print("=" * 80)
print("Step 11: Evaluate All Methods")
print("=" * 80)
print(f"Qrels: {QRELS_FILE}")
print()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_qrels_dict(qrels_file):
    """Load qrels in pytrec_eval format."""
    qrels = {}
    with open(qrels_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                qid = parts[0]
                did = parts[2]
                rel = int(parts[3])
                
                if qid not in qrels:
                    qrels[qid] = {}
                qrels[qid][did] = rel
    return qrels


def tsv_to_trec(tsv_file, trec_file):
    """Convert TSV ranking to TREC format."""
    import pandas as pd
    
    df = pd.read_csv(tsv_file, sep='\t')
    
    with open(trec_file, 'w') as f:
        for _, row in df.iterrows():
            qid = row['query_id']
            did = row['doc_id']
            rank = row['rank']
            score = row['score']
            f.write(f"{qid}\tQ0\t{did}\t{rank}\t{score}\tRUN\n")


def load_run_dict(run_file):
    """Load run in pytrec_eval format."""
    run = {}
    with open(run_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                qid = parts[0]
                did = parts[2]
                score = float(parts[4])
                
                if qid not in run:
                    run[qid] = {}
                run[qid][did] = score
    return run


def evaluate_run(run_file, qrels):
    """Evaluate a run file."""
    run = load_run_dict(run_file)
    
    # Define metrics as per paper
    metrics = {
        'recall_10', 'recall_100', 'recall_1000',
        'ndcg_cut_10', 'ndcg_cut_100', 'ndcg_cut_1000',
        'map_cut_10', 'map_cut_100', 'map_cut_1000'
    }
    
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)
    results = evaluator.evaluate(run)
    
    # Calculate averages
    avg_metrics = {}
    for metric in metrics:
        values = [q[metric] for q in results.values()]
        avg_metrics[metric] = sum(values) / len(values) if values else 0
    
    return avg_metrics


# Load qrels
print("Loading qrels...")
qrels = load_qrels_dict(QRELS_FILE)
print(f"Loaded {len(qrels)} queries\n")

# Convert TSV files to TREC format
print("Converting first-stage TSV files to TREC format...")
tsv_to_trec(FIRST_STAGE_DIR / "bm25_ranking.tsv", 
            OUTPUT_DIR / "bm25_ranking.txt")
tsv_to_trec(FIRST_STAGE_DIR / "minilm_ranking.tsv",
            OUTPUT_DIR / "minilm_ranking.txt")
tsv_to_trec(FIRST_STAGE_DIR / "colbert_ranking.tsv",
            OUTPUT_DIR / "colbert_ranking.txt")

# Evaluate all methods
methods = {
    'BM25': OUTPUT_DIR / "bm25_ranking.txt",
    'MiniLM': OUTPUT_DIR / "minilm_ranking.txt",
    'ColBERT': OUTPUT_DIR / "colbert_ranking.txt",
    'QAPR(BM25)': QAPR_DIR / "qapr_BM25_ranking.txt",
    'QAPR(MiniLM)': QAPR_DIR / "qapr_MiniLM_ranking.txt",
    'QAPR(ColBERT)': QAPR_DIR / "qapr_ColBERT_ranking.txt"
}

all_results = {}

print("Evaluating all methods...\n")

for method_name, run_file in methods.items():
    if not run_file.exists():
        print(f"Warning: {run_file} not found, skipping {method_name}")
        continue
    
    print(f"Evaluating {method_name}...")
    metrics = evaluate_run(str(run_file), qrels)
    all_results[method_name] = metrics

# Print results table
print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)
print()

# Recall results
print("RECALL@k")
print("-" * 80)
print(f"{'Method':<20} {'@10':>10} {'@100':>10} {'@1000':>10}")
print("-" * 80)
for method, metrics in all_results.items():
    print(f"{method:<20} "
          f"{metrics.get('recall_10', 0):>10.4f} "
          f"{metrics.get('recall_100', 0):>10.4f} "
          f"{metrics.get('recall_1000', 0):>10.4f}")
print()

# NDCG results
print("NDCG@k")
print("-" * 80)
print(f"{'Method':<20} {'@10':>10} {'@100':>10} {'@1000':>10}")
print("-" * 80)
for method, metrics in all_results.items():
    print(f"{method:<20} "
          f"{metrics.get('ndcg_cut_10', 0):>10.4f} "
          f"{metrics.get('ndcg_cut_100', 0):>10.4f} "
          f"{metrics.get('ndcg_cut_1000', 0):>10.4f}")
print()

# MAP results
print("MAP@k")
print("-" * 80)
print(f"{'Method':<20} {'@10':>10} {'@100':>10} {'@1000':>10}")
print("-" * 80)
for method, metrics in all_results.items():
    print(f"{method:<20} "
          f"{metrics.get('map_cut_10', 0):>10.4f} "
          f"{metrics.get('map_cut_100', 0):>10.4f} "
          f"{metrics.get('map_cut_1000', 0):>10.4f}")

print("=" * 80)

# Calculate improvements
print("\nPROPORTIONAL IMPROVEMENTS (QAPR vs Baseline)")
print("=" * 80)

comparisons = [
    ('BM25', 'QAPR(BM25)'),
    ('MiniLM', 'QAPR(MiniLM)'),
    ('ColBERT', 'QAPR(ColBERT)')
]

for baseline, qapr in comparisons:
    if baseline not in all_results or qapr not in all_results:
        continue
    
    print(f"\n{qapr}")
    print("-" * 40)
    
    baseline_metrics = all_results[baseline]
    qapr_metrics = all_results[qapr]
    
    for metric in ['recall_1000', 'ndcg_cut_10', 'map_cut_10']:
        baseline_val = baseline_metrics.get(metric, 0)
        qapr_val = qapr_metrics.get(metric, 0)
        
        if baseline_val > 0:
            improvement = ((qapr_val - baseline_val) / baseline_val) * 100
            print(f"  {metric:20s}: {improvement:+6.2f}%")

print("\n" + "=" * 80)

# Save detailed results
with open(OUTPUT_DIR / "detailed_results.txt", 'w') as f:
    f.write("WPI-PR Evaluation Results\n")
    f.write("Paper: Beyond BM25: Strengthening First-Stage Patent Retrieval\n")
    f.write("=" * 80 + "\n\n")
    
    for method, metrics in all_results.items():
        f.write(f"{method}:\n")
        for metric, value in sorted(metrics.items()):
            f.write(f"  {metric:20s}: {value:.4f}\n")
        f.write("\n")

print(f"\nDetailed results saved to: {OUTPUT_DIR / 'detailed_results.txt'}")
