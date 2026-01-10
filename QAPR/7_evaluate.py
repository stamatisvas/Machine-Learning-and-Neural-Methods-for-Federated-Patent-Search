#!/usr/bin/env python3
"""
Step 7: Evaluate Results
=========================
Evaluates re-ranking results using MAP, RECALL, and PRES metrics.
"""

import os
import subprocess
from pathlib import Path
from utils import load_config, print_results

config = load_config()

RESULTS_DIR = Path(config['output_dir']) / "results"
QRELS_FILE = config['qrels_file']

print("=" * 80)
print("Step 7: Evaluate Results")
print("=" * 80)
print(f"Qrels: {QRELS_FILE}")
print(f"Results directory: {RESULTS_DIR}")
print()

# Check for trec_eval
try:
    subprocess.run(['trec_eval', '-h'], capture_output=True, check=False)
    use_trec_eval = True
except FileNotFoundError:
    print("Warning: trec_eval not found, will use pytrec_eval")
    use_trec_eval = False
    import pytrec_eval


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


def evaluate_with_pytrec(run_file, qrels_file):
    """Evaluate using pytrec_eval."""
    qrels = load_qrels_dict(qrels_file)
    run = load_run_dict(run_file)
    
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, 
        {'map', 'recall_1000', 'P_10', 'P_20', 'ndcg'}
    )
    
    results = evaluator.evaluate(run)
    
    # Calculate averages
    metrics = {
        'MAP': sum(q['map'] for q in results.values()) / len(results),
        'RECALL@1000': sum(q['recall_1000'] for q in results.values()) / len(results),
        'P@10': sum(q['P_10'] for q in results.values()) / len(results),
        'P@20': sum(q['P_20'] for q in results.values()) / len(results),
        'NDCG': sum(q['ndcg'] for q in results.values()) / len(results)
    }
    
    return metrics


def evaluate_with_trec_eval(run_file, qrels_file):
    """Evaluate using trec_eval command."""
    result = subprocess.run(
        ['trec_eval', '-m', 'all_trec', qrels_file, run_file],
        capture_output=True,
        text=True
    )
    
    output = result.stdout
    
    # Parse output
    metrics = {}
    for line in output.split('\n'):
        if line.strip():
            parts = line.split()
            if len(parts) >= 3:
                metric_name = parts[0]
                value = float(parts[2])
                
                if metric_name == 'map':
                    metrics['MAP'] = value
                elif metric_name == 'recall_1000':
                    metrics['RECALL@1000'] = value
                elif metric_name == 'P_10':
                    metrics['P@10'] = value
                elif metric_name == 'P_20':
                    metrics['P@20'] = value
                elif metric_name == 'ndcg':
                    metrics['NDCG'] = value
    
    return metrics


# Evaluate all result files
result_files = list(RESULTS_DIR.glob('*.txt'))

if not result_files:
    print("No result files found!")
    exit(1)

print(f"Found {len(result_files)} result files to evaluate\n")

all_results = {}

for result_file in result_files:
    model_name = result_file.stem.replace('_ranking', '').upper()
    
    print(f"Evaluating {model_name}...")
    
    if use_trec_eval:
        metrics = evaluate_with_trec_eval(str(result_file), QRELS_FILE)
    else:
        metrics = evaluate_with_pytrec(str(result_file), QRELS_FILE)
    
    all_results[model_name] = metrics
    print_results(metrics, model_name)

# Compare results
print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)
print(f"{'Model':<20} {'MAP':>10} {'RECALL':>10} {'P@10':>10} {'P@20':>10} {'NDCG':>10}")
print("-" * 80)

for model_name, metrics in all_results.items():
    print(f"{model_name:<20} {metrics.get('MAP', 0):>10.4f} "
          f"{metrics.get('RECALL@1000', 0):>10.4f} "
          f"{metrics.get('P@10', 0):>10.4f} "
          f"{metrics.get('P@20', 0):>10.4f} "
          f"{metrics.get('NDCG', 0):>10.4f}")

print("=" * 80)

# Save results to file
results_summary = RESULTS_DIR / "evaluation_summary.txt"
with open(results_summary, 'w') as f:
    f.write("QAPR Evaluation Results\n")
    f.write("=" * 80 + "\n\n")
    
    for model_name, metrics in all_results.items():
        f.write(f"{model_name}:\n")
        for metric, value in metrics.items():
            f.write(f"  {metric}: {value:.4f}\n")
        f.write("\n")

print(f"\nResults summary saved to: {results_summary}")
