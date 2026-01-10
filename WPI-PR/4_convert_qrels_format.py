#!/usr/bin/env python3
"""
Step 4: Convert to Qrels Format
Converts ground truth to TREC qrels format
"""

INPUT_FILE = 'wpi_ground_truths.txt'
OUTPUT_FILE = 'wpi_qrels.txt'

print(f"Input: {INPUT_FILE}")
print(f"Output: {OUTPUT_FILE}")
print("-" * 80)

count = 0

with open(OUTPUT_FILE, 'w', encoding='utf-8') as writer:
    with open(INPUT_FILE, 'r', encoding='utf-8') as reader:
        for line in reader:
            parts = line.strip().split("<sep>")
            if len(parts) != 2:
                continue
            
            query_id = parts[0]
            relevant_docs = parts[1].split()
            
            for doc_id in relevant_docs:
                writer.write(f"{query_id}\t0\t{doc_id}\t1\n")
                count += 1

print("-" * 80)
print(f"Complete! Wrote {count} relevance judgments")
print(f"Output: {OUTPUT_FILE}")
