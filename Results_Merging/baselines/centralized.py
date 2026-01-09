"""
Centralized approach: query a single centralized index.

This is the baseline that doesn't use federated search - all documents
are in a single index.
"""

from typing import List, Tuple, Any
import config


def merge_centralized(
    results_centralized: List[Any]
) -> List[Tuple[float, str]]:
    """
    Return results from centralized index (no merging needed).
    
    Args:
        results_centralized: Results from centralized index
        
    Returns:
        List of (score, docid) tuples, sorted by score descending
    """
    final_list = []
    seen_ids = []
    
    for result in results_centralized:
        if result.docid not in seen_ids:
            final_list.append([result.score, result.docid])
            seen_ids.append(result.docid)
    
    final_list.sort(reverse=True)
    return final_list[0:config.FINAL_RESULTS]
