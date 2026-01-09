"""
Random merging baseline: merge results randomly.

This provides a lower bound for comparison.
"""

from typing import Dict, List, Tuple, Any
import random
from core.merging_utils import remove_duplicates
import config


def merge_random(
    results_main: Dict[str, List[Any]]
) -> List[Tuple[float, str]]:
    """
    Merge results randomly.
    
    Args:
        results_main: Dictionary mapping collection names to their results
        
    Returns:
        List of (random_score, docid) tuples, randomly shuffled
    """
    final_list = []
    
    for collection in results_main:
        for result in results_main[collection]:
            # Assign random score
            random_score = random.random()
            final_list.append([random_score, result.docid])
    
    # Remove duplicates
    final_list = remove_duplicates(final_list)
    
    # Shuffle randomly
    random.shuffle(final_list)
    
    return final_list[0:config.FINAL_RESULTS]
