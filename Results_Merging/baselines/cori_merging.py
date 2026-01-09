"""
CORI (Collection Inference Retrieval Network) results merging implementation.

CORI uses a weighted score merging scheme that combines document scores
with collection selection scores.
"""

from typing import Dict, List, Tuple, Any
from pyserini.search import SimpleSearcher
from core.cori import CORI2_for_CORI
from core.merging_utils import remove_duplicates
import config


def merge_cori(
    topic_text: str,
    results_main: Dict[str, List[Any]],
    index_reader_sample: Dict,
    avg_cw: float,
    collection_index_path: str
) -> List[Tuple[float, str]]:
    """
    Merge results using CORI algorithm.
    
    Args:
        topic_text: Query text
        results_main: Dictionary mapping collection names to their results
        index_reader_sample: Dictionary of index readers for source selection
        avg_cw: Average number of terms across collections
        collection_index_path: Base path to collection indices
        
    Returns:
        List of (score, docid) tuples, sorted by score descending
    """
    # Get CORI scores with min/max for normalization
    results_cori_list, Cmin, Cmax = CORI2_for_CORI(
        topic=topic_text,
        my_index=index_reader_sample,
        avg_cw=avg_cw
    )
    
    results_cori_list = results_cori_list[0:config.NUM_COLLECTIONS_TO_QUERY]
    final_merged_list = []
    
    for relevant_collection in results_cori_list:
        collection_name = relevant_collection[1]
        cori_score = relevant_collection[0]
        
        # Normalize CORI score
        Cminimum = Cmin[collection_name]
        Cmaximum = Cmax[collection_name]
        Ctonos = (cori_score - Cminimum) / (Cmaximum - Cminimum)
        
        # Merge scores: normalized_score = (doc_score + 0.4*doc_score*Ctonos) / 1.4
        for result in results_main[collection_name]:
            normalized_score = (result.score + 0.4 * result.score * Ctonos) / 1.4
            final_merged_list.append([normalized_score, result.docid])
    
    # Remove duplicates and sort
    final_merged_list = remove_duplicates(final_merged_list)
    final_merged_list.sort(reverse=True)
    
    return final_merged_list[0:config.FINAL_RESULTS]
