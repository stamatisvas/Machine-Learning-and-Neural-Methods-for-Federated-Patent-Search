"""
Uncooperative weighted environment handler.

In uncooperative weighted environments, documents are returned as ranked lists
with no scores. We assign artificial scores and multiply by source selection score.
"""

from typing import Dict, List, Any
from pyserini.search import SimpleSearcher
import config


class UncooperativeWeightedEnvironment:
    """Handler for uncooperative weighted federated search environment."""
    
    def __init__(self, collection_index_path: str):
        """
        Initialize uncooperative weighted environment.
        
        Args:
            collection_index_path: Base path to collection indices
        """
        self.collection_index_path = collection_index_path
    
    def get_results_from_collection(
        self,
        collection_name: str,
        topic_text: str,
        num_results: int = None
    ) -> List[Any]:
        """
        Get results from a collection (as ranked list, no scores).
        
        Args:
            collection_name: Name of the collection
            topic_text: Query text
            num_results: Number of results to retrieve
            
        Returns:
            List of search results (scores will be assigned)
        """
        if num_results is None:
            num_results = config.RESULTS_PER_COLLECTION
        
        searcher = SimpleSearcher(
            f"{self.collection_index_path}/{collection_name}/"
        )
        results = searcher.search(topic_text, num_results)
        return results
    
    def assign_scores(
        self,
        results: List[Any],
        collection_name: str,
        cori_score: float
    ) -> List[Any]:
        """
        Assign artificial scores weighted by source selection score.
        
        Formula: S(rank_i) = A(rank_i) * C_j
        where A(rank_i) is artificial score and C_j is CORI source selection score.
        
        Args:
            results: List of search results (ranked list)
            collection_name: Name of the collection (for logging)
            cori_score: CORI source selection score for this collection
            
        Returns:
            Results with assigned weighted scores
        """
        if len(results) == 0:
            return results
        
        results_length = len(results)
        step = (config.ARTIFICIAL_SCORE_FIRST - config.ARTIFICIAL_SCORE_LAST) / (results_length - 1)
        final_score = config.ARTIFICIAL_SCORE_FIRST
        
        for result in results:
            # Assign artificial score and multiply by CORI score
            artificial_score = final_score
            result.score = artificial_score * cori_score
            final_score -= step
        
        return results
