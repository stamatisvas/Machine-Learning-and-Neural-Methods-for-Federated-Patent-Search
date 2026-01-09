"""
Uncooperative environment handler.

In uncooperative environments, documents are returned as ranked lists
with no scores. We assign artificial scores linearly.
"""

from typing import Dict, List, Any
from pyserini.search import SimpleSearcher
import config


class UncooperativeEnvironment:
    """Handler for uncooperative federated search environment."""
    
    def __init__(self, collection_index_path: str):
        """
        Initialize uncooperative environment.
        
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
        collection_name: str = None
    ) -> List[Any]:
        """
        Assign artificial scores linearly according to rank.
        
        Formula: A(rank_i) = 0.6 - (0.6 - 0.4) * (rank_i - 1) / (n - 1)
        First document gets 0.6, last gets 0.4.
        
        Args:
            results: List of search results (ranked list)
            collection_name: Not used in simple uncooperative environment
            
        Returns:
            Results with assigned scores
        """
        if len(results) == 0:
            return results
        
        results_length = len(results)
        step = (config.ARTIFICIAL_SCORE_FIRST - config.ARTIFICIAL_SCORE_LAST) / (results_length - 1)
        final_score = config.ARTIFICIAL_SCORE_FIRST
        
        for result in results:
            result.score = final_score
            final_score -= step
        
        return results
