"""
Cooperative environment handler.

In cooperative environments, documents are returned with scores from
the remote search engines.
"""

from typing import Dict, List, Any
from pyserini.search import SimpleSearcher
import config


class CooperativeEnvironment:
    """Handler for cooperative federated search environment."""
    
    def __init__(self, collection_index_path: str):
        """
        Initialize cooperative environment.
        
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
        Get results from a collection with scores.
        
        Args:
            collection_name: Name of the collection
            topic_text: Query text
            num_results: Number of results to retrieve
            
        Returns:
            List of search results with scores
        """
        if num_results is None:
            num_results = config.RESULTS_PER_COLLECTION
        
        searcher = SimpleSearcher(
            f"{self.collection_index_path}/{collection_name}/"
        )
        results = searcher.search(topic_text, num_results)
        return results
    
    def assign_scores(self, results: List[Any], collection_name: str = None) -> List[Any]:
        """
        In cooperative environment, scores are already present.
        
        Args:
            results: List of search results
            collection_name: Not used in cooperative environment
            
        Returns:
            Results with scores (unchanged)
        """
        # Scores are already present, no need to assign
        return results
