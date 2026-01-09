"""
SSL (Semi-Supervised Learning) results merging implementation.

SSL uses linear regression to map local collection-specific scores to
global comparable scores from the centralized index.
"""

from typing import Dict, List, Tuple, Any
from core.cori import CORI2, CORI2_for_CORI
from core.data_preparation import (
    dataframe_for_training_SSL,
    dataframe_for_predictions_SSL
)
from core.merging_utils import remove_duplicates, final_merged_list_for_SSL
from models.ml_models import create_linear_regression, train_ml_model
import config


def merge_ssl(
    topic_text: str,
    results_centralized: List[Any],
    results_main: Dict[str, List[Any]],
    index_reader_sample: Dict,
    avg_cw: float,
    collection_index_path: str
) -> List[Tuple[float, str]]:
    """
    Merge results using SSL algorithm.
    
    Args:
        topic_text: Query text
        results_centralized: Results from centralized index
        results_main: Dictionary mapping collection names to their results
        index_reader_sample: Dictionary of index readers for source selection
        avg_cw: Average number of terms across collections
        collection_index_path: Base path to collection indices
        
    Returns:
        List of (score, docid) tuples, sorted by score descending
    """
    try:
        # Get CORI source selection results
        results_cori_list = CORI2(
            topic=topic_text,
            my_index=index_reader_sample,
            avg_cw=avg_cw
        )
        results_cori_list = results_cori_list[0:config.NUM_COLLECTIONS_TO_QUERY]
        
        the_final_merged_list = []
        
        # Train one model per collection (Multiple Models approach)
        for collection in results_main:
            # Prepare training and prediction dataframes
            dataframe_training = dataframe_for_training_SSL(
                results_centralized, results_main, results_cori_list, collection
            )
            dataframe_predict = dataframe_for_predictions_SSL(
                results_centralized, results_main, results_cori_list, collection
            )
            
            # Skip if no training data
            if len(dataframe_training) == 0:
                continue
            
            # Train linear regression model
            lin_reg = create_linear_regression()
            model = train_ml_model(lin_reg, df=dataframe_training)
            predictions = model.predict(dataframe_predict)
            
            # Merge results
            the_final_merged_list = final_merged_list_for_SSL(
                dataframe_training, dataframe_predict, predictions, the_final_merged_list
            )
        
        # Remove duplicates and sort
        the_final_merged_list = remove_duplicates(the_final_merged_list)
        the_final_merged_list.sort(reverse=True)
        
        return the_final_merged_list[0:config.FINAL_RESULTS]
    
    except Exception as e:
        # Fallback to CORI if SSL fails
        from baselines.cori_merging import merge_cori
        return merge_cori(
            topic_text, results_main, index_reader_sample, avg_cw, collection_index_path
        )
