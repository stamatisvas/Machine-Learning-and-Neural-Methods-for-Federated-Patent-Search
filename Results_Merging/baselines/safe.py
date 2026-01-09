"""
SAFE (Sample-Agglomerate Fitting Estimate) results merging implementation.

SAFE maps document ranks to centralized scores using linear regression.
It doesn't rely on overlapped documents between collections.
"""

from typing import Dict, List, Tuple, Any
import pandas as pd
from pyserini import index
from core.cori import CORI2, CORI2_for_CORI
from core.merging_utils import remove_duplicates
from models.ml_models import create_linear_regression
import config


def merge_safe(
    topic_text: str,
    results_centralized: List[Any],
    results_main: Dict[str, List[Any]],
    index_reader_sample: Dict,
    index_reader_full: Dict,
    avg_cw: float,
    collection_index_path: str
) -> List[Tuple[float, str]]:
    """
    Merge results using SAFE algorithm.
    
    Args:
        topic_text: Query text
        results_centralized: Results from centralized index
        results_main: Dictionary mapping collection names to their results
        index_reader_sample: Dictionary of index readers for sample collections
        index_reader_full: Dictionary of index readers for full collections
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
        results_cori_list = results_cori_list[0:10]  # SAFE uses top 10
        
        # Get collection names
        cori_collections = [coll[1] for coll in results_cori_list]
        
        # Get collection sizes
        full_sizes = {}
        sample_sizes = {}
        for coll in cori_collections:
            if coll in index_reader_full:
                full_sizes[coll] = index_reader_full[coll].stats()['documents']
            if coll in index_reader_sample:
                sample_sizes[coll] = index_reader_sample[coll].stats()['documents']
        
        the_final_merged_list = []
        
        for coll in cori_collections:
            # Create rank mapping for documents in this collection
            main_res = {}
            for rank, doc in enumerate(results_main[coll], start=1):
                main_res[doc.docid] = rank
            
            # Build training data: ranks and scores from centralized index
            ranks = []
            scores = []
            counter_c = 0
            
            for doc in results_centralized:
                # Check if document exists in sample collection
                if index_reader_sample[coll].doc(doc.docid):
                    counter_c += 1
                    if doc.docid in main_res:
                        # Document is in both: use actual rank
                        ranks.append(main_res[doc.docid])
                        scores.append(doc.score)
                    else:
                        # Estimate rank based on sample position
                        rank = round(counter_c * full_sizes[coll] / sample_sizes[coll])
                        ranks.append(rank)
                        scores.append(doc.score)
            
            # Skip if no training data
            if len(ranks) == 0:
                continue
            
            # Create training dataframe
            df_train = pd.DataFrame({'ranks': ranks, 'scores': scores})
            
            # Create prediction dataframe with ranks from main results
            pred_ranks = [main_res[docid] for docid in main_res]
            df_pred = pd.DataFrame({'ranks': pred_ranks})
            
            # Train linear regression: rank -> score
            lin_reg = create_linear_regression()
            lin_reg.fit(df_train[['ranks']], df_train['scores'])
            predictions = lin_reg.predict(df_pred)
            
            # Add to final list
            for num, doc in enumerate(results_main[coll]):
                the_final_merged_list.append([predictions[num], doc.docid])
        
        # Remove duplicates and sort
        the_final_merged_list = remove_duplicates(the_final_merged_list)
        the_final_merged_list.sort(reverse=True)
        
        return the_final_merged_list[0:config.FINAL_RESULTS]
    
    except Exception as e:
        # Fallback to CORI if SAFE fails
        from baselines.cori_merging import merge_cori
        return merge_cori(
            topic_text, results_main, index_reader_sample, avg_cw, collection_index_path
        )
