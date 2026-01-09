"""
Data preparation functions for training and prediction.

These functions create dataframes for training ML models and making predictions
in both Global Models (GMs) and Multiple Models (MMs) architectures.
"""

import pandas as pd
from typing import Dict, List, Any


def dataframe_for_training(
    results_centralized: List[Any],
    results_main: Dict[str, List[Any]],
    results_cori_list: List[tuple]
) -> pd.DataFrame:
    """
    Create training dataframe for Global Models (GMs).
    
    Uses overlapped documents between resources and centralized index.
    Each row represents a document with scores from all resources as features.
    
    Args:
        results_centralized: Results from centralized index
        results_main: Dictionary mapping collection names to their results
        results_cori_list: List of (score, collection_name) tuples from CORI
        
    Returns:
        DataFrame with columns: [collection1_local_score, ..., Centralized_Score]
    """
    # Get all IDs from centralized index
    centralized_ids = [r.docid for r in results_centralized]
    
    # Find common IDs between all resources and centralized index
    common_ids = []
    for collection in results_main:
        for result in results_main[collection]:
            if result.docid in centralized_ids:
                common_ids.append(result.docid)
    
    # Create column names
    columns = [f"{coll[1]}_local_score" for coll in results_cori_list]
    columns.append('Centralized_Score')
    df = pd.DataFrame(columns=columns)
    
    # Build dataframe
    for common_id in common_ids:
        # Get centralized score
        centr_score = None
        for centr_result in results_centralized:
            if common_id == centr_result.docid:
                centr_score = centr_result.score
                break
        
        # Get local scores from all collections
        df_dict = {}
        for collection in results_main:
            for result in results_main[collection]:
                if common_id == result.docid:
                    df_dict[f"{collection}_local_score"] = result.score
                    break
        
        # Fill missing collections with 0
        for key in columns:
            if key not in df_dict:
                df_dict[key] = 0
        
        df_dict['Centralized_Score'] = centr_score
        df.loc[common_id] = pd.Series(df_dict)
    
    return df


def dataframe_for_predictions(
    results_centralized: List[Any],
    results_main: Dict[str, List[Any]],
    results_cori_list: List[tuple]
) -> pd.DataFrame:
    """
    Create prediction dataframe for Global Models (GMs).
    
    Uses non-overlapped documents (documents not in centralized index).
    
    Args:
        results_centralized: Results from centralized index
        results_main: Dictionary mapping collection names to their results
        results_cori_list: List of (score, collection_name) tuples from CORI
        
    Returns:
        DataFrame with columns: [collection1_local_score, ...]
    """
    centralized_ids = [r.docid for r in results_centralized]
    
    # Find common IDs
    common_ids = []
    for collection in results_main:
        for result in results_main[collection]:
            if result.docid in centralized_ids:
                common_ids.append(result.docid)
    
    # Create column names
    columns = [f"{coll[1]}_local_score" for coll in results_cori_list]
    df = pd.DataFrame(columns=columns)
    
    # Get all document IDs
    all_ids = []
    for collection in results_main:
        for result in results_main[collection]:
            all_ids.append(result.docid)
    
    # Build dataframe for non-common documents
    for doc_id in all_ids:
        if doc_id not in common_ids:
            df_dict = {}
            for collection in results_main:
                for result in results_main[collection]:
                    if doc_id == result.docid:
                        df_dict[f"{collection}_local_score"] = result.score
                        break
            
            # Fill missing collections with 0
            for key in columns:
                if key not in df_dict:
                    df_dict[key] = 0
            
            df.loc[doc_id] = pd.Series(df_dict)
    
    return df


def dataframe_for_training_SSL(
    results_centralized: List[Any],
    results_main: Dict[str, List[Any]],
    results_cori_list: List[tuple],
    collection: str
) -> pd.DataFrame:
    """
    Create training dataframe for Multiple Models (MMs) - one model per collection.
    
    Uses overlapped documents between a specific collection and centralized index.
    
    Args:
        results_centralized: Results from centralized index
        results_main: Dictionary mapping collection names to their results
        results_cori_list: List of (score, collection_name) tuples from CORI
        collection: Name of the collection to process
        
    Returns:
        DataFrame with columns: [local_score, Centralized_Score]
    """
    centralized_ids = [r.docid for r in results_centralized]
    
    # Find common IDs for this collection
    common_ids = []
    for result in results_main[collection]:
        if result.docid in centralized_ids:
            common_ids.append(result.docid)
    
    columns = ['local_score', 'Centralized_Score']
    df = pd.DataFrame(columns=columns)
    
    # Build dataframe (limit to first 10 for training as in original code)
    count = 10
    for common_id in common_ids:
        # Get centralized score
        centr_score = None
        for centr_result in results_centralized:
            if common_id == centr_result.docid:
                centr_score = centr_result.score
                break
        
        # Get local score
        local_score = None
        for result in results_main[collection]:
            if result.docid == common_id:
                local_score = result.score
                break
        
        df_dict = {
            'local_score': local_score,
            'Centralized_Score': centr_score
        }
        df.loc[common_id] = pd.Series(df_dict)
        
        count -= 1
        if count == 0:
            break
    
    return df


def dataframe_for_predictions_SSL(
    results_centralized: List[Any],
    results_main: Dict[str, List[Any]],
    results_cori_list: List[tuple],
    collection: str
) -> pd.DataFrame:
    """
    Create prediction dataframe for Multiple Models (MMs).
    
    Uses non-overlapped documents from a specific collection.
    
    Args:
        results_centralized: Results from centralized index
        results_main: Dictionary mapping collection names to their results
        results_cori_list: List of (score, collection_name) tuples from CORI
        collection: Name of the collection to process
        
    Returns:
        DataFrame with columns: [local_score]
    """
    centralized_ids = [r.docid for r in results_centralized]
    
    # Find common IDs
    common_ids = []
    for result in results_main[collection]:
        if result.docid in centralized_ids:
            common_ids.append(result.docid)
    
    columns = ['local_score']
    df = pd.DataFrame(columns=columns)
    
    # Build dataframe for non-common documents
    for result in results_main[collection]:
        if result.docid not in common_ids:
            df_dict = {'local_score': result.score}
            df.loc[result.docid] = pd.Series(df_dict)
    
    return df


def dataframe_for_training_polynomial_x2_SSL(
    results_centralized: List[Any],
    results_main: Dict[str, List[Any]],
    results_cori_list: List[tuple],
    collection: str
) -> pd.DataFrame:
    """
    Create training dataframe for polynomial regression (x^2) in MMs architecture.
    
    Args:
        results_centralized: Results from centralized index
        results_main: Dictionary mapping collection names to their results
        results_cori_list: List of (score, collection_name) tuples from CORI
        collection: Name of the collection to process
        
    Returns:
        DataFrame with columns: [local_score, x2, Centralized_Score]
    """
    centralized_ids = [r.docid for r in results_centralized]
    
    common_ids = []
    for result in results_main[collection]:
        if result.docid in centralized_ids:
            common_ids.append(result.docid)
    
    columns = ['local_score', 'x2', 'Centralized_Score']
    df = pd.DataFrame(columns=columns)
    
    count = 10
    for common_id in common_ids:
        centr_score = None
        for centr_result in results_centralized:
            if common_id == centr_result.docid:
                centr_score = centr_result.score
                break
        
        local_score = None
        for result in results_main[collection]:
            if result.docid == common_id:
                local_score = result.score
                break
        
        df_dict = {
            'local_score': local_score,
            'x2': local_score * local_score,
            'Centralized_Score': centr_score
        }
        df.loc[common_id] = pd.Series(df_dict)
        
        count -= 1
        if count == 0:
            break
    
    return df


def dataframe_for_predictions_polynomial_x2_SSL(
    results_centralized: List[Any],
    results_main: Dict[str, List[Any]],
    results_cori_list: List[tuple],
    collection: str
) -> pd.DataFrame:
    """
    Create prediction dataframe for polynomial regression (x^2) in MMs architecture.
    
    Args:
        results_centralized: Results from centralized index
        results_main: Dictionary mapping collection names to their results
        results_cori_list: List of (score, collection_name) tuples from CORI
        collection: Name of the collection to process
        
    Returns:
        DataFrame with columns: [local_score, x2]
    """
    centralized_ids = [r.docid for r in results_centralized]
    
    common_ids = []
    for result in results_main[collection]:
        if result.docid in centralized_ids:
            common_ids.append(result.docid)
    
    columns = ['local_score', 'x2']
    df = pd.DataFrame(columns=columns)
    
    for result in results_main[collection]:
        if result.docid not in common_ids:
            local_score = result.score
            df_dict = {
                'local_score': local_score,
                'x2': local_score * local_score
            }
            df.loc[result.docid] = pd.Series(df_dict)
    
    return df


def dataframe_for_training_polynomial_x3_SSL(
    results_centralized: List[Any],
    results_main: Dict[str, List[Any]],
    results_cori_list: List[tuple],
    collection: str
) -> pd.DataFrame:
    """
    Create training dataframe for polynomial regression (x^3) in MMs architecture.
    
    Args:
        results_centralized: Results from centralized index
        results_main: Dictionary mapping collection names to their results
        results_cori_list: List of (score, collection_name) tuples from CORI
        collection: Name of the collection to process
        
    Returns:
        DataFrame with columns: [local_score, x2, x3, Centralized_Score]
    """
    centralized_ids = [r.docid for r in results_centralized]
    
    common_ids = []
    for result in results_main[collection]:
        if result.docid in centralized_ids:
            common_ids.append(result.docid)
    
    columns = ['local_score', 'x2', 'x3', 'Centralized_Score']
    df = pd.DataFrame(columns=columns)
    
    count = 10
    for common_id in common_ids:
        centr_score = None
        for centr_result in results_centralized:
            if common_id == centr_result.docid:
                centr_score = centr_result.score
                break
        
        local_score = None
        for result in results_main[collection]:
            if result.docid == common_id:
                local_score = result.score
                break
        
        df_dict = {
            'local_score': local_score,
            'x2': local_score * local_score,
            'x3': local_score * local_score * local_score,
            'Centralized_Score': centr_score
        }
        df.loc[common_id] = pd.Series(df_dict)
        
        count -= 1
        if count == 0:
            break
    
    return df


def dataframe_for_predictions_polynomial_x3_SSL(
    results_centralized: List[Any],
    results_main: Dict[str, List[Any]],
    results_cori_list: List[tuple],
    collection: str
) -> pd.DataFrame:
    """
    Create prediction dataframe for polynomial regression (x^3) in MMs architecture.
    
    Args:
        results_centralized: Results from centralized index
        results_main: Dictionary mapping collection names to their results
        results_cori_list: List of (score, collection_name) tuples from CORI
        collection: Name of the collection to process
        
    Returns:
        DataFrame with columns: [local_score, x2, x3]
    """
    centralized_ids = [r.docid for r in results_centralized]
    
    common_ids = []
    for result in results_main[collection]:
        if result.docid in centralized_ids:
            common_ids.append(result.docid)
    
    columns = ['local_score', 'x2', 'x3']
    df = pd.DataFrame(columns=columns)
    
    for result in results_main[collection]:
        if result.docid not in common_ids:
            local_score = result.score
            df_dict = {
                'local_score': local_score,
                'x2': local_score * local_score,
                'x3': local_score * local_score * local_score
            }
            df.loc[result.docid] = pd.Series(df_dict)
    
    return df
