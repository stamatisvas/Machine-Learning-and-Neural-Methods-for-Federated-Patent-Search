"""
Utilities for merging results from different sources.
"""

import pandas as pd
from typing import List, Tuple, Any


def remove_duplicates(my_list: List[Tuple[float, str]]) -> List[Tuple[float, str]]:
    """
    Remove duplicate document IDs from a list of (score, docid) tuples.
    
    Keeps the first occurrence of each document ID.
    
    Args:
        my_list: List of (score, docid) tuples
        
    Returns:
        List with duplicates removed
    """
    new_list = []
    id_list = []
    for line in my_list:
        if line[1] in id_list:
            continue
        elif line[1] not in id_list:
            id_list.append(line[1])
            new_list.append([line[0], line[1]])
    
    return new_list


def final_merged_list_for_SSL(
    dataframe_training: pd.DataFrame,
    dataframe_predictions: pd.DataFrame,
    predictions: List[float],
    final_merged_results_list: List[Tuple[float, str]]
) -> List[Tuple[float, str]]:
    """
    Create final merged list for SSL/Multiple Models approach.
    
    Combines training data scores (from overlapped documents) with predictions
    (for non-overlapped documents).
    
    Args:
        dataframe_training: Training dataframe with overlapped documents
        dataframe_predictions: Prediction dataframe with non-overlapped documents
        predictions: Model predictions for non-overlapped documents
        final_merged_results_list: Existing list to append to
        
    Returns:
        Final merged list of (score, docid) tuples
    """
    j = 0
    # Add training scores (overlapped documents)
    for row in dataframe_training.iterrows():
        doc_id = row[0]
        score = row[1]['Centralized_Score']
        final_merged_results_list.append([score, doc_id])
    
    # Add predictions (non-overlapped documents)
    for i, row in dataframe_predictions.iterrows():
        doc_id = i
        final_merged_results_list.append([predictions[j], doc_id])
        j += 1
    
    return final_merged_results_list


def final_merged_list_for_simple_machine_learning(
    dataframe_training: pd.DataFrame,
    dataframe_predictions: pd.DataFrame,
    predictions: List[float]
) -> List[Tuple[float, str]]:
    """
    Create final merged list for simple ML models (non-DNN).
    
    Args:
        dataframe_training: Training dataframe with overlapped documents
        dataframe_predictions: Prediction dataframe with non-overlapped documents
        predictions: Model predictions for non-overlapped documents
        
    Returns:
        Final merged list of (score, docid) tuples, sorted by score descending
    """
    j = 0
    final_merged_results_list = []
    
    # Add training scores (overlapped documents)
    for row in dataframe_training.iterrows():
        doc_id = row[0]
        score = row[1]['Centralized_Score']
        final_merged_results_list.append([score, doc_id])
    
    # Add predictions (non-overlapped documents)
    for i, row in dataframe_predictions.iterrows():
        doc_id = i
        final_merged_results_list.append([predictions[j], doc_id])
        j += 1
    
    final_merged_results_list.sort(reverse=True)
    return final_merged_results_list


def final_merged_list_for_deep_learning(
    dataframe_training: pd.DataFrame,
    dataframe_predictions: pd.DataFrame,
    predictions: List[List[float]]
) -> List[Tuple[float, str]]:
    """
    Create final merged list for deep learning models (DNN).
    
    Args:
        dataframe_training: Training dataframe with overlapped documents
        dataframe_predictions: Prediction dataframe with non-overlapped documents
        predictions: Model predictions (nested list format from DNN)
        
    Returns:
        Final merged list of (score, docid) tuples, sorted by score descending
    """
    j = 0
    final_merged_results_list = []
    
    # Add training scores (overlapped documents)
    for row in dataframe_training.iterrows():
        doc_id = row[0]
        score = row[1]['Centralized_Score']
        final_merged_results_list.append([score, doc_id])
    
    # Add predictions (non-overlapped documents)
    for i, row in dataframe_predictions.iterrows():
        doc_id = i
        final_merged_results_list.append([predictions[j][0], doc_id])
        j += 1
    
    final_merged_results_list.sort(reverse=True)
    return final_merged_results_list
