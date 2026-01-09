"""
CORI (Collection Inference Retrieval Network) implementation.

CORI is used for source selection in federated search.
"""

import math
from typing import Dict, List, Tuple
from pyserini import index


def get_number_of_collections_contain_word(word: str, the_index: Dict) -> int:
    """
    Count the number of collections that contain a given word.
    
    Args:
        word: The word to search for
        the_index: Dictionary of index readers for each collection
        
    Returns:
        Number of collections containing the word
    """
    count = 0
    for coll in the_index:
        try:
            df, cf = the_index[coll].get_term_counts(word)
            if cf > 0:
                count += 1
        except:
            continue
    return count


def get_avg_cw(index_reader_sample: Dict) -> float:
    """
    Calculate the average number of terms across all collections.
    
    Args:
        index_reader_sample: Dictionary of index readers for each collection
        
    Returns:
        Average number of terms across all collections
    """
    summary = 0
    index_count = 0
    for coll in index_reader_sample:
        nt = index_reader_sample[coll].stats()['total_terms']
        summary = summary + nt
        index_count += 1
    res = summary / index_count
    return res


def CORI2(topic: str, my_index: Dict, avg_cw: float) -> List[Tuple[float, str]]:
    """
    CORI source selection algorithm (simplified version).
    
    Args:
        topic: Query text
        my_index: Dictionary of index readers for each collection
        avg_cw: Average number of terms across collections
        
    Returns:
        List of (score, collection_name) tuples, sorted by score descending
    """
    my_list = []
    topic = topic.split()
    topic = set(topic)
    words = []
    for word in topic:
        if (word.startswith('section') and word.endswith('section')) or \
           (word.startswith('class') and word.endswith('class')) or \
           (word.startswith('subclass') and word.endswith('subclass')) or \
           (word.startswith('group') and word.endswith('group')) or \
           (word.startswith('subgroup') and word.endswith('subgroup')):
            words.append(word)
    topic = words
    Nc = len(my_index)
    b = 0.4
    score = {}
    count = {}
    for coll in my_index:
        count[coll] = 0
        score[coll] = 0

    for word in topic:
        num_of_coll = get_number_of_collections_contain_word(word, my_index)
        try:
            I = math.log((Nc + 0.5) / num_of_coll) / math.log(Nc + 1)
        except:
            continue
        for coll in my_index:
            cw = my_index[coll].stats()['total_terms']
            try:
                df, cf = my_index[coll].get_term_counts(word)
                T = (df) / (df + 50 + (150 * (cw / avg_cw)))
                P = b + (1 - b) * T * I

                count[coll] += 1
                score[coll] = score[coll] + P
            except:
                break

    for i in score:
        my_list.append([score[i]/count[i], i])

    my_list.sort(reverse=True)

    return my_list


def CORI2_for_CORI(topic: str, my_index: Dict, avg_cw: float) -> Tuple[List[Tuple[float, str]], Dict, Dict]:
    """
    CORI source selection algorithm with min/max normalization values.
    
    Used for CORI merging algorithm which needs Cmin and Cmax for normalization.
    
    Args:
        topic: Query text
        my_index: Dictionary of index readers for each collection
        avg_cw: Average number of terms across collections
        
    Returns:
        Tuple of (results_list, Cmin_dict, Cmax_dict)
    """
    my_list = []
    topic = topic.split()
    topic = set(topic)
    words = []
    for word in topic:
        if (word.startswith('section') and word.endswith('section')) or \
           (word.startswith('class') and word.endswith('class')) or \
           (word.startswith('subclass') and word.endswith('subclass')) or \
           (word.startswith('group') and word.endswith('group')) or \
           (word.startswith('subgroup') and word.endswith('subgroup')):
            words.append(word)
    topic = words
    Nc = len(my_index)
    b = 0.4
    score = {}
    count = {}
    Cmin = {}
    Cmax = {}
    for coll in my_index:
        count[coll] = 0
        score[coll] = 0
        Cmin[coll] = 0
        Cmax[coll] = 0

    for word in topic:
        num_of_coll = get_number_of_collections_contain_word(word, my_index)
        try:
            I = math.log((Nc + 0.5) / num_of_coll) / math.log(Nc + 1)
        except:
            continue
        for coll in my_index:
            cw = my_index[coll].stats()['total_terms']
            try:
                df, cf = my_index[coll].get_term_counts(word)
                T = (df) / (df + 50 + (150 * (cw / avg_cw)))
                P = b + (1 - b) * T * I
                Pmin = b
                Pmax = b + (1 - b) * I
                Cmin[coll] = Cmin[coll] + Pmin
                Cmax[coll] = Cmax[coll] + Pmax

                count[coll] += 1
                score[coll] = score[coll] + P
            except:
                break

    for i in score:
        my_list.append([score[i]/(count[i]), i])

    my_list.sort(reverse=True)

    for i in Cmin:
        Cmin[i] = Cmin[i]/count[i]
    for i in Cmax:
        Cmax[i] = Cmax[i]/count[i]

    return my_list, Cmin, Cmax


def CORI(topic: str, my_index: Dict, avg_cw: float) -> List[Tuple[float, str]]:
    """
    CORI source selection algorithm (original version).
    
    Args:
        topic: Query text
        my_index: Dictionary of index readers for each collection
        avg_cw: Average number of terms across collections
        
    Returns:
        List of top 19 (score, collection_name) tuples, sorted by score descending
    """
    my_list = []
    topic = topic.split()
    topic_set = set(topic)
    topic = topic_set
    Nc = len(my_index)
    b = 0.4

    for coll in my_index:
        cw = my_index[coll].stats()['total_terms']
        score = 0
        count = 1
        for word in topic:
            try:
                df, cf = my_index[coll].get_term_counts(word)
                T = (df) / (df + 50 + (150 * (cw / avg_cw)))
                num_of_coll = get_number_of_collections_contain_word(word, my_index)
                I = math.log((Nc + 0.5) / num_of_coll) / math.log(Nc + 1)
                P = b + (1 - b) * T * I
                score = score + P
                count += 1
            except:
                continue
        my_list.append([score / count, coll])
    my_list.sort(reverse=True)

    return my_list[0:19]
