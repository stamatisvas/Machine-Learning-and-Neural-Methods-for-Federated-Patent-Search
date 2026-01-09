"""
Main script to run MLRM experiments.

This script can run experiments for:
- Different environments (cooperative, uncooperative, uncooperative_weighted)
- Different models (MMs, GMs with various ML algorithms)
- Different baselines (CORI, SSL, SAFE, Centralized, Random)
- Different scenarios (realistic, optimal, random)
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyserini.search import SimpleSearcher
from pyserini import index
import config
from core.cori import CORI2, CORI2_for_CORI, get_avg_cw
from core.data_preparation import (
    dataframe_for_training,
    dataframe_for_predictions,
    dataframe_for_training_SSL,
    dataframe_for_predictions_SSL,
    dataframe_for_training_polynomial_x2_SSL,
    dataframe_for_predictions_polynomial_x2_SSL,
    dataframe_for_training_polynomial_x3_SSL,
    dataframe_for_predictions_polynomial_x3_SSL,
)
from core.merging_utils import remove_duplicates, final_merged_list_for_SSL, final_merged_list_for_simple_machine_learning
from models.ml_models import (
    train_ml_model,
    create_random_forest,
    create_decision_tree,
    create_svr,
    create_linear_regression,
    create_polynomial_x2,
    create_polynomial_x3,
    create_dnn,
    train_dnn,
)
from baselines.cori_merging import merge_cori
from baselines.ssl import merge_ssl
from baselines.safe import merge_safe
from baselines.centralized import merge_centralized
from baselines.random_merging import merge_random
from environments.cooperative import CooperativeEnvironment
from environments.uncooperative import UncooperativeEnvironment
from environments.uncooperative_weighted import UncooperativeWeightedEnvironment


def load_topics(topics_path: str) -> list:
    """Load topics from file."""
    with open(topics_path, 'r', encoding='utf-8') as f:
        topics = f.read()
    return topics.split('<seperator>')


def parse_topic(topic: str) -> tuple:
    """Parse topic into topic_id and topic_text."""
    parts = topic.split('<id_sep>')
    topic_id = parts[0]
    topic_text = parts[1] if len(parts) > 1 else ''
    # Limit to first 1000 words as per paper
    topic_text = ' '.join(topic_text.split()[:1000])
    return topic_id, topic_text


def run_mm_experiment(
    topic_id: str,
    topic_text: str,
    model_name: str,
    environment,
    results_centralized,
    results_main,
    results_cori_list,
    index_reader_sample,
    avg_cw,
    collection_index_path
):
    """Run Multiple Models (MMs) experiment."""
    the_final_merged_list = []
    
    for collection in results_main:
        # Prepare dataframes based on model type
        if model_name == config.ML_MODEL_POLYNOMIAL_X2:
            dataframe_training = dataframe_for_training_polynomial_x2_SSL(
                results_centralized, results_main, results_cori_list, collection
            )
            dataframe_predict = dataframe_for_predictions_polynomial_x2_SSL(
                results_centralized, results_main, results_cori_list, collection
            )
        elif model_name == config.ML_MODEL_POLYNOMIAL_X3:
            dataframe_training = dataframe_for_training_polynomial_x3_SSL(
                results_centralized, results_main, results_cori_list, collection
            )
            dataframe_predict = dataframe_for_predictions_polynomial_x3_SSL(
                results_centralized, results_main, results_cori_list, collection
            )
        else:
            dataframe_training = dataframe_for_training_SSL(
                results_centralized, results_main, results_cori_list, collection
            )
            dataframe_predict = dataframe_for_predictions_SSL(
                results_centralized, results_main, results_cori_list, collection
            )
        
        # Skip if no training data
        if len(dataframe_training) == 0:
            continue
        
        # Create and train model
        if model_name == config.ML_MODEL_RANDOM_FOREST:
            model = create_random_forest()
        elif model_name == config.ML_MODEL_DECISION_TREE:
            model = create_decision_tree()
        elif model_name == config.ML_MODEL_SVR:
            model = create_svr()
        elif model_name == config.ML_MODEL_LINEAR_REGRESSION:
            model = create_linear_regression()
        elif model_name == config.ML_MODEL_POLYNOMIAL_X2:
            model = create_polynomial_x2()
        elif model_name == config.ML_MODEL_POLYNOMIAL_X3:
            model = create_polynomial_x3()
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        model = train_ml_model(model, df=dataframe_training)
        predictions = model.predict(dataframe_predict)
        the_final_merged_list = final_merged_list_for_SSL(
            dataframe_training, dataframe_predict, predictions, the_final_merged_list
        )
    
    the_final_merged_list = remove_duplicates(the_final_merged_list)
    the_final_merged_list.sort(reverse=True)
    return the_final_merged_list[0:config.FINAL_RESULTS]


def run_gm_experiment(
    topic_id: str,
    topic_text: str,
    model_name: str,
    environment,
    results_centralized,
    results_main,
    results_cori_list,
    index_reader_sample,
    avg_cw,
    collection_index_path
):
    """Run Global Models (GMs) experiment."""
    # Prepare dataframes
    dataframe_training = dataframe_for_training(
        results_centralized, results_main, results_cori_list
    )
    dataframe_predictions = dataframe_for_predictions(
        results_centralized, results_main, results_cori_list
    )
    
    # Skip if no training data
    if len(dataframe_training) == 0:
        return []
    
    # Create and train model
    if model_name == config.ML_MODEL_RANDOM_FOREST:
        model = create_random_forest()
        model = train_ml_model(model, df=dataframe_training)
        predictions = model.predict(dataframe_predictions)
        return final_merged_list_for_simple_machine_learning(
            dataframe_training, dataframe_predictions, predictions
        )
    elif model_name == config.ML_MODEL_DECISION_TREE:
        model = create_decision_tree()
        model = train_ml_model(model, df=dataframe_training)
        predictions = model.predict(dataframe_predictions)
        return final_merged_list_for_simple_machine_learning(
            dataframe_training, dataframe_predictions, predictions
        )
    elif model_name == config.ML_MODEL_SVR:
        model = create_svr()
        model = train_ml_model(model, df=dataframe_training)
        predictions = model.predict(dataframe_predictions)
        return final_merged_list_for_simple_machine_learning(
            dataframe_training, dataframe_predictions, predictions
        )
    elif model_name == config.ML_MODEL_LINEAR_REGRESSION:
        model = create_linear_regression()
        model = train_ml_model(model, df=dataframe_training)
        predictions = model.predict(dataframe_predictions)
        return final_merged_list_for_simple_machine_learning(
            dataframe_training, dataframe_predictions, predictions
        )
    elif model_name == config.ML_MODEL_DNN:
        input_shape = len(dataframe_training.columns) - 1  # Exclude target
        model = create_dnn(input_shape)
        model = train_dnn(model, df=dataframe_training)
        predictions = model.predict(dataframe_predictions.values.astype('float32'))
        # DNN returns 2D array, flatten to 1D
        predictions = predictions.flatten()
        return final_merged_list_for_simple_machine_learning(
            dataframe_training, dataframe_predictions, predictions
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run MLRM experiments')
    parser.add_argument('--environment', type=str, required=True,
                       choices=[config.ENV_COOPERATIVE, config.ENV_UNCOOPERATIVE, config.ENV_UNCOOPERATIVE_WEIGHTED],
                       help='Environment type')
    parser.add_argument('--method', type=str, required=True,
                       help='Method: baseline name or model_type:model_name (e.g., MMs:random_forest)')
    parser.add_argument('--scenario', type=str, default=config.SCENARIO_REALISTIC,
                       choices=[config.SCENARIO_REALISTIC, config.SCENARIO_OPTIMAL, config.SCENARIO_RANDOM],
                       help='Scenario type')
    parser.add_argument('--output', type=str, required=True,
                       help='Output file path for results')
    
    args = parser.parse_args()
    
    # Load topics
    topics = load_topics(config.TOPICS_PATH)
    
    # Initialize indices
    if args.environment == config.ENV_COOPERATIVE:
        index_path = config.COLLECTION_INDEX
    else:
        index_path = config.REPRESENTATIONS_INDEX
    
    index_reader_sample = {}
    for coll in os.listdir(index_path):
        try:
            index_reader_sample[coll] = index.IndexReader(f"{index_path}/{coll}/")
        except:
            continue
    
    avg_cw = get_avg_cw(index_reader_sample)
    searcher_centralized = SimpleSearcher(config.CENTRALIZED_INDEX)
    
    # Initialize environment
    if args.environment == config.ENV_COOPERATIVE:
        env = CooperativeEnvironment(config.COLLECTION_INDEX)
    elif args.environment == config.ENV_UNCOOPERATIVE:
        env = UncooperativeEnvironment(config.COLLECTION_INDEX)
    else:
        env = UncooperativeWeightedEnvironment(config.COLLECTION_INDEX)
    
    # Open output file
    with open(args.output, 'w', encoding='utf-8') as writer:
        counter = 1
        
        for topic in topics:
            topic_id, topic_text = parse_topic(topic)
            
            # Source selection
            results_cori_list = CORI2(
                topic=topic_text,
                my_index=index_reader_sample,
                avg_cw=avg_cw
            )
            results_cori_list = results_cori_list[0:config.NUM_COLLECTIONS_TO_QUERY]
            
            # Search centralized index
            results_centralized = searcher_centralized.search(
                topic_text, config.CENTRALIZED_RESULTS
            )
            
            # Get results from collections
            results_main = {}
            for relevant_collection in results_cori_list:
                collection_name = relevant_collection[1]
                results = env.get_results_from_collection(
                    collection_name, topic_text
                )
                
                # Assign scores based on environment
                if args.environment == config.ENV_COOPERATIVE:
                    results = env.assign_scores(results)
                elif args.environment == config.ENV_UNCOOPERATIVE:
                    results = env.assign_scores(results)
                else:  # weighted
                    cori_score = relevant_collection[0]
                    results = env.assign_scores(results, collection_name, cori_score)
                
                results_main[collection_name] = results
            
            # Run experiment based on method
            try:
                if args.method == config.BASELINE_CORI:
                    final_list = merge_cori(
                        topic_text, results_main, index_reader_sample, avg_cw, config.COLLECTION_INDEX
                    )
                elif args.method == config.BASELINE_SSL:
                    final_list = merge_ssl(
                        topic_text, results_centralized, results_main, index_reader_sample, avg_cw, config.COLLECTION_INDEX
                    )
                elif args.method == config.BASELINE_SAFE:
                    # For SAFE, we need full index readers
                    index_reader_full = {}
                    for coll in os.listdir(config.COLLECTION_INDEX):
                        try:
                            index_reader_full[coll] = index.IndexReader(f"{config.COLLECTION_INDEX}/{coll}/")
                        except:
                            continue
                    final_list = merge_safe(
                        topic_text, results_centralized, results_main, index_reader_sample,
                        index_reader_full, avg_cw, config.COLLECTION_INDEX
                    )
                elif args.method == config.BASELINE_CENTRALIZED:
                    final_list = merge_centralized(results_centralized)
                elif args.method == config.BASELINE_RANDOM:
                    final_list = merge_random(results_main)
                elif ':' in args.method:
                    # ML method: model_type:model_name
                    model_type, model_name = args.method.split(':')
                    if model_type == config.MODEL_TYPE_MM:
                        final_list = run_mm_experiment(
                            topic_id, topic_text, model_name, env, results_centralized,
                            results_main, results_cori_list, index_reader_sample, avg_cw, config.COLLECTION_INDEX
                        )
                    elif model_type == config.MODEL_TYPE_GM:
                        final_list = run_gm_experiment(
                            topic_id, topic_text, model_name, env, results_centralized,
                            results_main, results_cori_list, index_reader_sample, avg_cw, config.COLLECTION_INDEX
                        )
                    else:
                        raise ValueError(f"Unknown model type: {model_type}")
                else:
                    raise ValueError(f"Unknown method: {args.method}")
                
                # Write results
                for ind, (score, docid) in enumerate(final_list, start=1):
                    writer.write(f"{topic_id} Q0 {docid} {ind} {score} {args.method}\n")
                
                print(f"Processed topic {counter}: {topic_id}")
                counter += 1
                
            except Exception as e:
                print(f"Error processing topic {topic_id}: {e}")
                # Fallback to CORI
                try:
                    final_list = merge_cori(
                        topic_text, results_main, index_reader_sample, avg_cw, config.COLLECTION_INDEX
                    )
                    for ind, (score, docid) in enumerate(final_list, start=1):
                        writer.write(f"{topic_id} Q0 {docid} {ind} {score} CORI_FALLBACK\n")
                except:
                    pass
                counter += 1


if __name__ == '__main__':
    main()
