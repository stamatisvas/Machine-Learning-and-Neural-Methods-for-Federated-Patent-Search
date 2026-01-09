"""
Example script showing how to run specific experiments.

This script demonstrates how to run the experiments from the paper.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
import subprocess
import sys

# Example: Run MMs Random Forest in cooperative environment
# python scripts/example_run.py

if __name__ == '__main__':
    # Example: Run experiments using subprocess
    # This is a template - modify paths as needed
    
    examples = [
        {
            'name': 'MMs Random Forest (best model) - Cooperative',
            'args': [
                '--environment', config.ENV_COOPERATIVE,
                '--method', f'{config.MODEL_TYPE_MM}:{config.ML_MODEL_RANDOM_FOREST}',
                '--scenario', config.SCENARIO_REALISTIC,
                '--output', str(config.RESULTS_PATH / 'MMs_random_forest_cooperative.res')
            ]
        },
        {
            'name': 'GMs DNN - Cooperative',
            'args': [
                '--environment', config.ENV_COOPERATIVE,
                '--method', f'{config.MODEL_TYPE_GM}:{config.ML_MODEL_DNN}',
                '--scenario', config.SCENARIO_REALISTIC,
                '--output', str(config.RESULTS_PATH / 'GMs_dnn_cooperative.res')
            ]
        },
        {
            'name': 'Baseline SSL - Cooperative',
            'args': [
                '--environment', config.ENV_COOPERATIVE,
                '--method', config.BASELINE_SSL,
                '--scenario', config.SCENARIO_REALISTIC,
                '--output', str(config.RESULTS_PATH / 'SSL_cooperative.res')
            ]
        },
        {
            'name': 'MMs Random Forest - Uncooperative Weighted (best from paper)',
            'args': [
                '--environment', config.ENV_UNCOOPERATIVE_WEIGHTED,
                '--method', f'{config.MODEL_TYPE_MM}:{config.ML_MODEL_RANDOM_FOREST}',
                '--scenario', config.SCENARIO_REALISTIC,
                '--output', str(config.RESULTS_PATH / 'MMs_random_forest_uncooperative_weighted.res')
            ]
        }
    ]
    
    for example in examples:
        print(f"\nRunning: {example['name']}")
        cmd = ['python', 'scripts/run_experiment.py'] + example['args']
        print(f"Command: {' '.join(cmd)}")
        # Uncomment to actually run:
        # subprocess.run(cmd)
