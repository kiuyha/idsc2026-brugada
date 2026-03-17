import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import json
from itertools import product
import argparse
from utils import load_config

SEARCH_SPACES = {
    'resnet_baseline': {
        'learning_rate': [0.0001, 0.0003, 0.0005, 0.001],
        'weight_decay': [0.0001, 0.001, 0.01],
        'dropout': [0.3, 0.4, 0.5],
        'resnet_channels': [
            [16, 32, 64],
            [32, 64, 128],
            [32, 64, 128, 256]
        ],
    },
    
    'spatial_gnn': {
        'learning_rate': [0.0001, 0.0003, 0.0005],
        'weight_decay': [0.001, 0.01, 0.02],
        'dropout': [0.3, 0.4, 0.5],
        'hidden_dim': [32, 64, 128],
        'num_gnn_layers': [2, 3, 4],
        'correlation_threshold': [0.2, 0.3, 0.4],
    },
    
    'temporal_gnn': {
        'learning_rate': [0.0003, 0.0005, 0.001],
        'weight_decay': [0.001, 0.01],
        'dropout': [0.3, 0.4, 0.5],
        'hidden_dim': [32, 64, 128],
        'num_gnn_layers': [2, 3, 4],
        'pooling': ['mean', 'max', 'attention'],
    }
}


def create_config_variant(base_config_path, params, variant_id):
    config = load_config(base_config_path)
    
    for key, value in params.items():
        if key in ['learning_rate', 'weight_decay']:
            config['training'][key] = float(value)
        elif key in ['dropout']:
            config['model']['params'][key] = float(value)
        elif key in ['resnet_channels', 'hidden_dim', 'num_gnn_layers', 
                     'correlation_threshold', 'pooling']:
            config['model']['params'][key] = value
        else:
            if key in ['correlation_threshold']:
                config['data'][key] = float(value)
    
    model_type = config['model']['type']
    config['experiment_name'] = f"{model_type}_variant_{variant_id}"
    
    variant_path = f"configs/variants/{model_type}_variant_{variant_id}.yml"
    Path("configs/variants").mkdir(parents=True, exist_ok=True)
    
    with open(variant_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return variant_path, config['experiment_name']


def run_hyperparameter_search(model_type, search_type='grid', n_random=20, max_trials=None):
    print(f"\n{'='*80}")
    print(f"Hyperparameter Search: {model_type}")
    print(f"Search Type: {search_type}")
    print(f"{'='*80}\n")
    
    search_space = SEARCH_SPACES[model_type]
    base_config_path = f"configs/{model_type}.yml"
    
    base_config = load_config(base_config_path)
    primary_metric = base_config.get('evaluation', {}).get('primary_metric', 'f2')
    
    if search_type == 'grid':
        param_names = list(search_space.keys())
        param_values = list(search_space.values())
        
        all_combinations = list(product(*param_values))
        param_combinations = [
            dict(zip(param_names, combo))
            for combo in all_combinations
        ]
        
        print(f"Grid search: {len(param_combinations)} total combinations")
        
    elif search_type == 'random':
        param_combinations = []
        for _ in range(n_random):
            params = {}
            for param_name, param_values in search_space.items():
                params[param_name] = np.random.choice(param_values)
            param_combinations.append(params)
        
        print(f"Random search: {n_random} random samples")
    
    else:
        raise ValueError(f"Unknown search_type: {search_type}")
    
    if max_trials and len(param_combinations) > max_trials:
        print(f"Limiting to {max_trials} trials")
        if search_type == 'grid':
            indices = np.linspace(0, len(param_combinations)-1, max_trials, dtype=int)
            param_combinations = [param_combinations[i] for i in indices]
        else:
            param_combinations = param_combinations[:max_trials]
    
    results = []
    for i, params in enumerate(param_combinations):
        print(f"\n--- Trial {i+1}/{len(param_combinations)} ---")
        print(f"Parameters: {json.dumps(params, indent=2)}")
        
        config_path, exp_name = create_config_variant(base_config_path, params, i)
        
        result = subprocess.run(
            ['python', 'main.py', '--config', config_path],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"Error in trial {i+1}")
            print(result.stderr)
            continue
        
        results_path = f"experiments/{exp_name}.csv"
        if Path(results_path).exists():
            trial_results = pd.read_csv(results_path).iloc[0].to_dict()
            trial_results.update(params)
            trial_results['trial_id'] = i
            trial_results['experiment_name'] = exp_name
            results.append(trial_results)
            
            metric_value = trial_results.get(primary_metric, 0)
            print(f"Result - {primary_metric}: {metric_value:.4f} | Acc: {trial_results['accuracy']:.4f}")
        else:
            print(f"Results not found: {results_path}")
    
    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(primary_metric, ascending=False)
        
        output_path = f"experiments/hyperparam_search_{model_type}_{search_type}.csv"
        results_df.to_csv(output_path, index=False)
        
        print(f"\n{'='*80}")
        print("HYPERPARAMETER SEARCH RESULTS")
        print(f"{'='*80}")
        
        print(f"\nTop 5 Configurations (sorted by {primary_metric}):")
        top5 = results_df.head(5)
        
        for idx, row in enumerate(top5.itertuples(), 1):
            print(f"\nRank {idx}:")
            print(f"  {primary_metric.upper()}: {getattr(row, primary_metric):.4f} | Accuracy: {row.accuracy:.4f}")
            print(f"  Learning Rate: {row.learning_rate}")
            print(f"  Weight Decay: {row.weight_decay}")
            print(f"  Dropout: {row.dropout}")
            if hasattr(row, 'hidden_dim'):
                print(f"  Hidden Dim: {row.hidden_dim}")
            if hasattr(row, 'num_gnn_layers'):
                print(f"  GNN Layers: {row.num_gnn_layers}")
        
        print(f"\nFull results saved to: {output_path}")
        
        best_params = results_df.iloc[0]
        best_config_path = f"configs/best_{model_type}.yml"
        
        best_config = load_config(base_config_path)
        
        for param_name in search_space.keys():
            if param_name in best_params:
                value = best_params[param_name]
                
                if param_name in ['learning_rate', 'weight_decay']:
                    best_config['training'][param_name] = float(value)
                elif param_name in ['dropout']:
                    best_config['model']['params'][param_name] = float(value)
                else:
                    best_config['model']['params'][param_name] = value
        
        best_config['experiment_name'] = f"best_{model_type}"
        
        with open(best_config_path, 'w') as f:
            yaml.dump(best_config, f, default_flow_style=False)
        
        print(f"Best config saved to: {best_config_path}")
        
    else:
        print("No successful trials")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter search")
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['resnet_baseline', 'spatial_gnn', 'temporal_gnn'],
        help='Model type to tune'
    )
    parser.add_argument(
        '--search',
        type=str,
        default='random',
        choices=['grid', 'random'],
        help='Search strategy'
    )
    parser.add_argument(
        '--n_random',
        type=int,
        default=20,
        help='Number of random samples (for random search)'
    )
    parser.add_argument(
        '--max_trials',
        type=int,
        default=None,
        help='Maximum number of trials'
    )
    
    args = parser.parse_args()
    
    run_hyperparameter_search(
        model_type=args.model,
        search_type=args.search,
        n_random=args.n_random,
        max_trials=args.max_trials
    )