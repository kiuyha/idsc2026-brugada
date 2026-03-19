import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
import json
import ast
from itertools import product
import argparse
from src.utils import load_config, set_seed, save_config

SEARCH_SPACES = {
    'hgb_baseline': {
        'max_depth': [3, 4, 5],
        'min_samples_leaf': [20, 30, 40],
        'max_iter': [50, 100, 200],
        'class_weight': ['balanced', None],
        'enable_basal_pattern': [True, False],
        'enable_sudden_death': [True, False],
    },
    'resnet_baseline': {
        'learning_rate': [0.0008, 0.001, 0.0012, 0.0015],
        'weight_decay': [0.0005, 0.001, 0.005, 0.01],
        'dropout': [0.4, 0.5, 0.6],
        'resnet_channels': [
            [32, 64, 128],
            [32, 64, 128, 256],
            [48, 96, 192],
        ],
        'enable_basal_pattern': [True, False],
        'enable_sudden_death': [True, False],
    },
    'spatial_gnn': {
        'learning_rate': [0.001, 0.0012, 0.0015],
        'weight_decay': [0.005, 0.01, 0.015],
        'dropout': [0.4, 0.5, 0.6],
        'hidden_dim': [32, 64, 96],
        'num_gnn_layers': [2, 3],
        'gnn_type': ['gcn', 'gat'],
        'correlation_threshold': [0.5, 0.6],
        'anatomic_weight': [0.7, 0.8],
        'enable_basal_pattern': [True, False],
        'enable_sudden_death': [True, False],
    },
}

HGB_MODEL_PARAMS = {'max_depth', 'min_samples_leaf', 'max_iter', 'class_weight'}

TEST_SEEDS = [42, 43, 44]

def _apply_param_to_config(config, key, value):
    if key in ['learning_rate', 'weight_decay']:
        config['training'][key] = float(value)

    elif key == 'dropout':
        config['model']['params'][key] = float(value)

    elif key in ['correlation_threshold', 'anatomic_weight']:
        config['data'][key] = float(value)

    elif key in HGB_MODEL_PARAMS:
        config['model']['params'][key] = value

    elif key in ['resnet_channels', 'hidden_dim', 'num_gnn_layers', 'gnn_type']:
        config['model']['params'][key] = value

    elif key == 'enable_basal_pattern':
        if 'basal_pattern' in config.get('tasks', {}):
            config['tasks']['basal_pattern']['enabled'] = bool(value)

    elif key == 'enable_sudden_death':
        if 'sudden_death' in config.get('tasks', {}):
            config['tasks']['sudden_death']['enabled'] = bool(value)

def create_config_variant(base_config_path, params, variant_id, seed):
    config = load_config(base_config_path)
    config['seed'] = seed

    for key, value in params.items():
        _apply_param_to_config(config, key, value)

    model_type = config['model']['type']
    config['experiment_name'] = f"{model_type}_variant_{variant_id}_seed_{seed}"

    variant_path = f"configs/variants/{model_type}_variant_{variant_id}_seed_{seed}.yml"
    save_config(config, variant_path)

    return variant_path, config['experiment_name']

def _safe_parse_value(key, value):
    if key == 'resnet_channels' and isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    if key in HGB_MODEL_PARAMS | {'hidden_dim', 'num_gnn_layers'}:
        # Numeric columns may survive as-is, but int columns could become floats
        if key in {'max_depth', 'min_samples_leaf', 'max_iter',
                   'hidden_dim', 'num_gnn_layers'}:
            try:
                if pd.isna(value) or value == 'None' or value is None:
                    return None
                return int(value)
            except (ValueError, TypeError):
                return value
    if key in {'enable_basal_pattern', 'enable_sudden_death'}:
        if isinstance(value, str):
            return value.strip().lower() == 'true'
        return bool(value)
    return value

def _print_top_results(results_df, primary_metric, acc_metric, search_space_keys, top_n=5):
    mean_primary_col = f'{primary_metric}_mean'
    print(f"\nTop {top_n} Configurations (sorted by {mean_primary_col}):")
    top_rows = results_df.head(top_n)

    for rank, (_, row) in enumerate(top_rows.iterrows(), 1):
        print(f"\nRank {rank}:")
        print(f"  {primary_metric.upper()}: {row[mean_primary_col]:.4f} ± {row[f'{primary_metric}_std']:.4f}")
        print(f"  Accuracy: {row[f'{acc_metric}_mean']:.4f} ± {row[f'{acc_metric}_std']:.4f}")
        # Print every hyperparameter that exists in the search space
        for key in search_space_keys:
            if key in row:
                val = row[key]
                is_missing = False
                if isinstance(val, (list, tuple, np.ndarray)):
                    is_missing = False
                else:
                    is_missing = pd.isna(val)
                
                if not is_missing:
                    print(f"  {key}: {val}")

def run_hyperparameter_search(model_type, search_type='grid', n_random=20, max_trials=None):
    print(f"Hyperparameter Search: {model_type}")
    print(f"Search Type: {search_type}")

    search_space = SEARCH_SPACES[model_type]
    base_config_path = f"configs/{model_type}.yml"
    base_config = load_config(base_config_path)

    # Seed once so random sampling is reproducible
    set_seed(base_config.get('seed', 42))

    primary_task = list(base_config['tasks'].keys())[0]
    base_metric = base_config.get('evaluation', {}).get('primary_metric', 'f2')
    primary_metric = f"{primary_task}_{base_metric}"
    acc_metric = f"{primary_task}_accuracy"

    if search_type == 'grid':
        param_names = list(search_space.keys())
        param_values = list(search_space.values())
        all_combinations = list(product(*param_values))
        param_combinations = [
            dict(zip(param_names, combo)) for combo in all_combinations
        ]
        print(f"Grid search: {len(param_combinations)} total combinations")

    elif search_type == 'random':
        param_combinations = []
        for _ in range(n_random):
            params = {
                name: vals[np.random.randint(len(vals))]
                for name, vals in search_space.items()
            }
            param_combinations.append(params)
        print(f"Random search: {n_random} random samples")

    else:
        raise ValueError(f"Unknown search_type: {search_type}")

    if max_trials and len(param_combinations) > max_trials:
        print(f"Limiting to {max_trials} trials")
        if search_type == 'grid':
            indices = np.linspace(0, len(param_combinations) - 1, max_trials, dtype=int)
            param_combinations = [param_combinations[i] for i in indices]
        else:
            param_combinations = param_combinations[:max_trials]

    results = []

    for i, params in enumerate(param_combinations):
        print(f"\n--- Trial {i + 1}/{len(param_combinations)} ---")
        print(f"Parameters: {json.dumps(params, indent=2, default=str)}")

        trial_metrics = []
        trial_accs = []
        valid_run = True

        for seed in TEST_SEEDS:
            print(f"  Running Seed {seed}...")
            config_path, exp_name = create_config_variant(
                base_config_path, params, i, seed
            )

            result = subprocess.run(
                ['python', 'main.py', '--config', config_path],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                print(f"  Error in trial {i + 1}, seed {seed}")
                print(result.stderr)
                valid_run = False
                break

            results_path = f"experiments/{exp_name}.csv"
            if Path(results_path).exists():
                trial_results = pd.read_csv(results_path).iloc[0].to_dict()
                trial_metrics.append(trial_results.get(primary_metric, 0))
                trial_accs.append(trial_results.get(acc_metric, 0))
            else:
                print(f"  Results not found: {results_path}")
                valid_run = False
                break

        if valid_run and len(trial_metrics) == len(TEST_SEEDS):
            mean_metric = np.mean(trial_metrics)
            std_metric = np.std(trial_metrics)
            mean_acc = np.mean(trial_accs)
            std_acc = np.std(trial_accs)

            print(
                f"Result — {primary_metric}: {mean_metric:.4f} ± {std_metric:.4f} "
                f"| Acc: {mean_acc:.4f} ± {std_acc:.4f}"
            )

            agg = params.copy()
            agg['trial_id'] = i
            agg[f'{primary_metric}_mean'] = mean_metric
            agg[f'{primary_metric}_std'] = std_metric
            agg[f'{acc_metric}_mean'] = mean_acc
            agg[f'{acc_metric}_std'] = std_acc
            results.append(agg)

    if not results:
        print("No successful trials.")
        return

    results_df = pd.DataFrame(results)
    mean_primary_col = f'{primary_metric}_mean'
    results_df = results_df.sort_values(mean_primary_col, ascending=False)

    output_path = f"experiments/hyperparam_search_{model_type}_{search_type}.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nHYPERPARAMETER SEARCH RESULTS (3-Seed Average)")
    _print_top_results(
        results_df, primary_metric, acc_metric,
        search_space_keys=list(search_space.keys())
    )
    print(f"\nFull results saved to: {output_path}")

    best_row = results_df.iloc[0]
    best_config = load_config(base_config_path)

    for param_name in search_space.keys():
        val = best_row.get(param_name)
        is_missing = False
        if val is None:
            is_missing = True
        elif not isinstance(val, (list, tuple, np.ndarray)):
            try:
                is_missing = pd.isna(val)
            except (TypeError, ValueError):
                is_missing = False
        if param_name in best_row and not is_missing:
            value = _safe_parse_value(param_name, best_row[param_name])
            _apply_param_to_config(best_config, param_name, value)

    best_config['experiment_name'] = model_type
    best_config_path = f"configs/best/{model_type}.yml"
    Path("configs/best").mkdir(parents=True, exist_ok=True)
    save_config(best_config, best_config_path)
    print(f"Best config saved to: {best_config_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter search")
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=list(SEARCH_SPACES.keys()),
        help='Model type to tune',
    )
    parser.add_argument(
        '--search',
        type=str,
        default='random',
        choices=['grid', 'random'],
        help='Search strategy',
    )
    parser.add_argument(
        '--n_random',
        type=int,
        default=20,
        help='Number of random samples (for random search)',
    )
    parser.add_argument(
        '--max_trials',
        type=int,
        default=None,
        help='Maximum number of trials',
    )

    args = parser.parse_args()
    run_hyperparameter_search(
        model_type=args.model,
        search_type=args.search,
        n_random=args.n_random,
        max_trials=args.max_trials,
    )