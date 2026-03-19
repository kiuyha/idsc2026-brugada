import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from src.utils import load_config, save_config

CONFIGS = [
    'configs/best/hgb_baseline.yml',
    'configs/best/resnet_baseline.yml',
    'configs/best/spatial_gnn.yml',
]

TEST_SEEDS = [42, 43, 44, 45, 46] 

def run_experiment_with_seed(base_config_path, seed):
    config = load_config(base_config_path)
    model_type = config['model']['type']
    
    config['seed'] = seed
    exp_name = f"{model_type}_eval_seed_{seed}"
    config['experiment_name'] = exp_name
    
    temp_config_path = f"configs/variants/temp_{exp_name}.yml"
    save_config(config, temp_config_path)
    
    result = subprocess.run(
        ['python', 'main.py', '--config', temp_config_path],
        capture_output=True,
        text=True
    )
    
    Path(temp_config_path).unlink(missing_ok=True)
    
    if result.returncode != 0:
        print(f"Error running {model_type} with seed {seed}")
        print(result.stderr)
        return None
    
    results_path = f"experiments/{exp_name}.csv"
    if Path(results_path).exists():
        results = pd.read_csv(results_path)
        return results.iloc[0].to_dict()
    else:
        print(f"Results file not found: {results_path}")
        return None


def main():
    print("Starting Model Comparison")
    print(f"Will run {len(CONFIGS)} models across {len(TEST_SEEDS)} seeds each.\n")
    
    base_config = load_config('configs/base.yml')
    primary_task = list(base_config['tasks'].keys())[0]
    
    metrics_to_track = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'f2']
    tracked_cols = [f"{primary_task}_{m}" for m in metrics_to_track]
    
    all_results = []
    
    for config_path in CONFIGS:
        model_type = load_config(config_path)['model']['type']
        print(f"\nEvaluating: {model_type}")
        
        # Store results for all seeds for this specific model
        model_metrics = {col: [] for col in tracked_cols}
        
        for seed in TEST_SEEDS:
            print(f"  Running seed {seed}...")
            res = run_experiment_with_seed(config_path, seed)
            
            if res:
                for col in tracked_cols:
                    if col in res:
                        model_metrics[col].append(res[col])
        
        agg_result = {'model_type': model_type}
        for col in tracked_cols:
            if model_metrics[col]:
                mean_val = np.mean(model_metrics[col])
                std_val = np.std(model_metrics[col])
                
                agg_result[f"{col}_mean"] = mean_val
                agg_result[f"{col}_std"] = std_val
                
                agg_result[f"{col}_display"] = f"{mean_val:.4f} ± {std_val:.4f}"
        
        all_results.append(agg_result)
    
    print(f"all_results: {all_results}")
    if all_results:
        comparison_df = pd.DataFrame(all_results)
        
        primary_sort_col = f"{primary_task}_f2_mean"
        if primary_sort_col in comparison_df.columns:
            comparison_df = comparison_df.sort_values(primary_sort_col, ascending=False)
        
        display_cols = ['model_type'] + [f"{col}_display" for col in tracked_cols if f"{col}_display" in comparison_df.columns]
        display_df = comparison_df[display_cols]
        
        comparison_path = "experiments/model_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        
        print("\nROBUST MODEL COMPARISON RESULTS")
        print(display_df.to_string(index=False))
        print(f"\nDetailed numerical results saved to: {comparison_path}")
        
    else:
        print("No results to compare")

if __name__ == "__main__":
    main()