import subprocess
import pandas as pd
from pathlib import Path
from utils import load_config

CONFIGS = [
    'configs/resnet_baseline.yml',
    'configs/spatial_gnn.yml',
    'configs/temporal_gnn.yml',
]

def run_experiment(config_path):
    result = subprocess.run(
        ['python', 'main.py', '--config', config_path],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"Error running {config_path}")
        print(result.stderr)
        return None
    
    config = load_config(config_path)
    exp_name = config['experiment_name']
    
    results_path = f"experiments/{exp_name}.csv"
    if Path(results_path).exists():
        results = pd.read_csv(results_path)
        return {
            'config': config_path,
            'experiment_name': exp_name,
            'model_type': config['model']['type'],
            **results.iloc[0].to_dict()
        }
    else:
        print(f"Results file not found: {results_path}")
        return None


def main():
    print("Starting Model Comparison")
    print(f"Will run {len(CONFIGS)} experiments\n")
    
    base_config = load_config(CONFIGS[0])
    primary_metric = base_config.get('evaluation', {}).get('primary_metric', 'f2')
    
    all_results = []
    
    for config_path in CONFIGS:
        result = run_experiment(config_path)
        if result:
            all_results.append(result)
    
    if all_results:
        comparison_df = pd.DataFrame(all_results)
        
        cols = [
            'experiment_name',
            'model_type',
            primary_metric,
            'accuracy', 
            'precision',
            'recall',
            'f1',
            'auc'
        ]
        cols = [c for c in cols if c in comparison_df.columns]
        comparison_df = comparison_df[cols]
        comparison_df = comparison_df.sort_values(primary_metric, ascending=False)
        
        comparison_path = "experiments/model_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        
        print("MODEL COMPARISON RESULTS")
        print(comparison_df.to_string(index=False))
        print(f"\nSaved to: {comparison_path}")
        
        best_model = comparison_df.iloc[0]
        print(f"\nBest Model: {best_model['experiment_name']}")
        print(f"   {primary_metric.upper()}: {best_model[primary_metric]:.4f}")
        print(f"   Accuracy: {best_model['accuracy']:.4f}")
        if 'auc' in best_model:
            print(f"   AUC: {best_model['auc']:.4f}")
    else:
        print("No results to compare")

if __name__ == "__main__":
    main()