import argparse
from src.utils import load_config, set_seed
from src.data_loader import get_dataloaders
from src.models import build_model
from src.trainer import Trainer, TraditionalTrainer
from src.interpretability import generate_xai_from_dataset
import pandas as pd

def main(args):
    config = load_config(args.config)

    if args.override:
        for override in args.override:
            key, value = override.split('=')
            keys = key.split('.')
            
            target = config
            for k in keys[:-1]:
                target = target[k]
            
            final_key = keys[-1]
            try:
                if value.lower() == 'true':
                    target[final_key] = True
                elif value.lower() == 'false':
                    target[final_key] = False
                elif '.' in value or 'e' in value.lower():
                    target[final_key] = float(value)
                elif value.isdigit():
                    target[final_key] = int(value)
                else:
                    target[final_key] = value
            except:
                target[final_key] = value
    
    model_type = config['model']['type']
    print(f"Experiment: {config['experiment_name']}")
    print(f"Model: {model_type}")
    print(f"Device: {config['device']}")
    print(f"Seed: {config['seed']}")
    
    set_seed(config['seed'])
    
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(config)
    print(f"Train: {len(train_loader.dataset)} samples")
    print(f"Val: {len(val_loader.dataset)} samples")
    print(f"Test: {len(test_loader.dataset)} samples\n")
    
    print("Building model...")
    model = build_model(config)
    print(f"Model parameters: {model.num_parameters:,}\n")

    trainer = Trainer if model_type != 'hgb_baseline' else TraditionalTrainer
    trainer = trainer(model, config, train_loader, val_loader, test_loader)
    trainer.train()
    test_result = trainer.testing()
    
    flat_results = {}
    for task, metrics in test_result.items():
        for metric, value in metrics.items():
            flat_results[f"{task}_{metric}"] = value

    pd.DataFrame([flat_results]).to_csv(f"experiments/{config['experiment_name']}.csv", index=False)
    
    if model_type != 'hgb_baseline':  # Skip XAI for traditional ML if it's unsupported
        try:
            # We use the test_loader's dataset so it doesn't leak training data
            generate_xai_from_dataset(
                model=trainer.model, 
                dataset=test_loader.dataset, 
                config=config, 
                args=args
            )
        except Exception as e:
            print(f"Failed to generate XAI report: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Brugada detection model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default config
  python main.py
  
  # Specify config
  python main.py --config configs/spatial_gnn.yml
  
  # Override parameters
  python main.py --config configs/spatial_gnn.yml --override training.learning_rate=0.0001
  
  # Multiple overrides
  python main.py --config configs/resnet_baseline.yml \\
      --device cuda --seed 123 \\
      --override training.epochs=200 data.batch_size=8
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default='configs/base.yml',
        help='Path to config file (default: configs/base.yml)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (overrides config): cpu, cuda, cuda:0, auto, etc'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed (overrides config)'
    )
    parser.add_argument(
        '--override',
        type=str,
        nargs='+',
        default=[],
        help='Override config values (e.g., training.learning_rate=0.001 data.batch_size=32)'
    )
    parser.add_argument(
        '--patient_id',
        type=str,
        default=None,
        help="Select specific patient ID (e.g. 1)"
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=3,
        help="Use the top k of lead according to importance (default: 3)"
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help="Output path for the plot image"
    )

    args = parser.parse_args()
    
    main(args)