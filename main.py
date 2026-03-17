import argparse
from src.utils import load_config, set_seed
from src.data_loader import get_dataloaders
from src.models import build_model
from src.trainer import Trainer
import pandas as pd

def main(args):
    config = load_config(args.config)
    
    print(f"Experiment: {config['experiment_name']}")
    print(f"Model: {config['model']['type']}")
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
    
    trainer = Trainer(model, config, train_loader, val_loader, test_loader)
    trainer.train()
    test_result = trainer.testing()
    pd.DataFrame(test_result).to_csv(f"results/{config['experiment_name']}.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Brugada detection model")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        default='configs/hybrid_model.yml',
        help='Path to config file (default: configs/hybrid_model.yml)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (overrides config): cpu, cuda, cuda:0, auto'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed (overrides config)'
    )
    args = parser.parse_args()
    
    main(args)