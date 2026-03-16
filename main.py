import argparse
import yaml
from pathlib import Path
from types import SimpleNamespace

from src.data_loader import get_dataloaders
from src.models import build_model
from src.trainer import MultiTaskTrainer

def load_config(config_path):
    """Load YAML config with inheritance (_base_ support)"""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Handle inheritance
    if '_base_' in config:
        base_config = load_config(Path(config_path).parent / config['_base_'])
        base_config.update(config)
        config = base_config
    
    # Convert to nested namespace for dot notation
    return dict_to_namespace(config)

def dict_to_namespace(d):
    """Recursively convert dict to SimpleNamespace"""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(item) for item in d]
    else:
        return d

def main(args):
    # Load configuration
    config = load_config(args.config)
    print(f"Running experiment: {config.experiment_name}")
    print(f"Model: {config.model.type}")
    
    # Data
    train_loader, val_loader = get_dataloaders(config)
    
    # Model
    model = build_model(config)
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Training
    trainer = MultiTaskTrainer(model, config, train_loader, val_loader)
    trainer.fit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file (e.g., configs/spatial_gcn.yaml)')
    args = parser.parse_args()
    
    main(args)