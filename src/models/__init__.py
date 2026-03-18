from .resnet_baseline import ResNetBaseline
from .spatial_gnn import SpatialGNN
from .temporal_gnn import TemporalGNN
from .hgb_baseline import HistBoostBaseline

MODEL_REGISTRY = {
    'hgb_baseline':HistBoostBaseline,
    'resnet_baseline': ResNetBaseline,
    'spatial_gnn': SpatialGNN,
    'temporal_gnn': TemporalGNN,
}

def build_model(config):
    model_type = config['model']['type']
    
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(MODEL_REGISTRY.keys())}")
    
    model_class = MODEL_REGISTRY[model_type]
    return model_class(config)