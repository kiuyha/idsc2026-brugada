from .resnet import ResNetBaseline
from .spatial_gcn import SpatialGCNModel
# from .temporal_gin import TemporalGINModel  # Future addition

MODEL_REGISTRY = {
    'resnet_baseline': ResNetBaseline,
    'spatial_gcn': SpatialGCNModel,
    # 'temporal_gin': TemporalGINModel,
}

def build_model(config):
    model_type = config.model.type
    
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(MODEL_REGISTRY.keys())}")
    
    model_class = MODEL_REGISTRY[model_type]
    return model_class(config)