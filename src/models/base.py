import torch.nn as nn
from abc import ABC, abstractmethod

class BaseECGModel(ABC, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.params_cfg = config['model'].get('params', {})
        self.tasks = [t for t, params in config['tasks'].items() if params['enabled']]
        self.num_leads = config['data']['leads']

    @abstractmethod
    def forward(self, x, **kwargs):
        """
        Forward pass
        
        Args:
            x: Input tensor [batch, 12, time_samples]
            **kwargs: Model-specific arguments (e.g., edge_index for GNN)
        
        Returns:
            predictions: [batch, 1] probabilities
        """
        pass
    
    def get_embeddings(self, x, **kwargs):
        raise NotImplementedError(f"{self.__class__.__name__} doesn't implement get_embeddings()")
    
    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())