import yaml
import random
import numpy as np
import torch
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F

def load_config(config_path):
    config_path = Path(config_path)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Handle inheritance (_base_ key)
    if '_base_' in config:
        base_path = config_path.parent / config['_base_']
        base_config = load_config(base_path)

        base_config = deep_update(base_config, config)
        config = base_config
        config.pop('_base_', None)
    
    return config

def deep_update(base_dict, update_dict):
    for key, value in update_dict.items():
        if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
            base_dict[key] = deep_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make CUDA deterministic
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device(config):
    device_str = config.get('device', 'auto')
    
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
    
    print(f"Using device: {device}")
    return device

def save_config(config, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.79, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
        p_t = torch.exp(-bce_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        
class MultiTaskLoss(nn.Module):
    def __init__(self, task_weights, loss_type, loss_params=None):
        super().__init__()
        self.task_weights = task_weights
        loss_params = loss_params or {}
        
        self.task_losses = nn.ModuleDict()
        for task in task_weights.keys():
            if loss_type == 'focal':
                self.task_losses[task] = FocalLoss(**loss_params)
            elif loss_type in ['bce', 'weighted_bce']:
                self.task_losses[task] = nn.BCEWithLogitsLoss(**loss_params)
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(self, predictions, targets):
        total_loss = 0
        task_losses_dict = {}
        
        for task in self.task_weights.keys():
            if task in predictions and task in targets:
                task_loss = self.task_losses[task](predictions[task], targets[task].float())
                weighted_loss = self.task_weights[task] * task_loss
                total_loss += weighted_loss
                task_losses_dict[task] = task_loss.item()
        
        return total_loss, task_losses_dict