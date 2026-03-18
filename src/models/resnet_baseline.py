from .base import BaseECGModel
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1):
        super().__init__()

        self.main_path = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2),
            nn.BatchNorm1d(out_channels),
            nn.SiLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(out_channels),
        )

        if in_channels != out_channels or stride != 1:
            self.skip = nn.Conv1d(in_channels, out_channels, 1, stride=stride)
        else:
            self.skip = nn.Identity()
        
    def forward(self, x):
        return F.silu(self.main_path(x) + self.skip(x))
    
class ResNetBaseline(BaseECGModel):
    def __init__(self, config):
        super().__init__(config)
        
        channels = self.params_cfg.get('resnet_channels', [32, 64, 128, 256])
        kernel_size = self.params_cfg.get('kernel_size', 7)
        dropout = self.params_cfg.get('dropout', 0.3)

        self.blocks = nn.ModuleList([
            ResNetBlock(self.num_leads, channels[0], kernel_size),
            *[ResNetBlock(channels[i], channels[i+1], kernel_size, stride=2) for i in range(len(channels) - 1)],
        ])

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)

        self.task_heads = nn.ModuleDict({
            task : nn.Sequential(
                nn.Linear(channels[-1], 1),
            )
            for task in self.tasks
        })

    def forward(self, x, **kwargs):
        x = self.get_embeddings(x)

        return {
            task: head(x)
            for task, head in self.task_heads.items()
        }
    
    def get_embeddings(self, x, **kwargs):
        for block in self.blocks:
            x = block(x)

        x = self.pool(x).squeeze(-1)
        return self.dropout(x)