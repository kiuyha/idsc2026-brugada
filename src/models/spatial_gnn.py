from .base import BaseECGModel
from .resnet_baseline import ResNetBlock
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv

class SpatialGNN(BaseECGModel):
    def __init__(self, config):
        super().__init__(config)
        
        channels = self.params_cfg.get('resnet_channels', [32, 64, 128, 256])
        kernel_size = self.params_cfg.get('kernel_size', 7)
        dropout = self.params_cfg.get('dropout', 0.3)
        hidden_dim = self.params_cfg.get('hidden_dim', 64)
        num_gnn_layers = self.params_cfg.get('num_gnn_layers', 3)
        gnn_type = self.params_cfg.get('gnn_type', "gcn")

        self.lead_encoder = nn.ModuleList([
            ResNetBlock(1, channels[0], kernel_size),
            *[ResNetBlock(channels[i], channels[i+1], kernel_size) for i in range(len(channels) - 1)],
        ])
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)

        if gnn_type == "gcn":
            GNN = GCNConv
        elif gnn_type == "gat":
            GNN = GATConv
        elif gnn_type == "gin":
            GNN = GINConv
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")
        
        self.gnns = nn.ModuleList([
            GNN(channels[-1], hidden_dim),
            *[GNN(hidden_dim, hidden_dim) for _ in range(num_gnn_layers - 1)],
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.task_heads = nn.ModuleDict({
            task : nn.Sequential(
                nn.Linear(hidden_dim * 12, 1),
            )
            for task in self.tasks
        })
    
    def forward(self, x, edge_index, edge_weight=None):
        final_embeddings = self.get_embeddings(x, edge_index, edge_weight, layer='final')
        h = final_embeddings.flatten(start_dim=1)
        h = self.dropout(h)
        
        return {
            task: head(h)
            for task, head in self.task_heads.items()
        }
    
    def get_embeddings(self, x, edge_index=None, edge_weight=None, layer='final'):
        batch_size = x.shape[0]
        
        # Extract temporal features using ResNet
        lead_features = []
        for lead_idx in range(self.num_leads):
            lead_signal = x[:, lead_idx:lead_idx+1, :]
            
            feat = lead_signal
            for block in self.lead_encoder:
                feat = block(feat)
            
            feat = self.temporal_pool(feat).squeeze(-1)
            lead_features.append(feat)
        
        temporal_embeddings = torch.stack(lead_features, dim=1)
        
        if layer == 'temporal':
            return temporal_embeddings
        
        node_features = temporal_embeddings.view(batch_size * self.num_leads, -1)
        
        h = node_features
        for i, gnn in enumerate(self.gnns):
            h = gnn(h, edge_index, edge_weight)
            
            # Activation (except last layer)
            if i < len(self.gnns) - 1:
                h = F.silu(h)
        
        return h.view(batch_size, self.num_leads, -1)     
    
    def get_lead_importance(self, x, edge_index=None, edge_weight=None):
        spatial_emb = self.get_embeddings(x, edge_index, edge_weight, layer='spatial')
        
        # Compute L2 norm as importance measure
        lead_importance = torch.norm(spatial_emb, p=2, dim=2)
        
        # Normalize to [0, 1]
        lead_importance = lead_importance / (lead_importance.sum(dim=1, keepdim=True) + 1e-8)
        
        return lead_importance
