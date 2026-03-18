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
        ggn_kwargs = self.params_cfg.get('ggn_kwargs', {})

        self.lead_encoder = nn.ModuleList([
            ResNetBlock(1, channels[0], kernel_size),
            *[ResNetBlock(channels[i], channels[i+1], kernel_size) for i in range(len(channels) - 1)],
        ])
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        
        # Disable self_loops since preprocessing already handles this
        self.gnns = nn.ModuleList([
            self._build_gnn_layer(gnn_type, channels[-1], hidden_dim, **ggn_kwargs),
            *[self._build_gnn_layer(gnn_type, hidden_dim, hidden_dim, **ggn_kwargs) for _ in range(num_gnn_layers - 1)],
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.task_heads = nn.ModuleDict({
            task : nn.Sequential(
                nn.Linear(hidden_dim * 12, 1),
            )
            for task in self.tasks
        })

    def _build_gnn_layer(self, gnn_type, input_dim, output_dim, **kwargs):
        if gnn_type == "gcn":
            return GCNConv(input_dim, output_dim, **kwargs)
        elif gnn_type == "gat":
            return GATConv(input_dim, output_dim, **kwargs)
        elif gnn_type == "gin":
            return GINConv(nn=nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.BatchNorm1d(output_dim),
                nn.SiLU(),
                nn.Linear(output_dim, output_dim),
            ), **kwargs)
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")
    
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
        edge_weight = edge_weight.clamp(-1.0, 1.0)
        
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
            if isinstance(gnn, GCNConv) and edge_weight is not None:
                h = gnn(h, edge_index, edge_weight)
            else:
                h = gnn(h, edge_index)
                
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
