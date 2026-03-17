from .base import BaseECGModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv
from src.preprocessing import build_visibility_graph

class TemporalGNN(BaseECGModel):
    def __init__(self, config):
        super().__init__(config)
        
        hidden_dim = self.params_cfg.get('hidden_dim', 64)
        num_gnn_layers = self.params_cfg.get('num_gnn_layers', 3)
        dropout = self.params_cfg.get('dropout', 0.3)
        gnn_type = self.params_cfg.get('gnn_type', 'gcn')
        pooling = self.params_cfg.get('pooling', 'mean')
        gnn_kwargs = self.params_cfg.get('gnn_kwargs', {})
        
        if gnn_type == "gcn":
            GNN = GCNConv
        elif gnn_type == "gat":
            GNN = GATConv
        elif gnn_type == "gin":
            GNN = GINConv
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")
        
        self.gnns = nn.ModuleList([
            GNN(1, hidden_dim),
            *[GNN(hidden_dim, hidden_dim, **gnn_kwargs) for _ in range(num_gnn_layers - 1)],
        ])
        
        self.pooling = pooling
        if pooling == 'attention':
            self.attention_pool = nn.Linear(hidden_dim, 1)
        
        self.dropout = nn.Dropout(dropout)
        
        self.task_heads = nn.ModuleDict({
            task: nn.Linear(hidden_dim * 12, 1)
            for task in self.tasks
        })
    
    def forward(self, x):
        all_embeddings = self.get_embeddings(x)
        
        h = all_embeddings.flatten(start_dim=1)
        h = self.dropout(h)
        
        outputs = {
            task: head(h)
            for task, head in self.task_heads.items()
        }
        
        return outputs
    
    def get_embeddings(self, x):
        batch_size = x.shape[0]
        num_leads = 12
        
        lead_embeddings = []
        
        for lead_idx in range(num_leads):
            lead_signal = x[:, lead_idx, :]
            batch_lead_emb = []
            
            for sample_idx in range(batch_size):
                signal = lead_signal[sample_idx]
                edge_index, edge_weight = build_visibility_graph(signal)
                edge_weight = edge_weight.clamp(-1.0, 1.0)
                node_features = signal.unsqueeze(-1)
                h = node_features

                for i, gnn in enumerate(self.gnns):
                    if isinstance(gnn, GCNConv) and edge_weight is not None:
                        h = gnn(h, edge_index, edge_weight)
                    else:
                        h = gnn(h, edge_index)
                        
                    if i < len(self.gnns) - 1:
                        h = F.silu(h)
                
                if self.pooling == 'mean':
                    graph_emb = h.mean(dim=0)
                elif self.pooling == 'max':
                    graph_emb = h.max(dim=0)[0]
                elif self.pooling == 'attention':
                    attn_scores = self.attention_pool(h)
                    attn_weights = F.softmax(attn_scores, dim=0)
                    graph_emb = (h * attn_weights).sum(dim=0)

                batch_lead_emb.append(graph_emb)
            
            lead_emb = torch.stack(batch_lead_emb, dim=0)
            lead_embeddings.append(lead_emb)
        
        return torch.stack(lead_embeddings, dim=1)