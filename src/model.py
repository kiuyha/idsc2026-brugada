
class ResNetBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        identity = self.skip(x) 
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.SiLU(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + identity 
        out = F.relu(out)
        
        return out
    
class SpatialBrugadaGNN(nn.Module):
    def __init__(self, num_leads=12, hidden_dim=64):
        super().__init__()
        
        # Use REAL ResNet blocks
        self.lead_encoder = nn.ModuleList([
            ResNetBlock1D(1, 32),
            ResNetBlock1D(32, 64),
            ResNetBlock1D(64, 128)
        ])
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)  # [batch, 128, 1]
        
        # Use GCN (simpler, more interpretable than GIN)
        self.gcn1 = GCNConv(128, hidden_dim)      # 128 → 64
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)  # 64 → 64
        self.gcn3 = GCNConv(hidden_dim, hidden_dim)  # 64 → 64
        
        # ===== CLASSIFICATION HEAD =====
        self.classifier_basal_pattern = nn.Sequential(
            nn.Linear(hidden_dim * num_leads, 128),
            nn.SiLu(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )
        self.classifier_sudden_death = nn.Sequential(
            nn.Linear(hidden_dim * num_leads, 128),
            nn.SiLu(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )
        self.classifier_brugada = nn.Sequential(
            nn.Linear(hidden_dim * num_leads, 128),
            nn.SiLu(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )
    
    def forward(self, x, edge_index, edge_weight=None):
        """
        Args:
            x: [batch, 12 leads, time_samples]
            edge_index: [2, num_edges] - computed from correlation
            edge_weight: [num_edges, 1] - correlation values
        """
        batch_size = x.shape[0]
        num_leads = x.shape[1]
        
        # ===== STEP 1: Extract temporal features from each lead =====
        lead_features = []
        for lead_idx in range(num_leads):
            lead_signal = x[:, lead_idx:lead_idx+1, :]  # [batch, 1, time]
            
            # Pass through ResNet blocks
            feat = lead_signal
            for block in self.lead_encoder:
                feat = block(feat)  # [batch, 128, time]
            
            # Pool over time dimension
            feat = self.temporal_pool(feat).squeeze(-1)  # [batch, 128]
            lead_features.append(feat)
        
        # Stack: [batch, 12 leads, 128 features]
        node_features = torch.stack(lead_features, dim=1)
        
        # ===== STEP 2: Process with GCN (spatial relationships) =====
        # Flatten batch and leads for GCN processing
        # GCN expects [total_nodes, features]
        node_features = node_features.view(batch_size * num_leads, -1)  # [batch*12, 128]
        
        # Create batch assignment for pooling later
        batch_assignment = torch.arange(batch_size, device=x.device).repeat_interleave(num_leads)
        
        # GCN layers
        h = F.relu(self.gcn1(node_features, edge_index, edge_weight))
        h = F.relu(self.gcn2(h, edge_index, edge_weight))
        h = self.gcn3(h, edge_index, edge_weight)  # [batch*12, 64]
        
        # ===== STEP 3: Reshape back and classify =====
        h = h.view(batch_size, num_leads, -1)  # [batch, 12, 64]
        h = h.flatten(start_dim=1)  # [batch, 12*64]
        
        # Final prediction
        basal_pattern = torch.sigmoid(self.classifier_basal_pattern(h))
        sudden_death = torch.sigmoid(self.classifier_sudden_death(h))
        brugada = torch.sigmoid(self.classifier_brugada(h))

        return basal_pattern, sudden_death, brugada
