import torch
from sklearn.ensemble import RandomForestClassifier
from src.models.base import BaseECGModel

class RFBaseline(BaseECGModel):
    def __init__(self, config):
        super().__init__(config)

        self.rf_model = RandomForestClassifier(
            max_depth=self.params_cfg.get('max_depth', 6),
            min_samples_split=self.params_cfg.get('min_samples_split', 2),
            n_estimators=self.params_cfg.get('n_estimators', 100),
            class_weight=self.params_cfg.get('class_weight', 'balanced'),
            random_state=config.get('seed', 42),
            n_jobs=config.get('num_workers', 1),
        )
        
    def extract_features(self, x):
        means = torch.mean(x, dim=2)
        stds = torch.std(x, dim=2)
        maxs = torch.max(x, dim=2)[0]
        mins = torch.min(x, dim=2)[0]
        
        features = torch.cat([means, stds, maxs, mins], dim=1)
        return features.numpy()
    
    def forward(self, x, **kwargs):
        features = self.extract_features(x)
        return self.rf_model.predict_proba(features)
    
    @property
    def num_parameters(self):
        return 0