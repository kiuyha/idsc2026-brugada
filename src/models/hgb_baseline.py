import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from src.models.base import BaseECGModel

class HistBoostBaseline(BaseECGModel):
    def __init__(self, config):
        super().__init__(config)

        self.model = MultiOutputClassifier(
            estimator=HistGradientBoostingClassifier(
                max_iter=self.params_cfg.get('max_iter', 100),
                min_samples_leaf=self.params_cfg.get('min_samples_leaf', 20),
                max_depth=self.params_cfg.get('max_depth', None),
                class_weight=self.params_cfg.get('class_weight', 'balanced'),
                random_state=config.get('seed', 42),
            ),
            n_jobs=config.get('num_workers', -1)
        )
        
    def forward(self, x, **kwargs):
        features = self.extract_features(x)
        return self.model.predict_proba(features)
    
    @property
    def num_parameters(self):
        try:
            return sum(
                tree.nodes.shape[0]
                for estimator in self.model.estimators_
                for iteration in estimator._predictors
                for tree in iteration
            )
        except Exception:
            return self.params_cfg.get('max_iter', 100)
    
    def extract_features(self, x):
        batch_size, num_leads, signal_len = x.shape
        fs = 100

        # Generic statistical baseline
        means = torch.mean(x, dim=2)
        stds  = torch.std(x, dim=2)
        maxs  = torch.max(x, dim=2)[0]
        mins  = torch.min(x, dim=2)[0]

        # ST-segment elevation (core Brugada marker)
        qrs_end_samples  = int(0.04 * fs)
        st_offset        = int(0.06 * fs)
        st_window        = int(0.04 * fs)

        r_peak_idx = torch.argmax(torch.abs(x), dim=2)
        st_elevations = torch.zeros(batch_size, num_leads)
        j_point_vals  = torch.zeros(batch_size, num_leads)

        for b in range(batch_size):
            for l in range(num_leads):
                rp = r_peak_idx[b, l].item()
                j_idx  = min(rp + qrs_end_samples, signal_len - 1)
                st_start = min(j_idx + st_offset, signal_len - 1)
                st_end   = min(st_start + st_window, signal_len)

                j_point_vals[b, l]  = x[b, l, j_idx]
                if st_end > st_start:
                    st_elevations[b, l] = x[b, l, st_start:st_end].mean()

        # ST slope (coved vs saddle-back shape) 
        st_slope = torch.zeros(batch_size, num_leads)
        slope_window = int(0.08 * fs)

        for b in range(batch_size):
            for l in range(num_leads):
                rp = r_peak_idx[b, l].item()
                seg_start = min(rp + qrs_end_samples, signal_len - 1)
                seg_end   = min(seg_start + slope_window, signal_len)
                seg_len   = seg_end - seg_start
                if seg_len > 1:
                    seg = x[b, l, seg_start:seg_end]
                    t   = torch.arange(seg_len, dtype=torch.float32)
                    t_mean   = t.mean()
                    seg_mean = seg.mean()
                    slope = ((t - t_mean) * (seg - seg_mean)).sum() / ((t - t_mean) ** 2).sum()
                    st_slope[b, l] = slope

        # R′ wave (secondary peak after R in V1–V2)
        r_prime_amp = torch.zeros(batch_size, num_leads)
        search_window = int(0.06 * fs)

        for b in range(batch_size):
            for l in range(num_leads):
                rp = r_peak_idx[b, l].item()
                search_start = min(rp + int(0.02 * fs), signal_len - 1)
                search_end   = min(search_start + search_window, signal_len)
                if search_end > search_start:
                    r_prime_amp[b, l] = x[b, l, search_start:search_end].max()

        # QRS duration proxy 
        threshold = 0.1
        qrs_duration = torch.zeros(batch_size, num_leads)

        for b in range(batch_size):
            for l in range(num_leads):
                lead_sig  = x[b, l]
                lead_max  = lead_sig.abs().max().item()
                thresh    = threshold * lead_max
                above     = (lead_sig.abs() > thresh).nonzero(as_tuple=True)[0]
                if len(above) > 1:
                    qrs_duration[b, l] = (above[-1] - above[0]).float() / fs * 1000

        # T-wave polarity (negative in V1 = Type 1 Brugada)
        t_wave_start = int(0.2 * fs)
        t_wave_end   = int(0.4 * fs)
        t_wave_mean  = torch.zeros(batch_size, num_leads)

        for b in range(batch_size):
            for l in range(num_leads):
                rp = r_peak_idx[b, l].item()
                t_start = min(rp + t_wave_start, signal_len - 1)
                t_end   = min(rp + t_wave_end, signal_len)
                if t_end > t_start:
                    t_wave_mean[b, l] = x[b, l, t_start:t_end].mean()

        features = torch.cat([
            means,
            stds,
            maxs,
            mins,
            st_elevations,
            j_point_vals,
            st_slope,
            r_prime_amp,
            qrs_duration,
            t_wave_mean,
        ], dim=1)

        return features.numpy()
    