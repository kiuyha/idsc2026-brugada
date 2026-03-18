import numpy as np
import torch
from scipy.signal import butter, filtfilt

def bandpass_filter(signal, lowcut=0.5, highcut=40, fs=100, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    
    b, a = butter(order, [low, high], btype="band")
    
    # Apply filter (handles both [time, ch] and [ch, time])
    if signal.shape[0] > signal.shape[1]:
        filtered = filtfilt(b, a, signal, axis=0)
    else:
        filtered = filtfilt(b, a, signal, axis=1)
    
    return filtered


def compute_corr_adjacency(signal):
    # Transpose to for correlation
    leads_signal = signal.T
    
    # Compute Pearson correlation between all lead pairs
    adj_matrix = np.corrcoef(leads_signal)
    
    # Handle NaN (in case of constant signals)
    adj_matrix = np.nan_to_num(adj_matrix, nan=0.0)
    
    return adj_matrix

def build_anatomical_adjacency():
    """
    Build fixed anatomical adjacency matrix based on lead positions
    
    Lead order: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
    Indices:    0   1   2    3    4    5    6   7   8   9  10  11
    """
    adj = np.zeros((12, 12), dtype=np.float32)
    
    # Limb lead connections (Einthoven's triangle)
    limb_connections = [
        (0, 1), (0, 4),  # I connects to II and aVL
        (1, 2), (1, 5),  # II connects to III and aVF
        (2, 5),          # III connects to aVF
        (3, 4), (3, 5),  # aVR connects to aVL and aVF
        (4, 5)           # aVL connects to aVF
    ]
    
    # Precordial connections (sequential across chest)
    precordial_connections = [
        (6, 7),   # V1-V2
        (7, 8),   # V2-V3
        (8, 9),   # V3-V4
        (9, 10),  # V4-V5
        (10, 11)  # V5-V6
    ]
    
    # Add all connections (bidirectional)
    for i, j in limb_connections + precordial_connections:
        adj[i, j] = 1.0
        adj[j, i] = 1.0
    
    # Add self-loops
    np.fill_diagonal(adj, 1.0)
    
    return adj

def build_visibility_graph(signal, device=None):
    if device is None:
        device = signal.device
    
    n = signal.shape[0]
    signal_np = signal.cpu().numpy()
    
    edges = []
    
    # Efficient visibility check
    for i in range(n):
        for j in range(i + 2, min(i + 50, n)):
            visible = True
            
            # Check if any point between i and j blocks visibility
            for k in range(i + 1, j):
                # Linear interpolation
                y_line = signal_np[i] + (signal_np[j] - signal_np[i]) * (k - i) / (j - i)
                
                if signal_np[k] >= y_line:
                    visible = False
                    break
            
            if visible:
                edges.append([i, j])
                edges.append([j, i])  # Undirected
    
    # Empty graph fallback
    if len(edges) > 0:
        edge_index = torch.tensor(edges, dtype=torch.long, device=device).t()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
    
    return edge_index, None

def adjacency_to_edge_index(adj_matrix, threshold=0.3):
    num_nodes = adj_matrix.shape[0]
    edge_list = []
    edge_weights = []
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and abs(adj_matrix[i, j]) > threshold:
                edge_list.append([i, j])
                
                # Force weights to be positive
                edge_weights.append(abs(adj_matrix[i, j]))

    # Convert to tensors
    if len(edge_list) > 0:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t() 
        edge_weight = torch.tensor(edge_weights, dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_weight = torch.zeros(0, dtype=torch.float)
    
    return edge_index, edge_weight

def normalize_signal(signal, method='zscore_per_lead'):
    if method == 'zscore_per_lead':
        mean = signal.mean(axis=0, keepdims=True)
        std = signal.std(axis=0, keepdims=True) + 1e-8
        return (signal - mean) / std
    
    elif method == 'zscore_global':
        mean = signal.mean()
        std = signal.std() + 1e-8
        return (signal - mean) / std
    
    elif method == 'minmax':
        min_val = signal.min(axis=0, keepdims=True)
        max_val = signal.max(axis=0, keepdims=True)
        return (signal - min_val) / (max_val - min_val + 1e-8)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
def augment_ecg(signal, config):
    augmented = signal.copy()

    if config.get('noise_std', 0) > 0:
        noise = np.random.normal(0, config['noise_std'], signal.shape)
        augmented += noise
    
    if config.get('time_shift', 0) > 0:
        shift = np.random.randint(-config['time_shift'], config['time_shift'])
        augmented = np.roll(augmented, shift, axis=0)
    
    if config.get('amplitude_scale', 0) > 0:
        scale = 1 + np.random.uniform(-config['amplitude_scale'], config['amplitude_scale'])
        augmented *= scale
    
    return augmented