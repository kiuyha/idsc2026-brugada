import numpy as np
from scipy.signal import butter, filtfilt
from data_loader import load_dataset


def bandpass_filter(signal, lowcut=0.5, highcut=40, fs=100, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(order, [low, high], btype="band")
    filtered = filtfilt(b, a, signal, axis=0)

    return filtered

def preprocess_dataset(X):
    filtered_X = []

    for sample in X:
        filtered_signal = bandpass_filter(sample)
        filtered_X.append(filtered_signal)

    return np.array(filtered_X)

def compute_adjacency_matrix(signal):
    leads_signal = signal.T

    adj_matrix = np.corrcoef(leads_signal)

    return adj_matrix

def compute_graph_dataset(X):
    adjacency_matrices = []

    for sample in X:
        adj = compute_adjacency_matrix(sample)
        adjacency_matrices.append(adj)

    return np.array(adjacency_matrices)

#Load Dataset
dataset_path = "./brugada-huca-12-lead-ecg-recordings-for-the-study-of-brugada-syndrome-1.0.0"

# load dataset
X, y = load_dataset(dataset_path)

# bandpass filtering
X_filtered = preprocess_dataset(X)

# adjacency matrices
A = compute_graph_dataset(X_filtered)

print("Signal shape:", X_filtered.shape)
print("Adjacency shape:", A.shape)