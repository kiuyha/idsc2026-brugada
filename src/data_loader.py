import os
import numpy as np
import pandas as pd
import wfdb
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from src.preprocessing import bandpass_filter, compute_corr_adjacency, adjacency_to_edge_index, build_anatomical_adjacency, normalize_signal, augment_ecg

class BrugadaDataset(Dataset):
    def __init__(self, dataset_path, normalize='zscore_per_lead', augment_config=None, model=None, correlation_threshold=0.3, anatomic_weight=0.3, self_loops=True, enabled_tasks=None):
        self.dataset_path = dataset_path
        self.normalize = normalize
        self.model_type = model
        self.correlation_threshold = correlation_threshold
        self.anatomic_weight = anatomic_weight
        self.self_loops = self_loops
        self.augment_config = augment_config
        
        metadata_path = os.path.join(dataset_path, "metadata.csv")
        self.metadata = pd.read_csv(metadata_path)
        self.enabled_tasks = enabled_tasks or ['brugada']
        
        self.patient_ids = self.metadata["patient_id"].values
        self.labels = {}
        for task in self.enabled_tasks:
            if task in self.metadata.columns:
                self.labels[task] = (self.metadata[task].values > 0).astype(np.float32)
            else:
                raise ValueError(f"Task {task} not found in metadata")
            
        # Precompute graph structure if needed
        if self.model_type == 'spatial_gnn':
            self.anatomical_adj = build_anatomical_adjacency()

    def __len__(self):
        return len(self.patient_ids)
    
    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        signal, fs = self._load_record(patient_id)
        
        signal = bandpass_filter(signal, lowcut=0.5, highcut=40, fs=fs, order=4)
        
        if self.normalize:
            signal = normalize_signal(signal, method=self.normalize)
        
        if self.augment_config.get('enabled', False):
            signal = augment_ecg(signal, self.augment_config)
        
        labels_dict = {}
        for task in self.enabled_tasks:
            labels_dict[task] = torch.FloatTensor([self.labels[task][idx]])
        
        sample = {
            'labels': labels_dict,
            'patient_id': patient_id
        }
        
        # Compute graph structure if needed
        if self.model_type == 'spatial_gnn':
            corr_adj = compute_corr_adjacency(signal)
            anatomical_adj = build_anatomical_adjacency()
            hybrid_adj_matrix = self.anatomic_weight * anatomical_adj + (1 - self.anatomic_weight) * corr_adj

            edge_index, edge_weight = adjacency_to_edge_index(
                hybrid_adj_matrix, 
                threshold=self.correlation_threshold,
            )
            sample['edge_index'] = edge_index
            sample['edge_weight'] = edge_weight
        
        sample['signal'] = torch.FloatTensor(signal.T)
        return sample
    
    def _load_record(self, patient_id):
        record_path = os.path.join(
            self.dataset_path,
            "files",
            str(patient_id),
            str(patient_id)
        )
        record = wfdb.rdrecord(record_path)
        
        return record.p_signal, record.fs


def get_dataloaders(config):
    data_cfg = config['data']
    enabled_tasks = [
        task_name 
        for task_name, task_config in config['tasks'].items() 
        if task_config['enabled']
    ]

    full_dataset = BrugadaDataset(
        dataset_path=data_cfg['path'],
        normalize=data_cfg.get('normalize', 'zscore_per_lead'),
        model=config['model']['type'],
        augment_config=data_cfg.get('augmentation', None),
        correlation_threshold=data_cfg.get('correlation_threshold', 0.3),
        anatomic_weight=data_cfg.get('anatomic_weight', 0.3),
        self_loops=data_cfg.get('self_loops', True),
        enabled_tasks=enabled_tasks
    )
    
    total_size = len(full_dataset)
    primary_task = enabled_tasks[0]
    labels = full_dataset.labels[primary_task]
    
    train_size = int(data_cfg['train_split'] * total_size)
    val_size = int(data_cfg['val_split'] * total_size)
    test_size = total_size - train_size - val_size
    
    print(f"Dataset splits: Train={train_size}, Val={val_size}, Test={test_size}")
    print(f"Total samples: {total_size}")

    for task in enabled_tasks:
        task_labels = full_dataset.labels[task]
        print(f"Task '{task}': Positive={task_labels.sum():.0f}, "
              f"Negative={len(task_labels)-task_labels.sum():.0f}")
    
    indices = np.arange(total_size)
    
    train_indices, temp_indices = train_test_split(
        indices,
        train_size=train_size,
        stratify=labels,
        random_state=config.get('seed', 42)
    )

    # Compute class weights for imbalanced datasets (Only used for training)
    train_labels = labels[train_indices]
    class_counts = np.bincount(train_labels.astype(int))
    class_weights = 1.0 / class_counts
    sample_weights = np.array([class_weights[int(label)] for label in train_labels])
    sample_weights = torch.from_numpy(sample_weights).double()
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    temp_labels = labels[temp_indices]
    val_indices, test_indices = train_test_split(
        temp_indices,
        train_size=val_size,
        test_size=test_size,
        stratify=temp_labels,
        random_state=config.get('seed', 42)
    )
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

    num_workers = config['num_workers'] if config.get('num_workers') > 0 else os.cpu_count()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg.get('batch_size', 16),
        num_workers=num_workers,
        sampler=sampler,
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=collate_fn_graph
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_cfg.get('batch_size', 16),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=collate_fn_graph
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=data_cfg.get('batch_size', 16),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=collate_fn_graph
    )
    
    return train_loader, val_loader, test_loader


def collate_fn_graph(batch):
    signals = torch.stack([item['signal'] for item in batch])
    patient_ids = [item['patient_id'] for item in batch]
    
    task_names = list(batch[0]['labels'].keys())
    labels_dict = {}
    for task in task_names:
        labels_dict[task] = torch.stack([item['labels'][task] for item in batch])
    
    result = {
        'signal': signals,
        'labels': labels_dict, 
        'patient_id': patient_ids
    }
    
    # Handle graph data if present
    if 'edge_index' in batch[0]:
        edge_indices = []
        edge_weights = []
        
        for i, item in enumerate(batch):
            edge_index = item['edge_index']
            edge_weight = item['edge_weight']
            
            # Add batch offset to node indices
            edge_index_with_offset = edge_index + (i * 12)
            
            edge_indices.append(edge_index_with_offset)
            edge_weights.append(edge_weight)
        
        # Concatenate all edges
        if len(edge_indices) > 0:
            result['edge_index'] = torch.cat(edge_indices, dim=1)
            result['edge_weight'] = torch.cat(edge_weights, dim=0)
    
    return result