import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, roc_auc_score

def compute_metrics(labels, predictions, threshold=0.5):
    # Convert to numpy
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    
    # Flatten if needed
    labels = labels.flatten()
    predictions = predictions.flatten()
    
    # Binary predictions
    preds_binary = (predictions >= threshold).astype(int)
    
    # Compute metrics
    metrics = {
        'accuracy': accuracy_score(labels, preds_binary),
        'precision': precision_score(labels, preds_binary, zero_division=0),
        'recall': recall_score(labels, preds_binary, zero_division=0),
        'f1': f1_score(labels, preds_binary, zero_division=0),
        'f2': fbeta_score(labels, preds_binary, beta=2.0, zero_division=0),
        'auc': roc_auc_score(labels, predictions) if len(np.unique(labels)) > 1 else 0.0
    }
    
    return metrics

def compute_metrics_multitask(labels_dict, predictions_dict, threshold=0.5):
    all_metrics = {}
    
    for task in labels_dict.keys():
        task_labels = labels_dict[task]
        task_preds = predictions_dict[task]
        
        all_metrics[task] = compute_metrics(task_labels, task_preds, threshold)
    
    return all_metrics