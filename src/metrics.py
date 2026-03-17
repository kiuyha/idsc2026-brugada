import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, roc_auc_score

METRICS = {
    'accuracy': accuracy_score,
    'precision': precision_score,
    'recall': recall_score,
    'f1': f1_score,
    'f2': fbeta_score,
    'auc': roc_auc_score
}

def compute_metrics(labels, predictions, threshold=0.5, metrics_list=['accuracy', 'precision', 'recall', 'f1', 'f2', 'auc']):
    # Convert to numpy
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    
    labels = labels.flatten()
    predictions = predictions.flatten()
    preds_binary = (predictions >= threshold).astype(int)
    
    results = {}
    results = {}
    for metric in metrics_list:
        if metric == 'auc':
            results[metric] = METRICS[metric](labels, predictions) if len(np.unique(labels)) > 1 else 0.0
        else:
            kwargs = {}
            
            if metric in ['precision', 'recall', 'f1', 'f2']:
                kwargs['zero_division'] = 0
                
            if metric == 'f2':
                kwargs['beta'] = 2
                
            results[metric] = METRICS[metric](labels, preds_binary, **kwargs)
            
    return results
    
def compute_metrics_multitask(labels_dict, predictions_dict, threshold=0.5, metrics_list=['accuracy', 'precision', 'recall', 'f1', 'f2', 'auc']):
    all_metrics = {}
    
    for task in labels_dict.keys():
        task_labels = labels_dict[task]
        task_preds = predictions_dict[task]
        
        all_metrics[task] = compute_metrics(task_labels, task_preds, threshold, metrics_list)
    
    return all_metrics