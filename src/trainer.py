import torch
import torch.nn as nn
from tqdm import tqdm
import os
from src.metrics import compute_metrics_multitask
from src.utils import get_device
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau, StepLR, CosineAnnealingLR, SequentialLR
from torch.nn.utils import clip_grad_norm_
from src.models.base import BaseECGModel

class Trainer:
    def __init__(self, model: BaseECGModel, config, train_loader, val_loader, test_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        self.train_cfg = config['training']
        eval_cfg = config['evaluation']

        self.epochs = self.train_cfg['epochs']
        self.lr = self.train_cfg['learning_rate']
        self.weight_decay = self.train_cfg['weight_decay']
        self.patience = self.train_cfg.get('early_stopping_patience', 15)
        self.primary_metric = eval_cfg.get('primary_metric', 'f2')
        self.metrics_list = eval_cfg.get('metrics_list', ['f2', 'acc'])
        
        self.device = get_device(config)
        self.model.to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        if self.train_cfg.get('warmup_epochs'):
            self.warmup = LinearLR(self.optimizer, start_factor=0.1, total_iters=self.train_cfg['warmup_epochs'])
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[self.warmup, self._build_scheduler()],
                milestones=[self.train_cfg['warmup_epochs']]
            )
        else:
            self.scheduler = self._build_scheduler() 
        
        task_weights = {
            task_name: task_config['weight']
            for task_name, task_config in config['tasks'].items()
            if task_config['enabled']
        }
        self.criterion = MultiTaskLoss(
            task_weights=task_weights,
            loss_type=self.train_cfg.get('loss_function', 'focal'),
            loss_params=self.train_cfg.get('loss_params', {})
        )
        self.device_name = config.get('device', 'cpu')
        self.scaler = GradScaler(self.device_name)

        self.best_metric_val = 0.0
        self.patience_counter = 0
        self.primary_task = list(task_weights.keys())[0]
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        task_losses_sum = {task: 0 for task in self.model.tasks}
        
        for batch in tqdm(self.train_loader, desc="Training"):
            signals = batch['signal'].to(self.device)
            labels = {
                task: batch['labels'][task].to(self.device) 
                for task in self.model.tasks
            }
            
            self.optimizer.zero_grad()
            kwargs = {}
            if 'edge_index' in batch:
                kwargs['edge_index'] = batch['edge_index'].to(self.device)
                kwargs['edge_weight'] = batch['edge_weight'].to(self.device)

            if self.train_cfg.get('use_mixing_precision'):
                with autocast(self.device_name):
                    outputs = self.model(signals, **kwargs)
                    loss, task_losses = self.criterion(outputs, labels)
                    
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)

                if self.train_cfg.get('gradient_clip_norm'):
                    clip_grad_norm_(self.model.parameters(), self.train_cfg['gradient_clip_norm'])

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(signals, **kwargs)
                loss, task_losses = self.criterion(outputs, labels)
                loss.backward()

                if self.train_cfg.get('gradient_clip_norm'):
                    clip_grad_norm_(self.model.parameters(), self.train_cfg['gradient_clip_norm'])

                self.optimizer.step()

            total_loss += loss.item()
            for task, task_loss in task_losses.items():
                task_losses_sum[task] += task_loss
        
        avg_loss = total_loss / len(self.train_loader)
        avg_task_losses = {
            task: loss / len(self.train_loader) 
            for task, loss in task_losses_sum.items()
        }
        
        return avg_loss, avg_task_losses
    
    def train(self):
        print(f"\nTraining for {self.epochs} epochs...")
        
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch+1}/{self.epochs}")
            
            train_loss, train_task_losses = self.train_epoch()
            val_metrics = self.validate()
            
            print(f"Train Loss: {train_loss:.4f}")
            for task, loss in train_task_losses.items():
                print(f"  {task}: {loss:.4f}")
            
            print(f"\nValidation Metrics:")
            for task in self.model.tasks:
                task_metrics = val_metrics[task]
                print(f"  {task}:")
                for metric, value in task_metrics.items():
                    print(f"    {metric.capitalize()}: {value:.4f}")
            
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"\nLearning Rate: {current_lr:.6f}")

            primary_metric = val_metrics[self.primary_task][self.primary_metric]
            if primary_metric > self.best_metric_val:
                self.best_metric_val = primary_metric
                self.patience_counter = 0
                self.save_checkpoint()
                print(f"New best {self.primary_task} {self.primary_metric}: {primary_metric:.4f}")
            else:
                self.patience_counter += 1
                print(f"No improvement ({self.patience_counter}/{self.patience})")
            
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics[self.primary_task][self.primary_metric])
                else:
                    self.scheduler.step()

            if self.patience_counter >= self.patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

    def validate(self, loader=None):
        if loader is None:
            loader = self.val_loader
        self.model.eval()
        all_preds = {task: [] for task in self.model.tasks}
        all_labels = {task: [] for task in self.model.tasks}
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Validation"):
                signals = batch['signal'].to(self.device)
                labels = {
                    task: batch['labels'][task].to(self.device) 
                    for task in self.model.tasks
                }

                kwargs = {}
                if 'edge_index' in batch:
                    kwargs['edge_index'] = batch['edge_index'].to(self.device)
                    kwargs['edge_weight'] = batch['edge_weight'].to(self.device)
                
                outputs = self.model(signals, **kwargs)
                
                for task in self.model.tasks:
                    probs = torch.sigmoid(outputs[task])
                    all_preds[task].append(probs.cpu())
                    all_labels[task].append(labels[task].cpu())
        
        for task in self.model.tasks:
            all_preds[task] = torch.cat(all_preds[task])
            all_labels[task] = torch.cat(all_labels[task])
        
        metrics = compute_metrics_multitask(all_labels, all_preds, metrics_list=self.metrics_list)
        return metrics
    
    def testing(self):
        print("\n" + "="*60)
        print("Evaluating on test set...")
        
        test_metrics = self.validate(self.test_loader)
        
        print(f"\nTest Metrics:")
        for task in self.model.tasks:
            task_metrics = test_metrics[task]
            print(f"  {task}:")
            for metric, value in task_metrics.items():
                print(f"    {metric.capitalize()}: {value:.4f}")

        return test_metrics
    
    def save_checkpoint(self):
        path = os.path.join(self.train_cfg['checkpoint_dir'], self.train_cfg['checkpoint_name'])
        torch.save(self.model.state_dict(), path)
    
    def _build_scheduler(self):
        scheduler_type = self.train_cfg.get('scheduler')
        
        if scheduler_type == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs,
                eta_min=self.train_cfg.get('min_lr', 1e-6)
            )
        elif scheduler_type == 'step':
            return StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        
        elif scheduler_type == 'plateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=10
            )
        
        return None

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.79, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
        p_t = torch.exp(-bce_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        
class MultiTaskLoss(nn.Module):
    def __init__(self, task_weights, loss_type, loss_params=None):
        super().__init__()
        self.task_weights = task_weights
        loss_params = loss_params or {}
        
        self.task_losses = nn.ModuleDict()
        for task in task_weights.keys():
            if loss_type == 'focal':
                self.task_losses[task] = FocalLoss(**loss_params)
            elif loss_type in ['bce', 'weighted_bce']:
                self.task_losses[task] = nn.BCEWithLogitsLoss(**loss_params)
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(self, predictions, targets):
        total_loss = 0
        task_losses_dict = {}
        
        for task in self.task_weights.keys():
            if task in predictions and task in targets:
                task_loss = self.task_losses[task](predictions[task], targets[task].float())
                weighted_loss = self.task_weights[task] * task_loss
                total_loss += weighted_loss
                task_losses_dict[task] = task_loss.item()
        
        return total_loss, task_losses_dict