import torch
from tqdm import tqdm
import os
from src.metrics import compute_metrics_multitask
from src.utils import get_device
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau, StepLR, CosineAnnealingLR, SequentialLR
from torch.nn.utils import clip_grad_norm_
from src.models.base import BaseECGModel
import numpy as np
from src.utils import MultiTaskLoss

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
        self.warmup_epochs = self.train_cfg.get('warmup_epochs', 0)
        
        self.device = get_device(config)
        self.model.to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        if self.warmup_epochs > 0:
            self.warmup = LinearLR(self.optimizer, start_factor=0.1, total_iters=self.warmup_epochs)
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[self.warmup, self._build_scheduler()],
                milestones=[self.warmup_epochs]
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
        
        pbar = tqdm(self.train_loader, desc="Training")
        
        for batch in pbar:
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

            if torch.isnan(loss):
                raise ValueError("Loss is NaN")
            
            current_loss = loss.item()
            total_loss += current_loss
            for task, task_loss in task_losses.items():
                task_losses_sum[task] += task_loss
                
            pbar.set_postfix({'loss': f"{current_loss:.4f}"})
        
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
        
        return compute_metrics_multitask(all_labels, all_preds, metrics_list=self.metrics_list)
    
    def testing(self):
        print("\n" + "="*60)
        print("Evaluating on test set...")

        self.load_checkpoint()
        test_metrics = self.validate(self.test_loader)
        
        print(f"\nTest Metrics:")
        for task in self.model.tasks:
            task_metrics = test_metrics[task]
            print(f"  {task}:")
            for metric, value in task_metrics.items():
                print(f"    {metric.capitalize()}: {value:.4f}")

        return test_metrics
    
    def load_checkpoint(self):
        path = os.path.join(self.train_cfg['checkpoint_dir'], self.train_cfg['checkpoint_name'])
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found at {path}")
        print(f"Loading best model weights from: {path}")
        self.model.load_state_dict(torch.load(path, map_location=self.device))
    
    def save_checkpoint(self):
        path = os.path.join(self.train_cfg['checkpoint_dir'], self.train_cfg['checkpoint_name'])
        torch.save(self.model.state_dict(), path)
    
    def _build_scheduler(self):
        scheduler_type = self.train_cfg.get('scheduler')
        
        if scheduler_type == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs - self.warmup_epochs,
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
    
class TraditionalTrainer:
    def __init__(self, model, config, train_loader, val_loader, test_loader):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.tasks = model.tasks
        self.metrics_list = config['evaluation'].get('metrics_list', ['f2', 'acc'])
 
    def _extract_all_data(self, loader):
        X_all = []
        y_all = {task: [] for task in self.tasks}
        
        for batch in loader:
            features = self.model.extract_features(batch['signal'])
            X_all.append(features)
            
            for task in self.tasks:
                y_all[task].append(batch['labels'][task].numpy().flatten())
                
        X_matrix = np.concatenate(X_all, axis=0)
        y_matrix = np.column_stack([np.concatenate(y_all[task]) for task in self.tasks])

        return X_matrix, y_matrix
 
    def train(self):
        print("Extracting features from Train Loader...")
        X_train, y_train = self._extract_all_data(self.train_loader)
        # y_train = y_train.ravel() if len(self.tasks) == 1 else y_train

        print(f"Training Histogram Boosting on {X_train.shape[0]} samples with {X_train.shape[1]} features...")
        self.model.model.fit(X_train, y_train)
        print("Training complete!")
        
        val_metrics = self.validate()
        print(f"\nValidation Metrics:")
        for task in self.model.tasks:
            task_metrics = val_metrics[task]
            print(f"  {task}:")
            for metric, value in task_metrics.items():
                print(f"    {metric.capitalize()}: {value:.4f}")
 
    def validate(self, loader=None):
        if loader is None:
            loader = self.val_loader
 
        X, y_true_matrix = self._extract_all_data(loader)
        raw_probs = self.model.model.predict_proba(X)

        if isinstance(raw_probs, np.ndarray):
            probs_list = [raw_probs]
        else:
            probs_list = raw_probs

        all_preds = {
            task: torch.tensor(probs_list[i][:, 1] if probs_list[i].ndim == 2 else probs_list[i].flatten(), dtype=torch.float32)
            for i, task in enumerate(self.tasks)
        }
        all_labels = {
            task: torch.tensor(y_true_matrix[:, i], dtype=torch.float32)
            for i, task in enumerate(self.tasks)
        }
 
        return compute_metrics_multitask(all_labels, all_preds, metrics_list=self.metrics_list)
 
    def testing(self):
        print("\n" + "="*60)
        print("Evaluating on test set...")
 
        test_metrics = self.validate(self.test_loader)
 
        print(f"\nTest Metrics:")
        for task, task_metrics in test_metrics.items():
            print(f"  {task}:")
            for metric, value in task_metrics.items():
                print(f"    {metric.capitalize()}: {value:.4f}")
 
        return test_metrics