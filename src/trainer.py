import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from src.metrics import compute_metrics
from src.utils import get_device
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau, StepLR, CosineAnnealingLR, SequentialLR
from torch.nn.utils import clip_grad_norm_

class Trainer:
    def __init__(self, model, config, train_loader, val_loader, test_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        self.train_cfg = config['training']
        self.epochs = self.train_cfg['epochs']
        self.lr = self.train_cfg['learning_rate']
        self.weight_decay = self.train_cfg['weight_decay']
        self.patience = self.train_cfg.get('early_stopping_patience', 15)
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
        
        self.criterion = self._loss_function().to(self.device)
        self.device_name = config.get('device', 'cpu')
        self.scaler = GradScaler(self.device_name)

        self.best_val_f2 = 0.0
        self.patience_counter = 0
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(self.train_loader, desc="Training"):
            signals = batch['signal'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()

            if self.train_cfg.get('use_mixing_precision'):
                with autocast(self.device_name):
                    outputs = self.model(signals)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)

                if self.train_cfg.get('gradient_clip_norm'):
                    clip_grad_norm_(self.model.parameters(), self.train_cfg['gradient_clip_norm'])

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(signals)
                loss = self.criterion(outputs, labels)
                loss.backward()

                if self.train_cfg.get('gradient_clip_norm'):
                    clip_grad_norm_(self.model.parameters(), self.train_cfg['gradient_clip_norm'])

                self.optimizer.step()

            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def train(self):
        print(f"\nTraining for {self.epochs} epochs...")
        
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch+1}/{self.epochs}")
            
            train_loss = self.train_epoch()
            val_metrics = self.validate()
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val F2: {val_metrics['f2']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
            
            # Early stopping
            if val_metrics['f2'] > self.best_val_f2:
                self.best_val_f2 = val_metrics['f2']
                self.patience_counter = 0
                self.save_checkpoint('best_model.pt')
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= self.patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        print("\nEvaluating on test set...")
        test_metrics = self.testing()
        print(f"Test F2: {test_metrics['f2']:.4f} | Test Acc: {test_metrics['accuracy']:.4f}")

    def validate(self, loader=None):
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(loader or self.val_loader, desc="Validation"):
                signals = batch['signal'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(signals)
                
                all_preds.append(outputs.cpu())
                all_labels.append(labels.cpu())
        
        preds = torch.cat(all_preds)
        labels = torch.cat(all_labels)
        
        metrics = compute_metrics(labels, preds)
        return metrics
    
    def testing(self):
        return self.validate(self.test_loader)
    
    def save_checkpoint(self, filename):
        Path('experiments').mkdir(exist_ok=True)
        torch.save(self.model.state_dict(), f'experiments/{filename}')

    def _loss_function(self):
        loss_type = self.train_cfg['loss_function']
        params = self.train_cfg.get('loss_params', {})
        
        if loss_type == 'weighted_bce':
            pos_weight = torch.tensor([287.0 / 76.0])  # 3.78 = 287 / 76
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        elif loss_type == 'focal':
            return FocalLoss(
                alpha=params.get('alpha'),
                gamma=params.get('gamma'),
                reduction=params.get('reduction', 'mean')
            )
    
        return nn.BCELoss()
    
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
        super(self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        
        p_t = inputs * targets + (1 - inputs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss