from __future__ import annotations

from pathlib import Path
import copy
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from losses.focal_loss import FocalLoss
from metrics.evaluation import evaluate_model, MetricsLogger


class CentralisedTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader, device, num_classes,
                 lr=1e-3, weight_decay=1e-4, loss_fn='ce', class_weights=None, focal_gamma=2.0):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.num_classes = num_classes
        if class_weights is not None:
            class_weights = class_weights.to(device)
        if loss_fn == 'ce':
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        elif loss_fn == 'weighted_ce':
            if class_weights is None:
                raise ValueError('weighted_ce requires class_weights')
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        elif loss_fn == 'focal':
            self.criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        else:
            raise ValueError(f'Unknown loss_fn: {loss_fn}')
        self.optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, patience=5, factor=0.5, mode='max')

    def train_epoch(self):
        self.model.train()
        total_loss, total = 0.0, 0
        for x, y in self.train_loader:
            x = x.to(self.device)
            y = y.to(self.device).view(-1).long()
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(x), y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * y.numel()
            total += y.numel()
        return total_loss / max(total, 1)

    def evaluate(self, dataloader):
        return evaluate_model(self.model, dataloader, self.device, self.num_classes)

    def train(self, num_epochs, save_dir, class_names=None):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        logger = MetricsLogger(save_dir, class_names)
        best_state = copy.deepcopy(self.model.state_dict())
        best_bacc, best_epoch = -1.0, -1
        for epoch in tqdm(range(1, num_epochs + 1), desc='centralised'):
            train_loss = self.train_epoch()
            val_metrics = self.evaluate(self.val_loader)
            self.scheduler.step(val_metrics['balanced_accuracy'])
            row = dict(val_metrics)
            row['train_loss'] = train_loss
            logger.log_round(epoch, row)
            print(f"epoch={epoch} train_loss={train_loss:.4f} val_bacc={val_metrics['balanced_accuracy']:.4f} val_worst_f1={val_metrics['worst_class_f1']:.4f}")
            if val_metrics['balanced_accuracy'] > best_bacc:
                best_bacc = val_metrics['balanced_accuracy']
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                torch.save(best_state, save_dir / 'best_model.pt')
        logger.save()
        self.model.load_state_dict(best_state)
        test_metrics = self.evaluate(self.test_loader)
        return {
            'test_metrics': test_metrics,
            'history': logger.get_dataframe(),
            'best_epoch': best_epoch,
            'best_val_bacc': best_bacc,
        }
