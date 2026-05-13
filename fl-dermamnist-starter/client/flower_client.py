from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.optim import SGD
import flwr as fl
from losses.focal_loss import FocalLoss
from metrics.evaluation import evaluate_model


def get_state_dict_as_numpy(model):
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]


def set_state_dict_from_numpy(model, np_arrays):
    state_dict = model.state_dict()
    keys = list(state_dict.keys())
    assert len(keys) == len(np_arrays), f'Parameter count mismatch: model has {len(keys)}, received {len(np_arrays)}'
    new_state = {}
    for key, arr in zip(keys, np_arrays):
        new_state[key] = torch.as_tensor(arr, dtype=state_dict[key].dtype)
    model.load_state_dict(new_state, strict=True)


class FLClient(fl.client.NumPyClient):
    def __init__(self, cid, model, train_loader, val_loader, device, num_classes,
                 local_epochs=1, lr=0.01, momentum=0.9, weight_decay=0.0,
                 loss_fn='ce', class_weights=None, focal_gamma=2.0, proximal_mu=0.0):
        self.cid = str(cid)
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_classes = num_classes
        self.local_epochs = local_epochs
        self.proximal_mu = proximal_mu
        if class_weights is not None:
            class_weights = class_weights.to(device)
        if loss_fn in ['ce', 'weighted_ce']:
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        elif loss_fn == 'focal':
            self.loss_fn = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        else:
            raise ValueError(f'Unknown loss_fn: {loss_fn}')
        self.optimizer = SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    def get_parameters(self, config):
        return get_state_dict_as_numpy(self.model)

    def set_parameters(self, parameters):
        set_state_dict_from_numpy(self.model, parameters)
        self.model.to(self.device)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        # System heterogeneity: server may override local_epochs per round per client.
        local_epochs = int(config.get('local_epochs', self.local_epochs))
        # PATCH 7: server may override proximal_mu per round (adaptive μ)
        proximal_mu = float(config.get('proximal_mu', self.proximal_mu))
        if proximal_mu > 0:
            global_params = [p.clone().detach() for p in self.model.parameters()]
        else:
            global_params = None
        self.model.train()
        total_loss, num_batches = 0.0, 0
        for _ in range(local_epochs):
            for x, y in self.train_loader:
                x = x.to(self.device)
                y = y.to(self.device).view(-1).long()
                self.optimizer.zero_grad()
                loss = self.loss_fn(self.model(x), y)
                if global_params is not None:
                    prox = 0.0
                    for local_p, global_p in zip(self.model.parameters(), global_params):
                        prox = prox + torch.sum((local_p - global_p.to(self.device)) ** 2)
                    # Use the per-round proximal_mu (supports adaptive μ from server)
                    loss = loss + (proximal_mu / 2.0) * prox
                loss.backward()
                self.optimizer.step()
                total_loss += float(loss.item())
                num_batches += 1
        avg_loss = total_loss / max(num_batches, 1)
        return self.get_parameters(config={}), len(self.train_loader.dataset), {
            'train_loss': avg_loss,
            'cid': self.cid,
            'local_epochs_actual': local_epochs,
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        metrics = evaluate_model(self.model, self.val_loader, self.device, self.num_classes)
        return float(metrics['loss']), len(self.val_loader.dataset), {
            'accuracy': float(metrics['accuracy']),
            'balanced_accuracy': float(metrics['balanced_accuracy']),
            'macro_f1': float(metrics['macro_f1']),
            'worst_class_f1': float(metrics['worst_class_f1']),
            'cid': self.cid,
        }


def create_client_fn(model_fn, train_loaders, val_loaders, device, num_classes, config, client_class_weights=None):
    def client_fn(cid: str):
        cid_int = int(cid)
        model = model_fn().to(device)
        class_weights = None
        if client_class_weights is not None:
            class_weights = client_class_weights[cid_int]
        client = FLClient(
            cid=cid,
            model=model,
            train_loader=train_loaders[cid_int],
            val_loader=val_loaders[cid_int],
            device=device,
            num_classes=num_classes,
            local_epochs=int(config.get('local_epochs', 1)),
            lr=float(config.get('lr', 0.01)),
            momentum=float(config.get('momentum', 0.9)),
            weight_decay=float(config.get('weight_decay', 0.0)),
            loss_fn=config.get('loss_fn', 'ce'),
            class_weights=class_weights,
            focal_gamma=float(config.get('focal_gamma', 2.0)),
            proximal_mu=float(config.get('proximal_mu', 0.0)),
        )
        return client.to_client()
    return client_fn
