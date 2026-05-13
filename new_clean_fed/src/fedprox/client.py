"""Flower client for FedProx — local CE + proximal term `(μ/2)||w - w_global||²`.

The proximal term is added to the per-batch loss; gradients flow only through
the LOCAL parameters (the global snapshot is detached). This is the canonical
Li et al. 2020 FedProx formulation.
"""
from __future__ import annotations

from typing import List

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from new_clean_fed.src.fedprox.model import CNNClassifier
from new_clean_fed.src.fedavg.client import state_dict_to_numpy, numpy_to_state_dict


class FedProxClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cpu",
        num_local_epochs: int = 5,
        lr: float = 0.02,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        dropout: float = 0.2,
        mu: float = 0.1,
    ):
        self.cid = int(cid)
        self.device = torch.device(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_local_epochs = num_local_epochs
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.mu = float(mu)
        self.model = CNNClassifier(in_channels=3, num_classes=7, dropout=dropout).to(self.device)
        self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self, config=None):
        return state_dict_to_numpy(self.model)

    def set_parameters(self, parameters):
        numpy_to_state_dict(self.model, parameters)
        self.model.to(self.device)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        # Snapshot of GLOBAL params before any local update — detached so they
        # do not receive gradient. This is the "w_t" in the FedProx paper.
        global_params = [p.clone().detach() for p in self.model.parameters()]

        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay,
        )
        self.model.train()
        total_loss, n_batches = 0.0, 0
        for _ in range(self.num_local_epochs):
            for x, y in self.train_loader:
                x = x.to(self.device); y = y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(x)
                task_loss = self.criterion(logits, y)
                if self.mu > 0:
                    prox = torch.zeros(1, device=self.device)
                    for local_p, global_p in zip(self.model.parameters(), global_params):
                        prox = prox + ((local_p - global_p.to(self.device)) ** 2).sum()
                    loss = task_loss + (self.mu / 2.0) * prox.squeeze()
                else:
                    loss = task_loss
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
                n_batches += 1
        return (
            self.get_parameters(),
            len(self.train_loader.dataset),
            {"train_loss": total_loss / max(n_batches, 1), "cid": self.cid},
        )

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        total_loss, total, correct = 0.0, 0, 0
        with torch.no_grad():
            for x, y in self.val_loader:
                x = x.to(self.device); y = y.to(self.device)
                logits = self.model(x)
                total_loss += float(self.criterion(logits, y).item()) * y.size(0)
                pred = logits.argmax(1)
                correct += int((pred == y).sum().item())
                total += y.size(0)
        loss = total_loss / max(total, 1)
        acc = correct / max(total, 1)
        return loss, total, {"accuracy": acc, "cid": self.cid}
