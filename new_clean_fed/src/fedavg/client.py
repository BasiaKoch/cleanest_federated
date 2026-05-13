"""Flower client for FedAvg — vanilla local CE training.

Mirrors src/seg_fedavg/client.py from the reference repo but uses PyTorch +
cross-entropy classification (instead of Keras + Dice segmentation).
"""
from __future__ import annotations

from typing import List

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from new_clean_fed.src.fedavg.model import CNNClassifier


def state_dict_to_numpy(model: nn.Module) -> List[np.ndarray]:
    """All state_dict values (incl. BatchNorm buffers) as numpy arrays."""
    return [v.detach().cpu().numpy() for v in model.state_dict().values()]


def numpy_to_state_dict(model: nn.Module, arrays: List[np.ndarray]) -> None:
    """Load numpy arrays back into the model's state_dict (in order)."""
    sd = model.state_dict()
    new_sd = {}
    for (key, ref_val), arr in zip(sd.items(), arrays):
        new_sd[key] = torch.tensor(arr, dtype=ref_val.dtype)
    model.load_state_dict(new_sd, strict=True)


class FedAvgClient(fl.client.NumPyClient):
    """Local SGD with plain cross-entropy. No proximal term."""

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
    ):
        self.cid = int(cid)
        self.device = torch.device(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_local_epochs = num_local_epochs
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.model = CNNClassifier(in_channels=3, num_classes=7, dropout=dropout).to(self.device)
        self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self, config=None):
        return state_dict_to_numpy(self.model)

    def set_parameters(self, parameters):
        numpy_to_state_dict(self.model, parameters)
        self.model.to(self.device)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
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
                loss = self.criterion(self.model(x), y)
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
