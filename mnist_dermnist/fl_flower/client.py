"""Flower NumPyClient for FedAvg / FedProx using DermMNISTCNN.

A thin Flower wrapper around the same local-training and proximal-term logic
that lives in `mnist_dermnist.fl.local_train`. Both algorithms share this
class; FedProx is just FedAvg with `proximal_mu > 0`.

Used by `experiments/run_cpu_flower.py`. Kept deliberately minimal so the
behavioural surface area mirrors the pure-PyTorch `run_cpu_quick.py`.
"""
from __future__ import annotations

from typing import List

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mnist_dermnist.models import DermMNISTCNN


def state_dict_to_numpy(model: nn.Module) -> List[np.ndarray]:
    """state_dict (parameters + buffers) → list of numpy arrays."""
    return [v.detach().cpu().numpy() for v in model.state_dict().values()]


def numpy_to_state_dict(model: nn.Module, arrays: List[np.ndarray]) -> None:
    """Load a list of numpy arrays back into model.state_dict (in order)."""
    sd = model.state_dict()
    new_sd = {}
    for (key, ref), arr in zip(sd.items(), arrays):
        new_sd[key] = torch.tensor(arr, dtype=ref.dtype)
    model.load_state_dict(new_sd, strict=True)


class FlClient(fl.client.NumPyClient):
    """Flower client for FedAvg/FedProx on DermMNISTCNN.

    Uses the SAME gating logic as the pure-PyTorch path:
      - proximal_mu == 0 → plain cross-entropy (FedAvg path)
      - proximal_mu  > 0 → CE + (μ/2)·Σ‖w − w_global‖²  (FedProx path)
    """

    def __init__(
        self,
        cid: int,
        train_loader: DataLoader,
        num_local_epochs: int,
        lr: float,
        momentum: float,
        proximal_mu: float,
        dropout: float = 0.2,
    ):
        self.cid = int(cid)
        self.train_loader = train_loader
        self.num_local_epochs = int(num_local_epochs)
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.proximal_mu = float(proximal_mu)
        self.device = torch.device("cpu")
        self.model = DermMNISTCNN(num_classes=7, dropout=dropout).to(self.device)
        self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self, config=None):
        return state_dict_to_numpy(self.model)

    def set_parameters(self, parameters):
        numpy_to_state_dict(self.model, parameters)
        self.model.to(self.device)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        # Snapshot global params ONCE per round; detached so they receive no
        # gradient. Identical mechanism to the pure-PyTorch path.
        global_params = [p.clone().detach() for p in self.model.parameters()]

        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
        )
        self.model.train()
        total_loss, n_batches = 0.0, 0
        for _ in range(self.num_local_epochs):
            for x, y in self.train_loader:
                x = x.to(self.device)
                y = y.to(self.device).view(-1).long()
                optimizer.zero_grad()
                loss = self.criterion(self.model(x), y)
                # GATED — μ=0 path is bit-identical to plain CE.
                if self.proximal_mu > 0:
                    prox = torch.zeros((), device=self.device)
                    for w, w_g in zip(self.model.parameters(), global_params):
                        prox = prox + ((w - w_g) ** 2).sum()
                    loss = loss + (self.proximal_mu / 2.0) * prox
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
        # Per-client validation under specialist partitions is degenerate
        # (single class). Return a placeholder so Flower doesn't complain;
        # the real evaluation happens centrally via the strategy's evaluate_fn.
        self.set_parameters(parameters)
        return 0.0, 1, {"cid": self.cid}
