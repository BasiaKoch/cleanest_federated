"""Flower NumPyClient implementing FedAvg / FedProx local training.

This client mirrors the local-training logic in
`fl_dermamnist/fl/local_train.py` so a Flower simulation produces results
equivalent (within floating-point noise) to the pure-PyTorch loop.

Per-client per-round local epochs are read from `config["local_epochs"]`
passed by the server, allowing the same client class to be used for both
the statistical-heterogeneity sweeps (uniform E) and the
system-heterogeneity sweeps (varying E per round).
"""
from __future__ import annotations

import random
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from fl_dermamnist.fl.seeding import dataloader_generator_seed, numpy_legacy_seed


def state_dict_to_numpy(model: nn.Module) -> List[np.ndarray]:
    """Convert a model state_dict (parameters + buffers) to a list of numpy arrays.

    Flower passes parameters around as lists of numpy arrays. We include
    buffers (e.g., GroupNorm has no running stats but other layers might)
    so the aggregated state is complete.
    """
    return [v.detach().cpu().numpy() for v in model.state_dict().values()]


def numpy_to_state_dict(model: nn.Module, arrays: List[np.ndarray]) -> None:
    """Load a list of numpy arrays back into model.state_dict in order."""
    sd = model.state_dict()
    new_sd = {}
    for (key, ref), arr in zip(sd.items(), arrays):
        new_sd[key] = torch.tensor(arr, dtype=ref.dtype)
    model.load_state_dict(new_sd, strict=True)


class FlClient(fl.client.NumPyClient):
    """Flower client implementing FedAvg / FedProx.

    Parameters
    ----------
    cid : int
        Client id (used to seed per-client RNG state).
    train_dataset : Dataset
        Full training set; the client's slice is taken via Subset(indices).
    indices : list[int]
        Client's training-sample indices.
    model_builder : callable
        Zero-arg callable returning a fresh model instance.
    seed : int
        Global experiment seed; used to derive per-(round, cid) RNG state.
    lr, momentum, weight_decay, batch_size : float / int
        SGD hyperparameters.
    proximal_mu : float
        FedProx μ. Zero ⇒ FedAvg (gated branch).
    device : str
        "cpu" or "cuda".
    """

    def __init__(
        self,
        cid: int,
        train_dataset: Dataset,
        indices: List[int],
        model_builder,
        seed: int,
        lr: float,
        momentum: float,
        weight_decay: float,
        batch_size: int,
        proximal_mu: float,
        device: str = "cpu",
        epoch_schedule: "np.ndarray | None" = None,
        criterion: nn.Module | None = None,
    ) -> None:
        self.cid = int(cid)
        self.indices = list(indices)
        self.seed = int(seed)
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.proximal_mu = float(proximal_mu)
        self.device = torch.device(device)
        # epoch_schedule[r, k] is the local-epoch count for client k in 1-based round r+1
        # If None, fall back to config["local_epochs"] passed by the server.
        self.epoch_schedule = epoch_schedule

        self.train_subset = Subset(train_dataset, self.indices)
        self.model = model_builder().to(self.device)
        # Local-training loss (audit HV2). CE by default; the runner can
        # inject a class-weighted CE or FocalLoss to evaluate loss-side
        # imbalance baselines without changing the FedAvg/FedProx
        # aggregation rule.
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()

    def get_parameters(self, config=None):
        return state_dict_to_numpy(self.model)

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        numpy_to_state_dict(self.model, parameters)
        self.model.to(self.device)

    def fit(self, parameters, config: Dict):
        """Local training on this client.

        config keys consumed:
          - "round" (int): the current communication round (1-based).
          - "local_epochs" (int): number of local epochs for THIS client THIS round.
        """
        self.set_parameters(parameters)

        round_num = int(config.get("round", 1))
        # Prefer per-client schedule if provided; fall back to config; final default 1.
        if self.epoch_schedule is not None:
            local_epochs = int(self.epoch_schedule[round_num - 1, self.cid])
        else:
            local_epochs = int(config.get("local_epochs", 1))

        # Per-(seed, round, cid) RNG: identical to the pure-PyTorch path's
        # dataloader_generator_seed() so results match between the two runtimes.
        raw_seed = dataloader_generator_seed(self.seed, round_num, self.cid)
        gen = torch.Generator().manual_seed(raw_seed)
        torch.manual_seed(raw_seed)
        random.seed(raw_seed)      # Defensive: Ray workers don't inherit driver-process RNG state
        np.random.seed(numpy_legacy_seed(raw_seed))  # NumPy requires a 32-bit seed

        loader = DataLoader(
            self.train_subset,
            batch_size=min(self.batch_size, max(1, len(self.train_subset))),
            shuffle=True,
            generator=gen,
            num_workers=0,
            drop_last=False,
        )

        # Snapshot global params ONCE per round for the proximal term.
        global_params = [p.clone().detach() for p in self.model.parameters()]

        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

        self.model.train()
        total_loss, n_batches = 0.0, 0
        for _ in range(local_epochs):
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device).view(-1).long()
                optimizer.zero_grad()
                loss = self.criterion(self.model(x), y)
                # GATED - μ=0 path is bit-identical to plain CE (no overhead).
                if self.proximal_mu > 0:
                    prox = torch.zeros((), device=self.device)
                    for w, w_g in zip(self.model.parameters(), global_params):
                        prox = prox + ((w - w_g) ** 2).sum()
                    loss = loss + (self.proximal_mu / 2.0) * prox
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
                n_batches += 1

        # Mechanism diagnostic - L2 norm of (post-training - anchor), computed
        # over all trainable parameters before the server aggregates. Cheap
        # (one pass over params); returned as a scalar in the fit metrics
        # so the strategy / runner can capture it per (round, client) when
        # --log-update-norms is set on the server side. Cost is negligible
        # so we always compute and emit it; the runner decides whether to
        # write a CSV.
        with torch.no_grad():
            sq = 0.0
            for w, w_g in zip(self.model.parameters(), global_params):
                diff = (w.detach() - w_g).flatten()
                sq += float((diff.to(torch.float64) ** 2).sum())
        update_norm = float(sq ** 0.5)

        return (
            self.get_parameters(),
            len(self.train_subset),
            {
                "train_loss": total_loss / max(n_batches, 1),
                "cid": self.cid,
                "local_epochs": local_epochs,
                "update_norm": update_norm,
            },
        )

    def evaluate(self, parameters, config):
        """Per-client evaluation is a no-op here; eval happens centrally."""
        self.set_parameters(parameters)
        return 0.0, 1, {"cid": self.cid}
