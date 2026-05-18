"""Flower NumPyClient implementing FedAvg / FedProx local training.

This client mirrors the local-training logic in
`mnist_dermnist/fl/local_train.py` so a Flower simulation produces results
equivalent (within floating-point noise) to the pure-PyTorch loop.

Per-client per-round local epochs are read from `config["local_epochs"]`
passed by the server, allowing the same client class to be used for both
the statistical-heterogeneity sweeps (uniform E) and the
system-heterogeneity sweeps (varying E per round).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset


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
        self.criterion = nn.CrossEntropyLoss()

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
        rng_seed = self.seed * 10_000 + round_num * 100 + self.cid
        gen = torch.Generator().manual_seed(rng_seed)
        torch.manual_seed(rng_seed)

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
                # GATED — μ=0 path is bit-identical to plain CE (no overhead).
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
            len(self.train_subset),
            {
                "train_loss": total_loss / max(n_batches, 1),
                "cid": self.cid,
                "local_epochs": local_epochs,
            },
        )

    def evaluate(self, parameters, config):
        """Per-client evaluation is a no-op here; eval happens centrally."""
        self.set_parameters(parameters)
        return 0.0, 1, {"cid": self.cid}
