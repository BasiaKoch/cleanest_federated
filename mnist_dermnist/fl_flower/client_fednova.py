"""FedNova NumPyClient — normalised aggregation for heterogeneous local steps.

Implements the algorithm of Wang et al. (2020), "Tackling the Objective
Inconsistency Problem in Heterogeneous Federated Optimization" (NeurIPS).

Background
----------
Under heterogeneous local epoch counts (`E_i` varies per client per round),
the FedAvg size-weighted aggregate is biased: clients that perform more
local SGD steps contribute proportionally larger parameter updates, even
though all client updates are weighted only by `n_i`. FedNova removes this
"objective inconsistency" by normalising each client's update by the
effective number of local SGD steps `tau_i` before aggregation.

Aggregation rule
----------------
Let `d_i = w^t - w_i^{t+1}` be client i's parameter delta (positive when
the client moved away from the global anchor). FedNova's normalised
aggregate is

    d_norm    = sum_i (p_i * d_i / tau_i)
    tau_eff   = sum_i (p_i * tau_i)
    w^{t+1}   = w^t  -  tau_eff * d_norm

where `p_i = n_i / sum_j n_j` is the standard size weight. With uniform
`tau_i` this reduces to FedAvg.

Implementation note
-------------------
Flower's built-in strategies aggregate the full state_dict (parameters
plus buffers); FedNova should normalise only parameter deltas, not
buffers. For the DermMNISTCNN architecture (GroupNorm, no running stats)
the state_dict contains only learnable parameters, so the practical
distinction does not arise in this thesis. The implementation below
normalises every entry in the returned list of arrays uniformly --- valid
under the GroupNorm-only assumption stated in the methodology.

This client returns its parameter delta and `tau_i` via the fit metrics
dict; the aggregation logic lives in a custom Strategy
(`PairedFedNovaStrategy` in this module's accompanying strategy file, to
be added before the system-het + FedNova sweep is run).
"""
from __future__ import annotations

from typing import Dict, List

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from mnist_dermnist.fl_flower.client import state_dict_to_numpy, numpy_to_state_dict


class FlClientFedNova(fl.client.NumPyClient):
    """FedNova client: returns parameter delta + effective local steps.

    The accompanying strategy aggregates `delta / tau` rather than
    parameters directly.
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
        self.device = torch.device(device)
        self.epoch_schedule = epoch_schedule

        self.train_subset = Subset(train_dataset, self.indices)
        self.model = model_builder().to(self.device)
        self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self, config=None):
        return state_dict_to_numpy(self.model)

    def set_parameters(self, parameters):
        numpy_to_state_dict(self.model, parameters)
        self.model.to(self.device)

    def fit(self, parameters, config: Dict):
        self.set_parameters(parameters)
        round_num = int(config.get("round", 1))
        if self.epoch_schedule is not None:
            local_epochs = int(self.epoch_schedule[round_num - 1, self.cid])
        else:
            local_epochs = int(config.get("local_epochs", 1))

        rng_seed = self.seed * 10_000 + round_num * 100 + self.cid
        gen = torch.Generator().manual_seed(rng_seed)
        torch.manual_seed(rng_seed)

        loader = DataLoader(
            self.train_subset,
            batch_size=min(self.batch_size, max(1, len(self.train_subset))),
            shuffle=True, generator=gen, num_workers=0, drop_last=False,
        )

        # Snapshot the global parameters (the anchor used to compute delta)
        anchor = [p.detach().clone().cpu().numpy()
                  for p in self.model.parameters()]

        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay,
        )

        # Effective local SGD step counter (for FedNova normalisation)
        # tau_i = number of local update steps (= local_epochs * batches_per_epoch).
        # We count batches directly since drop_last=False and batch sizes vary
        # for the smallest client.
        tau = 0
        total_loss = 0.0
        for _ in range(local_epochs):
            for x, y in loader:
                x = x.to(self.device); y = y.to(self.device).view(-1).long()
                optimizer.zero_grad()
                loss = self.criterion(self.model(x), y)
                loss.backward(); optimizer.step()
                tau += 1
                total_loss += float(loss.item())

        # Compute delta = anchor - new_params (parameter-wise)
        new_params = [p.detach().cpu().numpy()
                      for p in self.model.parameters()]
        delta = [a - p for a, p in zip(anchor, new_params)]

        # The strategy will divide by tau and re-weight by client size.
        # We return the FULL state_dict via get_parameters() so Flower's
        # parameter handling is intact, and pass tau + the delta-of-params
        # in metrics; the strategy reads metrics to do FedNova aggregation.
        # Note: passing arrays through metrics is awkward, so we instead
        # return the *new* params and let the strategy reconstruct delta
        # from (anchor, new_params).
        return (
            self.get_parameters(),
            len(self.train_subset),
            {
                "train_loss": total_loss / max(tau, 1),
                "tau": tau,
                "cid": self.cid,
                "local_epochs": local_epochs,
            },
        )

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        return 0.0, 1, {"cid": self.cid}
