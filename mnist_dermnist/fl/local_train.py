"""Local client training step.

Both FedAvg and FedProx share this code path. The proximal term is gated on
`proximal_mu > 0`, which means:

  - FedAvg            : proximal_mu=0 → branch skipped → identical to plain CE
  - FedProx (μ=0)     : also takes the skip branch → BIT-IDENTICAL to FedAvg
  - FedProx (μ>0)     : adds (μ/2)·sum(||w − w_global||²) to the per-batch loss

This gating is the only correct way to satisfy the spec's requirement that
"FedProx with μ=0 must produce numerically identical results to FedAvg". A
naive implementation that always adds `0.5 * 0 * prox_term` would differ in
floating-point summation order.
"""
from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def local_train(
    model: nn.Module,
    train_loader: DataLoader,
    *,
    num_epochs: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    proximal_mu: float,
    global_weights_frozen: Sequence[torch.Tensor] | None,
    device: torch.device | str,
) -> dict:
    """One client's local training pass.

    Args:
        model: client's PyTorch model. Will be updated in place.
        train_loader: client's DataLoader (its `generator` should be seeded by
                      the server with base_seed*10000 + round*100 + cid).
        num_epochs: local epochs E.
        lr, momentum, weight_decay: SGD hyperparameters.
        proximal_mu: 0 → plain CE; >0 → CE + (μ/2)·Σ‖w − w_g‖².
        global_weights_frozen: a DETACHED clone of model.parameters() at the
                               start of the communication round. Must NOT
                               change during local training. Pass None when
                               proximal_mu == 0 (the branch is gated off).
        device: torch device.

    Returns:
        {'train_loss': avg loss over local steps, 'n_batches': int}
    """
    device = torch.device(device)
    model = model.to(device).train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    if proximal_mu > 0:
        if global_weights_frozen is None:
            raise ValueError("global_weights_frozen must be provided when proximal_mu > 0")
        # Move the frozen copy to device once (cheap; reused every batch)
        global_weights_frozen = [w.to(device) for w in global_weights_frozen]

    total_loss = 0.0
    n_batches = 0
    for _ in range(num_epochs):
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device).view(-1).long()

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)

            # PROXIMAL TERM BRANCH — gated on μ>0 so μ=0 path is bit-identical to FedAvg
            if proximal_mu > 0:
                prox = torch.zeros((), device=device)
                for w, w_g in zip(model.parameters(), global_weights_frozen):
                    prox = prox + ((w - w_g) ** 2).sum()
                loss = loss + (proximal_mu / 2.0) * prox

            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1

    return {
        "train_loss": total_loss / max(n_batches, 1),
        "n_batches": n_batches,
    }


def freeze_global_weights(model: nn.Module) -> List[torch.Tensor]:
    """Make a detached deep copy of the model's parameters.

    Called by the SERVER ONCE per round, before broadcasting to clients.
    The returned list must NOT be re-assigned to live model parameters; it's a
    snapshot of the round-start global model for the proximal term.
    """
    return [p.detach().clone() for p in model.parameters()]
