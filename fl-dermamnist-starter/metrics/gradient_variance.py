"""Gradient-variance dissimilarity metric — Li et al. 2020 (FedProx) Figure 2.

Measures E_k[||∇F_k(w) - ∇f(w)||²] where:
  - F_k = local loss on client k's data
  - f   = global (size-weighted) loss
  - w   = current global model parameters

Higher value = clients pull in more different directions = stronger
statistical heterogeneity. The FedProx paper's primary mechanistic claim is
that μ > 0 reduces this variance.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset, DataLoader


@torch.no_grad()
def _flatten_grads(model: nn.Module) -> torch.Tensor:
    pieces = []
    for p in model.parameters():
        if p.grad is None:
            pieces.append(torch.zeros_like(p).flatten())
        else:
            pieces.append(p.grad.detach().flatten())
    return torch.cat(pieces)


def compute_gradient_variance(
    model: nn.Module,
    train_dataset: Dataset,
    client_indices: Sequence[Sequence[int]],
    device: torch.device,
    batch_size: int = 64,
    loss_fn: nn.Module | None = None,
) -> dict:
    """Compute the weighted variance of client gradients around the global gradient.

    The model is assumed to already have the *current global* weights loaded.
    For each client, evaluates a single mini-batch (up to `batch_size` samples
    from the client's data; full data if smaller) and computes ∇L_k(w).

    Returns a dict with:
      - grad_variance: E_k[||∇F_k - ∇f||²]
      - grad_norm_global: ||∇f||
      - grad_norm_local_mean: mean_k ||∇F_k||
      - grad_norm_local_max: max_k ||∇F_k||
    """
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    model = model.to(device)
    # Disable BN running-stat updates; we want a clean gradient signal
    model.eval()

    n_total = sum(len(idxs) for idxs in client_indices)
    if n_total == 0:
        return {'grad_variance': float('nan')}
    p_k = [len(idxs) / n_total for idxs in client_indices]

    grads_per_client: List[torch.Tensor] = []
    losses_per_client: List[float] = []
    for idxs in client_indices:
        if not idxs:
            grads_per_client.append(None)
            losses_per_client.append(float('nan'))
            continue
        # Use up to `batch_size` samples (random shuffle of client subset for fairness)
        n = min(batch_size, len(idxs))
        rng = np.random.default_rng(0)   # deterministic per-call
        sel = rng.choice(len(idxs), size=n, replace=False)
        subset = Subset(train_dataset, [idxs[int(i)] for i in sel])
        loader = DataLoader(subset, batch_size=n, shuffle=False)
        x, y = next(iter(loader))
        x = x.to(device)
        y = y.to(device).view(-1).long()

        model.zero_grad(set_to_none=False)
        with torch.enable_grad():
            loss = loss_fn(model(x), y)
            loss.backward()
        grads_per_client.append(_flatten_grads(model).clone())
        losses_per_client.append(float(loss.item()))

    valid = [(p, g) for p, g in zip(p_k, grads_per_client) if g is not None]
    if not valid:
        return {'grad_variance': float('nan')}
    total_p = sum(p for p, _ in valid)
    valid = [(p / total_p, g) for p, g in valid]

    g_global = torch.zeros_like(valid[0][1])
    for p, g in valid:
        g_global = g_global + p * g
    var = sum(p * ((g - g_global) ** 2).sum() for p, g in valid)

    local_norms = [float(g.norm().item()) for _, g in valid]

    return {
        'grad_variance': float(var.item()),
        'grad_norm_global': float(g_global.norm().item()),
        'grad_norm_local_mean': float(np.mean(local_norms)),
        'grad_norm_local_max': float(np.max(local_norms)),
        'loss_global': float(np.average([losses_per_client[i] for i, (p, _) in enumerate(zip(p_k, grads_per_client)) if grads_per_client[i] is not None],
                                        weights=[p_k[i] for i, _ in enumerate(p_k) if grads_per_client[i] is not None])),
    }
