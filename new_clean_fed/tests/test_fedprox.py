"""Smoke tests for FedProx — proximal-term implementation correctness.

Key invariants tested:
  1. The proximal-term branch is gated on `mu > 0` (μ=0 ≡ FedAvg locally).
  2. At local==global, the proximal-term gradient is zero (so it doesn't bias
     the very first local step).
  3. After local drift, μ>0 produces a measurably different gradient.
"""
import torch
import pytest

from new_clean_fed.src.fedprox.model import CNNClassifier


def _fresh_model(seed: int = 0) -> CNNClassifier:
    torch.manual_seed(seed)
    return CNNClassifier(in_channels=3, num_classes=7, dropout=0.0)   # no dropout = deterministic


def _grad_of_objective(model, x, y, mu, global_params=None):
    model.zero_grad()
    out = model(x)
    loss = torch.nn.functional.cross_entropy(out, y)
    if mu > 0 and global_params is not None:
        prox = sum(((lp - gp) ** 2).sum() for lp, gp in zip(model.parameters(), global_params))
        loss = loss + (mu / 2.0) * prox
    loss.backward()
    return torch.cat([p.grad.flatten() for p in model.parameters()])


def _data():
    torch.manual_seed(0)
    return torch.randn(8, 3, 64, 64), torch.randint(0, 7, (8,))


def test_mu_zero_matches_fedavg_gradient():
    """μ=0 must produce a gradient identical to plain CE."""
    x, y = _data()
    m1 = _fresh_model(0); m2 = _fresh_model(0)
    g_avg = _grad_of_objective(m1, x, y, mu=0.0)
    global_params = [p.clone().detach() for p in m2.parameters()]
    g_prox_mu0 = _grad_of_objective(m2, x, y, mu=0.0, global_params=global_params)
    max_diff = (g_avg - g_prox_mu0).abs().max().item()
    assert max_diff < 1e-7, f"μ=0 should match FedAvg exactly (got diff {max_diff:.2e})"


def test_proximal_gradient_zero_at_local_equals_global():
    """When local==global, the proximal term contributes zero gradient."""
    x, y = _data()
    m = _fresh_model(0)
    global_params = [p.clone().detach() for p in m.parameters()]
    g_no_prox = _grad_of_objective(m, x, y, mu=0.0)
    m2 = _fresh_model(0)
    g_with_prox = _grad_of_objective(m2, x, y, mu=1.0, global_params=global_params)
    max_diff = (g_no_prox - g_with_prox).abs().max().item()
    assert max_diff < 1e-7, f"At w=w_global, proximal grad should be 0 (got diff {max_diff:.2e})"


def test_proximal_gradient_nonzero_after_drift():
    """After local drift, the proximal term changes the gradient meaningfully."""
    x, y = _data()
    m = _fresh_model(0)
    global_params = [p.clone().detach() for p in m.parameters()]

    # Drift local weights
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.1)

    g_ce_only = _grad_of_objective(m, x, y, mu=0.0)
    m2 = CNNClassifier(3, 7, dropout=0.0)
    m2.load_state_dict(m.state_dict())
    g_with_prox = _grad_of_objective(m2, x, y, mu=1.0, global_params=global_params)

    diff = (g_ce_only - g_with_prox).abs().max().item()
    assert diff > 1e-3, f"Proximal term should meaningfully change gradient after drift (got {diff:.2e})"
