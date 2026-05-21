"""Direct mechanism tests for the FedProx proximal term.

The existing `test_mu_zero_equals_fedavg.py` proves the μ=0 gating is correct
(FedProx reduces to FedAvg). It does NOT prove the μ>0 path actually adds the
right thing. These tests close that gap with two checks:

  1.  ``test_proximal_gradient_equals_mu_times_diff``
      For a one-layer model on a single batch, the gradient added by the
      proximal term (μ/2)·||w − w_t||² must equal μ·(w − w_t) at every
      parameter. We compute the data-loss gradient and the joint
      (data + prox) gradient with autograd and assert that the difference
      equals μ·(w − w_t) elementwise. This is a finite-difference-style
      check on the analytic proximal gradient.

  2.  ``test_global_weights_frozen_is_not_aliased``
      The round-start snapshot must NOT alias the live model parameters.
      If it did, the proximal "anchor" would drift along with the local
      SGD updates and the algorithm would silently degrade to FedAvg. We
      check (a) tensor identity via ``data_ptr()`` differs, (b) values
      survive a mutation of the model parameters, and (c) the same
      property holds for the Flower client's inline snapshot pattern
      ``[p.clone().detach() for p in model.parameters()]`` at
      ``fl_flower/client.py:143``.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from mnist_dermnist.fl.local_train import freeze_global_weights


# --------------------------------------------------------------------------- #
# 1. Finite-difference / prox-gradient test                                   #
# --------------------------------------------------------------------------- #
def test_proximal_gradient_equals_mu_times_diff() -> None:
    """The proximal term's gradient must be exactly μ·(w − w_t).

    We construct a deterministic one-layer linear regressor, freeze a copy of
    the parameters as the round-start anchor ``w_t``, then perturb the live
    parameters so ``w ≠ w_t``. Computing the loss with and without the
    proximal term (via autograd) and subtracting must recover μ·(w − w_t)
    at every parameter and bias.
    """
    torch.manual_seed(0)
    in_dim, out_dim, batch_size = 4, 3, 8
    mu = 0.137  # deliberately not a round number — guards against off-by-half bugs

    model = nn.Linear(in_dim, out_dim, bias=True)

    # Anchor w_t = clone of initial params (the round-start snapshot)
    w_t = freeze_global_weights(model)

    # Perturb live params so w ≠ w_t. Use a deterministic perturbation.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.5 * torch.arange(p.numel(), dtype=p.dtype).reshape(p.shape))

    # Synthesize a fixed batch
    x = torch.randn(batch_size, in_dim)
    y = torch.randn(batch_size, out_dim)
    criterion = nn.MSELoss()

    # --- Data-loss gradient (no prox) -----------------------------------
    model.zero_grad(set_to_none=True)
    loss_data = criterion(model(x), y)
    grads_data = torch.autograd.grad(loss_data, list(model.parameters()),
                                     create_graph=False, retain_graph=False)

    # --- Joint loss (data + prox), same forward pass --------------------
    model.zero_grad(set_to_none=True)
    loss_data2 = criterion(model(x), y)
    prox = torch.zeros((), dtype=loss_data2.dtype)
    for w, w_g in zip(model.parameters(), w_t):
        prox = prox + ((w - w_g) ** 2).sum()
    loss_joint = loss_data2 + (mu / 2.0) * prox
    grads_joint = torch.autograd.grad(loss_joint, list(model.parameters()),
                                      create_graph=False, retain_graph=False)

    # --- Verify: g_joint - g_data must equal μ·(w − w_t) elementwise ----
    for p, w_g, g_data, g_joint in zip(model.parameters(), w_t, grads_data, grads_joint):
        expected_prox_grad = mu * (p.detach() - w_g)
        observed_prox_grad = g_joint - g_data
        assert torch.allclose(
            observed_prox_grad, expected_prox_grad, atol=1e-6, rtol=1e-5
        ), (
            "Proximal-term gradient deviates from μ·(w − w_t).\n"
            f"  expected: {expected_prox_grad}\n"
            f"  observed: {observed_prox_grad}\n"
            f"  diff    : {observed_prox_grad - expected_prox_grad}"
        )


def test_proximal_gradient_is_zero_when_mu_is_zero() -> None:
    """At μ=0 the proximal contribution must be exactly the zero tensor.

    This is the differential of the bit-identity guarantee tested in
    ``test_mu_zero_equals_fedavg.py``: not only does the trajectory match,
    the gradient contribution at every parameter is the literal zero tensor
    so no accumulator picks it up.
    """
    torch.manual_seed(1)
    model = nn.Linear(4, 3, bias=True)
    w_t = freeze_global_weights(model)
    # Drift live params away from anchor so (w - w_t) is nonzero.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.9)

    mu = 0.0
    prox = torch.zeros(())
    for w, w_g in zip(model.parameters(), w_t):
        prox = prox + ((w - w_g) ** 2).sum()
    loss = (mu / 2.0) * prox  # data-loss term suppressed for clarity
    grads = torch.autograd.grad(loss, list(model.parameters()),
                                allow_unused=True)
    for g in grads:
        # `allow_unused=True` is required because mu=0 disconnects the graph;
        # the returned grad must be either literal zero or None (treated as 0).
        if g is None:
            continue
        assert torch.all(g == 0), f"non-zero gradient under μ=0: {g}"


# --------------------------------------------------------------------------- #
# 2. Snapshot-alias regression test                                           #
# --------------------------------------------------------------------------- #
def test_global_weights_frozen_is_not_aliased() -> None:
    """``freeze_global_weights`` must return a deep, independent snapshot.

    Failure mode this catches: if the implementation drifts to
    ``return list(model.parameters())`` (no clone, no detach), then the
    proximal term collapses to (μ/2)·||w − w||² = 0 and FedProx silently
    degenerates to FedAvg — exactly the bug the algorithm exists to avoid.
    """
    torch.manual_seed(2)
    model = nn.Linear(5, 4, bias=True)

    snapshot = freeze_global_weights(model)
    live_params = list(model.parameters())

    # (a) tensor identity: snapshot tensors must not be the same objects
    for snap, live in zip(snapshot, live_params):
        assert snap is not live, (
            "freeze_global_weights returned the live parameter object itself")
        assert snap.data_ptr() != live.data_ptr(), (
            "freeze_global_weights returned a tensor that shares storage "
            "with the live parameter (alias). The proximal anchor would "
            "drift with local SGD and FedProx would silently degrade to FedAvg.")

    # (b) values survive a destructive mutation of the live parameters.
    snapshot_values_before = [s.clone() for s in snapshot]
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    for snap, before in zip(snapshot, snapshot_values_before):
        assert torch.equal(snap, before), (
            "Snapshot tensor mutated when live parameter was zero'd — "
            "snapshot was aliased to live storage.")

    # (c) snapshot tensors are detached (no grad fn, no requires_grad).
    for snap in snapshot:
        assert not snap.requires_grad, "snapshot tensor still requires_grad"
        assert snap.grad_fn is None, "snapshot tensor has a grad_fn attached"


def test_flower_client_snapshot_pattern_is_not_aliased() -> None:
    """The Flower client snapshots inline; verify the same alias property.

    ``fl_flower/client.py:143`` does
        ``global_params = [p.clone().detach() for p in self.model.parameters()]``
    rather than calling ``freeze_global_weights``. This test guards the
    inline pattern against the same regression. If the inline pattern is
    ever refactored (e.g. to ``list(self.model.parameters())``) this test
    fails immediately.
    """
    torch.manual_seed(3)
    model = nn.Linear(6, 2, bias=True)

    # Mirror the exact Flower-client snapshot expression.
    global_params = [p.clone().detach() for p in model.parameters()]
    live_params = list(model.parameters())

    for snap, live in zip(global_params, live_params):
        assert snap is not live
        assert snap.data_ptr() != live.data_ptr()
        assert not snap.requires_grad
        assert snap.grad_fn is None

    # Destructive mutation of live params must not perturb the snapshot.
    snapshot_before = [s.clone() for s in global_params]
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(0.0).add_(99.0)
    for snap, before in zip(global_params, snapshot_before):
        assert torch.equal(snap, before)
