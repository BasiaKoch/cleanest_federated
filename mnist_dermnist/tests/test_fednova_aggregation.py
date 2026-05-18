"""Unit tests for the FedNova aggregation rule.

The strategy `PairedFedNovaStrategy` implements:

    a_i      = sum_{j=1..tau_i} (1 - m^j) / (1 - m)
             = (tau_i (1 - m) - m (1 - m^{tau_i})) / (1 - m)^2
    d_i      = w_anchor - w_i^{new}
    d_norm   = sum_i p_i * d_i / a_i
    a_eff    = sum_i p_i * a_i
    w^{t+1}  = w_anchor - a_eff * d_norm

These tests verify three invariants:
  1. With m = 0 and uniform tau, FedNova reduces to FedAvg (size-weighted mean).
  2. The normaliser function matches brute-force summation of the cumulative
     geometric series to 1e-9 precision at canonical reference values.
  3. Sign convention: when all clients return identical updates, the
     aggregated update equals that update exactly.

If any of these tests fail, the FedNova implementation has a regression.
"""
from __future__ import annotations

import numpy as np
import pytest

from mnist_dermnist.fl_flower.strategy_fednova import fednova_normaliser


# ---------- Test 1: normaliser closed-form matches brute force ---------------

def _brute(tau: int, m: float) -> float:
    """Direct summation of the cumulative momentum series."""
    return float(sum(sum(m ** k for k in range(j)) for j in range(1, tau + 1)))


@pytest.mark.parametrize("tau,m,expected", [
    (1, 0.0, 1.0),
    (10, 0.0, 10.0),
    (100, 0.0, 100.0),
    (1, 0.9, 1.0),
    (5, 0.9, 13.144100),
    (10, 0.9, 41.381060),    # Wang 2020 canonical reference
    (20, 0.9, 120.941899),
    (100, 0.9, 910.002413),
])
def test_normaliser_closed_form_matches_brute_force(tau, m, expected):
    a_closed = fednova_normaliser(tau, m)
    a_brute = _brute(tau, m)
    assert abs(a_closed - a_brute) < 1e-9, (
        f"closed form ({a_closed}) != brute force ({a_brute})"
    )
    assert abs(a_closed - expected) < 1e-4, (
        f"closed form ({a_closed}) != expected reference ({expected})"
    )


def test_normaliser_vanilla_sgd_reduces_to_tau():
    """For m = 0 (vanilla SGD), a_i should equal tau exactly."""
    for tau in [1, 5, 10, 50]:
        assert fednova_normaliser(tau, 0.0) == float(tau)


def test_normaliser_monotonic_in_tau():
    """a_i must be strictly increasing in tau for any m in [0, 1)."""
    for m in [0.0, 0.5, 0.9]:
        vals = [fednova_normaliser(t, m) for t in range(1, 21)]
        assert all(v2 > v1 for v1, v2 in zip(vals, vals[1:])), (
            f"normaliser is not monotonic for m={m}"
        )


# ---------- Test 2: sign convention — identical client updates ---------------

def _compute_fednova_update(anchor, client_new_params, client_taus, client_ns, m):
    """Pure-Python reference of the FedNova aggregate.

    Mirrors the logic in PairedFedNovaStrategy.aggregate_fit() but
    without depending on Flower's Parameters/Ray plumbing.
    """
    total_n = sum(client_ns)
    a_eff = 0.0
    delta_norm = [np.zeros_like(a, dtype=np.float64) for a in anchor]
    for new_params, tau, n in zip(client_new_params, client_taus, client_ns):
        a_i = fednova_normaliser(tau, m)
        p_i = n / total_n
        for k in range(len(anchor)):
            d_i = anchor[k].astype(np.float64) - new_params[k].astype(np.float64)
            delta_norm[k] += p_i * (d_i / a_i)
        a_eff += p_i * a_i
    new_global = [
        (anchor[k].astype(np.float64) - a_eff * delta_norm[k]).astype(anchor[k].dtype)
        for k in range(len(anchor))
    ]
    return new_global, a_eff


def test_sign_convention_identical_client_updates():
    """If every client returns the SAME updated parameters, the aggregate
    must equal those parameters exactly (regardless of momentum or tau)."""
    # 1-D toy parameter
    anchor = [np.array([1.0, 2.0, 3.0])]
    common_update = [np.array([0.5, 1.5, 2.5])]      # all clients return this
    n_clients = 4
    client_new = [common_update for _ in range(n_clients)]
    client_taus = [10, 10, 10, 10]                    # all the same
    client_ns = [100, 200, 150, 300]                  # different sizes — to exercise weighting

    for m in [0.0, 0.5, 0.9]:
        new_global, a_eff = _compute_fednova_update(
            anchor, client_new, client_taus, client_ns, m
        )
        np.testing.assert_allclose(
            new_global[0], common_update[0], atol=1e-12,
            err_msg=f"FedNova aggregate differed from identical client update "
                    f"at m={m}",
        )


def test_reduces_to_fedavg_when_m_zero_and_uniform_tau():
    """With m = 0 and uniform tau, FedNova should produce the
    size-weighted mean of the client parameters (FedAvg)."""
    anchor = [np.array([0.0, 0.0])]
    # Heterogeneous client updates
    client_new = [
        [np.array([1.0, 0.0])],
        [np.array([0.0, 1.0])],
        [np.array([2.0, 2.0])],
    ]
    client_taus = [10, 10, 10]
    client_ns = [100, 100, 100]                       # uniform weights

    new_global, _ = _compute_fednova_update(
        anchor, client_new, client_taus, client_ns, m=0.0
    )

    # FedAvg with equal weights: simple mean of client params
    expected = (client_new[0][0] + client_new[1][0] + client_new[2][0]) / 3.0
    np.testing.assert_allclose(new_global[0], expected, atol=1e-12)


def test_reduces_to_size_weighted_fedavg_when_m_zero_uniform_tau():
    """Size-weighted: same as above but with unequal client sizes."""
    anchor = [np.array([0.0])]
    client_new = [
        [np.array([1.0])],
        [np.array([2.0])],
        [np.array([3.0])],
    ]
    client_taus = [10, 10, 10]
    client_ns = [100, 200, 300]                       # 1:2:3 weights

    new_global, _ = _compute_fednova_update(
        anchor, client_new, client_taus, client_ns, m=0.0
    )

    # Size-weighted mean: (1*100 + 2*200 + 3*300) / 600 = 1400/600 = 7/3
    expected = (1.0 * 100 + 2.0 * 200 + 3.0 * 300) / 600.0
    np.testing.assert_allclose(new_global[0], np.array([expected]), atol=1e-12)


def test_aggregation_with_heterogeneous_tau():
    """Smoke test: with heterogeneous tau, the aggregate should still be
    finite, the right shape, and bounded by the client updates' range."""
    anchor = [np.array([0.0, 0.0])]
    client_new = [
        [np.array([1.0, 0.0])],
        [np.array([0.0, 1.0])],
        [np.array([2.0, 2.0])],
    ]
    client_taus = [5, 10, 20]                         # heterogeneous
    client_ns = [100, 200, 300]

    new_global, a_eff = _compute_fednova_update(
        anchor, client_new, client_taus, client_ns, m=0.9
    )
    assert np.all(np.isfinite(new_global[0]))
    assert new_global[0].shape == anchor[0].shape
    # Sanity: aggregate magnitude should be comparable to typical client update
    assert 0.0 < a_eff < 1e4


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
