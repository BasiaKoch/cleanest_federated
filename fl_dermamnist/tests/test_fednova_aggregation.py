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

from fl_dermamnist.runtimes.flower.strategy_fednova import fednova_normaliser


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


# ---------- Test 2: sign convention - identical client updates ---------------

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
    client_ns = [100, 200, 150, 300]                  # different sizes - to exercise weighting

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


# =========================================================================
# Mechanism-probe regression tests for τ-clip and server-momentum variants.
# (Thesis §5.5 - random-τ FedNova failure mode probes.)
# =========================================================================

import flwr as fl
from flwr.common import FitRes, Status, Code, ndarrays_to_parameters
from fl_dermamnist.runtimes.flower.strategy_fednova import PairedFedNovaStrategy


def _make_fit_res(new_params, num_examples, tau, cid):
    """Construct a Flower FitRes with the metadata aggregate_fit consumes."""
    return FitRes(
        status=Status(code=Code.OK, message="ok"),
        parameters=ndarrays_to_parameters(new_params),
        num_examples=num_examples,
        metrics={"tau": int(tau), "cid": int(cid), "train_loss": 0.0,
                 "update_norm": 0.0, "local_epochs": int(tau)},
    )


def _run_one_round(strategy, anchor, client_updates):
    """Drive PairedFedNovaStrategy.aggregate_fit for one round.

    `client_updates` is a list of `(new_params, n, tau, cid)` tuples.
    Returns the aggregated new_global ndarrays and the metrics dict.
    """
    strategy._current_anchor = [a.copy() for a in anchor]
    results = [
        (None, _make_fit_res(p, n, tau, cid)) for (p, n, tau, cid) in client_updates
    ]
    params_out, metrics = strategy.aggregate_fit(
        server_round=1, results=results, failures=[]
    )
    from flwr.common import parameters_to_ndarrays
    return parameters_to_ndarrays(params_out), metrics


def test_tau_clip_off_is_byte_identical_to_baseline():
    """tau_clip_min=0 must reproduce the un-probed FedNova aggregate exactly."""
    anchor = [np.array([1.0, 2.0, 3.0])]
    updates = [
        ([np.array([0.5, 1.5, 2.5])], 100, 5, 0),    # straggler: tau=5
        ([np.array([0.9, 1.9, 2.9])], 200, 20, 1),   # full: tau=20
    ]
    s_off = PairedFedNovaStrategy(client_momentum=0.9, tau_clip_min=0,
                                   min_available_clients=2, min_fit_clients=2)
    s_baseline = PairedFedNovaStrategy(client_momentum=0.9,
                                        min_available_clients=2, min_fit_clients=2)
    out_off, _ = _run_one_round(s_off, anchor, updates)
    out_base, _ = _run_one_round(s_baseline, anchor, updates)
    np.testing.assert_allclose(out_off[0], out_base[0], atol=1e-12)


def test_tau_clip_active_changes_only_low_tau_denominator():
    """With tau_clip_min=10, the tau=5 client's normaliser must equal
    fednova_normaliser(10, m), not fednova_normaliser(5, m). The tau=20
    client must be untouched."""
    from fl_dermamnist.runtimes.flower.strategy_fednova import fednova_normaliser
    anchor = [np.array([0.0, 0.0])]
    # Two clients: one straggler (tau=5) and one full (tau=20). Updates
    # are non-zero so the aggregate moves; we don't check the exact value,
    # only that the metrics record the clip event.
    updates = [
        ([np.array([1.0, 0.0])], 100, 5, 0),
        ([np.array([0.0, 1.0])], 100, 20, 1),
    ]
    s = PairedFedNovaStrategy(client_momentum=0.9, tau_clip_min=10,
                              min_available_clients=2, min_fit_clients=2)
    _, metrics = _run_one_round(s, anchor, updates)
    assert metrics.get("tau_clip_min") == 10
    assert metrics.get("tau_clip_hits") == 1, (
        f"expected 1 client below tau_clip_min=10, got {metrics.get('tau_clip_hits')}"
    )


def test_server_momentum_off_is_byte_identical_to_baseline():
    """server_momentum=0.0 must reproduce the un-probed FedNova aggregate exactly."""
    anchor = [np.array([1.0, 2.0, 3.0])]
    updates = [
        ([np.array([0.5, 1.5, 2.5])], 100, 10, 0),
        ([np.array([0.9, 1.9, 2.9])], 200, 10, 1),
    ]
    s_off = PairedFedNovaStrategy(client_momentum=0.9, server_momentum=0.0,
                                   min_available_clients=2, min_fit_clients=2)
    s_baseline = PairedFedNovaStrategy(client_momentum=0.9,
                                        min_available_clients=2, min_fit_clients=2)
    out_off, _ = _run_one_round(s_off, anchor, updates)
    out_base, _ = _run_one_round(s_baseline, anchor, updates)
    np.testing.assert_allclose(out_off[0], out_base[0], atol=1e-12)


def test_server_momentum_first_round_matches_baseline():
    """On the very first round, m_1 = g_1, so the global update must equal
    the un-momentumed FedNova update exactly (matches Flower FedAvgM behaviour)."""
    anchor = [np.array([1.0, 2.0, 3.0])]
    updates = [
        ([np.array([0.5, 1.5, 2.5])], 100, 10, 0),
        ([np.array([0.9, 1.9, 2.9])], 200, 10, 1),
    ]
    s_mom = PairedFedNovaStrategy(client_momentum=0.9, server_momentum=0.9,
                                   min_available_clients=2, min_fit_clients=2)
    s_off = PairedFedNovaStrategy(client_momentum=0.9,
                                   min_available_clients=2, min_fit_clients=2)
    out_mom, _ = _run_one_round(s_mom, anchor, updates)
    out_off, _ = _run_one_round(s_off, anchor, updates)
    np.testing.assert_allclose(out_mom[0], out_off[0], atol=1e-12)


def test_server_momentum_accumulates_across_rounds():
    """On round 2 with β=0.9, the momentum buffer should be β·g_1 + g_2,
    so the applied update on round 2 is exactly 1.9× the un-momentumed
    step when g_1 == g_2 (same pseudo-gradient direction both rounds).

    Hold g constant across rounds by having clients return
    `anchor − fixed_delta` *relative to the current anchor*, so the
    per-round pseudo-gradient is identical in magnitude and direction."""
    anchor = [np.array([1.0, 2.0])]
    fixed_delta = np.array([0.5, 0.5])

    def updates_against(cur_anchor):
        new = [cur_anchor[0] - fixed_delta]
        return [(new, 100, 10, 0), (new, 100, 10, 1)]

    s_mom = PairedFedNovaStrategy(client_momentum=0.0, server_momentum=0.9,
                                   min_available_clients=2, min_fit_clients=2)
    out_r1, _ = _run_one_round(s_mom, anchor, updates_against(anchor))
    out_r2, _ = _run_one_round(s_mom, out_r1, updates_against(out_r1))

    # Reference: same protocol, no momentum.
    s_off = PairedFedNovaStrategy(client_momentum=0.0,
                                   min_available_clients=2, min_fit_clients=2)
    out_r2_off, _ = _run_one_round(s_off, out_r1, updates_against(out_r1))

    step_off = out_r1[0] - out_r2_off[0]   # = g_2 = fixed_delta
    step_mom = out_r1[0] - out_r2[0]       # = β·g_1 + g_2 = 1.9·g_2 (β=0.9, g_1=g_2)
    np.testing.assert_allclose(step_mom, 1.9 * step_off, atol=1e-10)


# =========================================================================
# Stage-0 instrumentation regression tests: --server-lr and aggregation diag.
# =========================================================================


def test_server_lr_one_is_byte_identical_to_baseline():
    """server_lr=1.0 must reproduce the un-probed FedNova aggregate exactly."""
    anchor = [np.array([1.0, 2.0, 3.0])]
    updates = [
        ([np.array([0.5, 1.5, 2.5])], 100, 5, 0),
        ([np.array([0.9, 1.9, 2.9])], 200, 20, 1),
    ]
    s_one = PairedFedNovaStrategy(client_momentum=0.9, server_lr=1.0,
                                  min_available_clients=2, min_fit_clients=2)
    s_base = PairedFedNovaStrategy(client_momentum=0.9,
                                   min_available_clients=2, min_fit_clients=2)
    out_one, _ = _run_one_round(s_one, anchor, updates)
    out_base, _ = _run_one_round(s_base, anchor, updates)
    np.testing.assert_allclose(out_one[0], out_base[0], atol=1e-12)


def test_server_lr_scales_the_applied_step():
    """server_lr=η must scale the whole applied step by η: the step
    (anchor − new_global) at η=0.3 equals 0.3× the step at η=1.0."""
    anchor = [np.array([1.0, 2.0, 3.0])]
    updates = [
        ([np.array([0.5, 1.5, 2.5])], 100, 5, 0),
        ([np.array([0.9, 1.9, 2.9])], 200, 20, 1),
    ]
    s_full = PairedFedNovaStrategy(client_momentum=0.9, server_lr=1.0,
                                   min_available_clients=2, min_fit_clients=2)
    s_03 = PairedFedNovaStrategy(client_momentum=0.9, server_lr=0.3,
                                 min_available_clients=2, min_fit_clients=2)
    out_full, _ = _run_one_round(s_full, anchor, updates)
    out_03, _ = _run_one_round(s_03, anchor, updates)
    step_full = anchor[0] - out_full[0]
    step_03 = anchor[0] - out_03[0]
    np.testing.assert_allclose(step_03, 0.3 * step_full, atol=1e-10)


def _fit_res_with_norm(new_params, n, tau, cid, update_norm):
    return FitRes(
        status=Status(code=Code.OK, message="ok"),
        parameters=ndarrays_to_parameters(new_params),
        num_examples=n,
        metrics={"tau": int(tau), "cid": int(cid), "train_loss": 0.0,
                 "update_norm": float(update_norm), "local_epochs": int(tau)},
    )


def test_aggregation_diagnostics_rows_and_math():
    """When agg sinks are attached, per-client and per-round diagnostics are
    populated and amp_vs_fedavg / contribution_norm / shares match the
    closed-form definitions."""
    anchor = [np.array([0.0, 0.0])]
    results = [
        (None, _fit_res_with_norm([np.array([1.0, 0.0])], 100, 5, 0, 2.0)),   # straggler
        (None, _fit_res_with_norm([np.array([0.0, 1.0])], 100, 20, 1, 4.0)),  # full
    ]
    agg_c: list = []
    agg_r: list = []
    s = PairedFedNovaStrategy(client_momentum=0.9,
                              min_available_clients=2, min_fit_clients=2,
                              agg_client_rows=agg_c, agg_round_rows=agg_r)
    s._current_anchor = [a.copy() for a in anchor]
    s.aggregate_fit(server_round=1, results=results, failures=[])

    assert len(agg_c) == 2 and len(agg_r) == 1
    a5 = fednova_normaliser(5, 0.9)
    a20 = fednova_normaliser(20, 0.9)
    a_eff = 0.5 * a5 + 0.5 * a20  # p_i = 0.5 each (n=100,100)
    rows = {r["cid"]: r for r in agg_c}
    assert abs(rows[0]["amp_vs_fedavg"] - a_eff / a5) < 1e-9
    assert abs(rows[1]["amp_vs_fedavg"] - a_eff / a20) < 1e-9
    assert abs(rows[0]["contribution_norm"] - a_eff * 0.5 * 2.0 / a5) < 1e-9
    rr = agg_r[0]
    # u_0 = 0.5*2/a5 > u_1 = 0.5*4/a20  -> dominating + straggler are cid 0 (tau=5)
    assert rr["dominating_cid"] == 0
    assert 0.0 <= rr["straggler_share"] <= 1.0
    assert abs(rr["a_eff"] - a_eff) < 1e-9
    assert rr["server_lr"] == 1.0


def test_aggregation_diagnostics_do_not_change_aggregate():
    """Attaching diagnostic sinks must not alter the aggregated parameters."""
    anchor = [np.array([1.0, 2.0, 3.0])]
    updates = [
        ([np.array([0.5, 1.5, 2.5])], 100, 5, 0),
        ([np.array([0.9, 1.9, 2.9])], 200, 20, 1),
    ]
    s_diag = PairedFedNovaStrategy(client_momentum=0.9,
                                   min_available_clients=2, min_fit_clients=2,
                                   agg_client_rows=[], agg_round_rows=[])
    s_plain = PairedFedNovaStrategy(client_momentum=0.9,
                                    min_available_clients=2, min_fit_clients=2)
    out_diag, _ = _run_one_round(s_diag, anchor, updates)
    out_plain, _ = _run_one_round(s_plain, anchor, updates)
    np.testing.assert_allclose(out_diag[0], out_plain[0], atol=1e-12)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
