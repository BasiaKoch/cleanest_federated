"""Tests for StragglerDroppingFedAvg strategy.

Verifies the strategy correctly filters client updates whose reported
``local_epochs`` is strictly less than ``E_max``, implementing the
Li et al. 2020 §5.2 FedAvg-drops-stragglers protocol.
"""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock

from flwr.common import (
    Code, FitRes, ndarrays_to_parameters, parameters_to_ndarrays, Status,
)

from fl_dermamnist.runtimes.flower.strategy_straggler_dropping import (
    StragglerDroppingFedAvg,
)


def make_fit_res(cid: int, local_epochs: int, num_examples: int = 100,
                 param_value: float = 1.0):
    """Build a minimal FitRes for testing."""
    # Two-parameter dummy: a conv weight and a bias
    params = ndarrays_to_parameters(
        [np.full((4, 4), param_value, dtype=np.float32),
         np.full((4,), param_value, dtype=np.float32)]
    )
    return FitRes(
        status=Status(code=Code.OK, message=""),
        parameters=params,
        num_examples=num_examples,
        metrics={"cid": cid, "local_epochs": local_epochs},
    )


def make_initial_parameters():
    return ndarrays_to_parameters(
        [np.zeros((4, 4), dtype=np.float32),
         np.zeros((4,), dtype=np.float32)]
    )


def test_drops_stragglers_below_E_max():
    """Clients with local_epochs < E_max are dropped from aggregation."""
    strat = StragglerDroppingFedAvg(
        E_max=20,
        initial_parameters=make_initial_parameters(),
    )
    # 3 non-stragglers (E=20) and 2 stragglers (E=5, E=12)
    results = [
        (MagicMock(), make_fit_res(cid=0, local_epochs=20, param_value=1.0)),
        (MagicMock(), make_fit_res(cid=1, local_epochs=20, param_value=1.0)),
        (MagicMock(), make_fit_res(cid=2, local_epochs=20, param_value=1.0)),
        (MagicMock(), make_fit_res(cid=3, local_epochs=5,  param_value=99.0)),
        (MagicMock(), make_fit_res(cid=4, local_epochs=12, param_value=99.0)),
    ]
    params, metrics = strat.aggregate_fit(server_round=1, results=results,
                                           failures=[])
    assert params is not None
    arrays = parameters_to_ndarrays(params)
    # Aggregate should reflect only the 3 non-stragglers (value 1.0)
    # If stragglers were included, value would be ~40 (mean of [1,1,1,99,99])
    assert abs(arrays[0].mean() - 1.0) < 1e-5, (
        f"aggregate contaminated by stragglers; mean={arrays[0].mean()}"
    )
    assert metrics["n_kept"] == 3
    assert metrics["n_dropped"] == 2
    assert "3" in metrics["dropped_cids"]
    assert "4" in metrics["dropped_cids"]


def test_no_stragglers_behaves_like_FedAvg():
    """When all clients are non-stragglers, the strategy is a no-op."""
    strat = StragglerDroppingFedAvg(
        E_max=20,
        initial_parameters=make_initial_parameters(),
    )
    results = [
        (MagicMock(), make_fit_res(cid=0, local_epochs=20, param_value=2.0,
                                    num_examples=100)),
        (MagicMock(), make_fit_res(cid=1, local_epochs=20, param_value=4.0,
                                    num_examples=100)),
    ]
    params, metrics = strat.aggregate_fit(server_round=1, results=results,
                                           failures=[])
    assert params is not None
    arrays = parameters_to_ndarrays(params)
    # Equal num_examples => simple mean = 3.0
    assert abs(arrays[0].mean() - 3.0) < 1e-5
    assert metrics["n_kept"] == 2
    assert metrics["n_dropped"] == 0


def test_all_stragglers_returns_none():
    """When every client is a straggler, return None (skip the round)."""
    strat = StragglerDroppingFedAvg(
        E_max=20,
        initial_parameters=make_initial_parameters(),
    )
    results = [
        (MagicMock(), make_fit_res(cid=0, local_epochs=5)),
        (MagicMock(), make_fit_res(cid=1, local_epochs=10)),
    ]
    params, metrics = strat.aggregate_fit(server_round=1, results=results,
                                           failures=[])
    assert params is None
    assert metrics["n_kept"] == 0
    assert metrics["n_dropped"] == 2


def test_empty_results_returns_none():
    """Empty results (Flower edge case) returns None gracefully."""
    strat = StragglerDroppingFedAvg(
        E_max=20,
        initial_parameters=make_initial_parameters(),
    )
    params, metrics = strat.aggregate_fit(server_round=1, results=[],
                                           failures=[])
    assert params is None


def test_exactly_E_max_is_NOT_a_straggler():
    """Boundary: a client with local_epochs == E_max is kept (not dropped)."""
    strat = StragglerDroppingFedAvg(
        E_max=20,
        initial_parameters=make_initial_parameters(),
    )
    results = [
        (MagicMock(), make_fit_res(cid=0, local_epochs=20, param_value=1.0)),
        (MagicMock(), make_fit_res(cid=1, local_epochs=19, param_value=99.0)),
    ]
    params, metrics = strat.aggregate_fit(server_round=1, results=results,
                                           failures=[])
    arrays = parameters_to_ndarrays(params)
    assert abs(arrays[0].mean() - 1.0) < 1e-5
    assert metrics["n_kept"] == 1
    assert metrics["n_dropped"] == 1


def test_raises_on_missing_local_epochs():
    """A broken client metric (missing local_epochs) must NOT silently
    fall back to E_max; that would turn straggler dropping into a no-op."""
    strat = StragglerDroppingFedAvg(
        E_max=20,
        initial_parameters=make_initial_parameters(),
    )
    # Build a FitRes WITHOUT local_epochs in metrics
    params = ndarrays_to_parameters(
        [np.full((4, 4), 1.0, dtype=np.float32),
         np.full((4,), 1.0, dtype=np.float32)]
    )
    broken_res = FitRes(
        status=Status(code=Code.OK, message=""),
        parameters=params,
        num_examples=100,
        metrics={"cid": 0},  # no local_epochs key
    )
    with pytest.raises(KeyError, match="local_epochs"):
        strat.aggregate_fit(
            server_round=1,
            results=[(MagicMock(), broken_res)],
            failures=[],
        )


def test_cumulative_counters_track_across_rounds():
    """Cumulative counters should accumulate over rounds for diagnostics."""
    strat = StragglerDroppingFedAvg(
        E_max=20,
        initial_parameters=make_initial_parameters(),
    )
    results = [
        (MagicMock(), make_fit_res(cid=0, local_epochs=20)),
        (MagicMock(), make_fit_res(cid=1, local_epochs=10)),
    ]
    strat.aggregate_fit(server_round=1, results=results, failures=[])
    strat.aggregate_fit(server_round=2, results=results, failures=[])
    assert strat._cumulative_kept == 2
    assert strat._cumulative_dropped == 2
