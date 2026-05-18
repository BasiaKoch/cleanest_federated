"""FedNova aggregation strategy (Wang et al., 2020, NeurIPS).

Subclasses Flower's FedAvg strategy to apply the normalised-averaging
rule of FedNova on heterogeneous local-step counts. For uniform local
step counts AND zero momentum the strategy reduces exactly to FedAvg.

Aggregation rule (Wang et al., 2020, Algorithm 1 + §3.3)
--------------------------------------------------------
For SGD with momentum coefficient `m` and local step count `tau_i` at
client i, FedNova defines the per-step coefficient vector

    a_vec = [a^{(1)}, a^{(2)}, ..., a^{(tau_i)}]
    a^{(j)} = sum_{k=0}^{j-1} m^k = (1 - m^j) / (1 - m)

i.e. the L1 norm of the cumulative momentum series after `j` local
steps. The FedNova normaliser is its L1 norm:

    a_i = ||a_vec||_1
        = sum_{j=1}^{tau_i} (1 - m^j) / (1 - m)
        = ( tau_i (1 - m) - m (1 - m^{tau_i}) ) / (1 - m)^2     (closed form)
        = tau_i                                if m == 0  (vanilla SGD)

CAUTION: a common mistake (made by an earlier version of this file)
is to use a^{(tau_i)} = (1 - m^{tau_i}) / (1 - m) directly. That is the
LAST element of `a_vec`, not its L1 norm; it gives the wrong
normalisation under momentum. The correct closed form above is the
sum of the geometric-cumulative series. Reference values used in the
unit tests below:

    m = 0,    tau = 10:  a_i = 10
    m = 0.9,  tau = 1:   a_i = 1
    m = 0.9,  tau = 5:   a_i = 13.1441    (NOT 4.0951)
    m = 0.9,  tau = 10:  a_i = 41.3811    (NOT 6.5132)
    m = 0.9,  tau = 100: a_i ≈ 909.999

The aggregation step is then::

    d_i      = w^t - w_i^{t+1}              (parameter delta from anchor)
    d_norm   = sum_i p_i * d_i / a_i        (normalised, weighted average)
    a_eff    = sum_i p_i * a_i              (effective coupling constant)
    w^{t+1}  = w^t - a_eff * d_norm         (global update)

where `p_i = n_i / sum_j n_j` is the standard sample-size weight.

Reduction to FedAvg under uniform tau and m=0
---------------------------------------------
If all `tau_i = T` and `m = 0`, then `a_i = T` for all i, hence
`a_eff = T`, and `d_norm = (1/T) * sum_i p_i d_i`. The global update is
`w^t - T * (1/T) sum p_i d_i = w^t - sum p_i d_i`, which is exactly
FedAvg's size-weighted-mean aggregation of parameters. So the strategy
is safe to use in any condition, including the no-heterogeneity
baseline, for sanity-check comparisons.

Note: under momentum > 0, FedNova and FedAvg differ even when tau is
uniform, because the normaliser is no longer equal to tau. The
intuition is that momentum makes each local step contribute weighted
gradient information from earlier steps, so the "effective work"
accumulates super-linearly in tau.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
from flwr.common import (
    EvaluateRes, FitRes, NDArrays, Parameters,
    ndarrays_to_parameters, parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


def fednova_normaliser(tau: float, momentum: float) -> float:
    """Compute the FedNova L1 normaliser ||a_vec||_1.

    For local SGD with momentum coefficient `m` and `tau` local steps:

        a_i = sum_{j=1}^{tau} (1 - m^j) / (1 - m)
            = ( tau (1 - m) - m (1 - m^tau) ) / (1 - m)^2

    Reduces to `tau` when m = 0 (vanilla SGD).

    Reference (Wang et al. 2020, Algorithm 1; canonical FedNova
    implementation `JYWa/FedNova`). The earlier (incorrect) version of
    this function returned a^{(tau)} = (1 - m^tau) / (1 - m), which is
    the last entry of a_vec, NOT its L1 norm.
    """
    tau = float(tau)
    m = float(momentum)
    if m <= 0.0:
        return tau
    if abs(1.0 - m) < 1e-12:
        # Limit case m -> 1: a^{(j)} -> j, so sum_j j = tau(tau+1)/2
        return tau * (tau + 1.0) / 2.0
    # Closed-form L1 norm of the cumulative-geometric vector a_vec.
    return (tau * (1.0 - m) - m * (1.0 - m ** tau)) / ((1.0 - m) ** 2)


class PairedFedNovaStrategy(FedAvg):
    """FedAvg subclass implementing the FedNova aggregation rule.

    Parameters
    ----------
    client_momentum : float
        The momentum coefficient used by the local SGD optimiser on every
        client (must be the SAME across all clients for the FedNova
        derivation to apply). Default 0.9, matching the rest of the
        thesis. Set to 0.0 for vanilla-SGD experiments.
    All other arguments are forwarded to flwr.server.strategy.FedAvg.
    """

    def __init__(self, *args, client_momentum: float = 0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.client_momentum = float(client_momentum)
        # Tracks the most-recent global parameters (anchor) for delta computation.
        self._current_anchor: Optional[NDArrays] = None

    def initialize_parameters(self, client_manager):
        params = super().initialize_parameters(client_manager)
        if params is not None:
            self._current_anchor = parameters_to_ndarrays(params)
        return params

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List,
    ) -> Tuple[Optional[Parameters], Dict]:
        if not results:
            return None, {}
        if self._current_anchor is None:
            # Shouldn't happen, but fall back to FedAvg if no anchor recorded
            return super().aggregate_fit(server_round, results, failures)

        anchor = self._current_anchor
        total_n = sum(r.num_examples for _, r in results)
        m = self.client_momentum  # local-optimiser momentum, same across clients

        # Compute weighted-mean of (delta_i / a_i) across clients, where
        # a_i is the FedNova momentum-aware normaliser (Wang 2020, §3.3).
        normalised_delta = [np.zeros_like(a, dtype=np.float64) for a in anchor]
        a_eff = 0.0
        per_client_a: List[Tuple[int, float, float]] = []  # (cid, tau, a_i) for logging

        for _, fit_res in results:
            tau = float(fit_res.metrics.get("tau", 1))
            if tau <= 0:
                continue
            a_i = fednova_normaliser(tau, m)        # ← MF2: momentum-aware
            if a_i <= 0:
                continue
            n = fit_res.num_examples
            p_i = n / total_n
            new_params = parameters_to_ndarrays(fit_res.parameters)
            for k, (anc, p) in enumerate(zip(anchor, new_params)):
                d_i = anc.astype(np.float64) - p.astype(np.float64)
                normalised_delta[k] += p_i * (d_i / a_i)
            a_eff += p_i * a_i
            per_client_a.append((int(fit_res.metrics.get("cid", -1)), tau, a_i))

        # Aggregated update: w_new = w_anchor - a_eff * normalised_delta
        new_global = [
            (anc.astype(np.float64) - a_eff * d).astype(anc.dtype)
            for anc, d in zip(anchor, normalised_delta)
        ]
        self._current_anchor = new_global

        # Aggregate fit metrics (size-weighted mean of train_loss, as in FedAvg)
        metrics_agg = {}
        if results:
            total_n_local = sum(r.num_examples for _, r in results)
            if total_n_local > 0:
                metrics_agg["train_loss"] = sum(
                    r.metrics.get("train_loss", 0.0) * r.num_examples
                    for _, r in results
                ) / total_n_local
                metrics_agg["mean_tau"] = sum(
                    r.metrics.get("tau", 0.0) * r.num_examples
                    for _, r in results
                ) / total_n_local
                metrics_agg["a_eff"] = float(a_eff)
                metrics_agg["client_momentum"] = float(m)

        return ndarrays_to_parameters(new_global), metrics_agg
