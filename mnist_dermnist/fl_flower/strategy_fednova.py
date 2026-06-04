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
    tau_clip_min : int, optional
        Lower clamp on the per-client local-step count `tau_i` used in the
        FedNova normaliser denominator. When `tau_i < tau_clip_min`, the
        client's `tau_i` is replaced by `tau_clip_min` *only* in the
        normaliser computation (the partial parameter delta still enters
        aggregation unchanged). This is a 1/τ-amplification mitigation
        proposed for random-τ straggler regimes; see the thesis §5.5
        discussion. Default 0 = OFF (byte-identical to FedNova).
    server_momentum : float, optional
        Heavy-ball server-side momentum coefficient β applied to the
        FedNova pseudo-gradient `g_t = a_eff * d_norm`. The server tracks
        a running buffer `m_t = β·m_{t-1} + g_t` and steps with `m_t`
        instead of `g_t`. Pattern mirrors flwr.server.strategy.FedAvgM
        (Hsu et al. 2019, "Measuring the Effects of Non-Identical Data
        Distribution for Federated Visual Classification"). Default 0.0
        = OFF (byte-identical to FedNova).
    All other arguments are forwarded to flwr.server.strategy.FedAvg.
    """

    def __init__(self, *args, client_momentum: float = 0.9,
                 tau_clip_min: int = 0,
                 server_momentum: float = 0.0,
                 update_norm_rows: Optional[List[Dict]] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.client_momentum = float(client_momentum)
        self.tau_clip_min = int(tau_clip_min)
        self.server_momentum = float(server_momentum)
        # Server-momentum running buffer (lazy-init on round 1). Shape
        # matches `anchor`; one ndarray per parameter tensor.
        self._momentum_buffer: Optional[NDArrays] = None
        # Diagnostic counter: number of (round, client) pairs whose tau
        # fell below tau_clip_min and triggered the clamp. Persisted in
        # the aggregated fit metrics so the runner can log it.
        self._tau_clip_hits_total: int = 0
        # Tracks the most-recent global parameters (anchor) for delta computation.
        self._current_anchor: Optional[NDArrays] = None
        # Optional sink for per-(round, client) update norms. When the runner
        # passes a list reference, aggregate_fit appends one row per
        # participating client per round. Default None = no logging
        # (zero overhead, byte-identical to the pre-flag behaviour).
        self._update_norm_rows = update_norm_rows

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

        tau_clip_hits_round = 0
        for _, fit_res in results:
            tau = float(fit_res.metrics.get("tau", 1))
            if tau <= 0:
                continue
            # --- τ-clipping (default OFF; tau_clip_min == 0). ---
            # When active, raise the τ used in the normaliser to at least
            # tau_clip_min. The partial parameter delta d_i still enters
            # aggregation unchanged; only the denominator a_i is bounded.
            # Rationale: under γ-inexact partial-update acceptance, FedNova's
            # 1/τ rescaling can blow up when a straggler returns τ ≈ 1.
            # Mathematically this is the τ analog of gradient clipping
            # (Pascanu et al. 2013); no FedNova-specific paper proposes it,
            # so this is a thesis-original mechanism probe.
            tau_for_normaliser = tau
            if self.tau_clip_min > 0 and tau < self.tau_clip_min:
                tau_for_normaliser = float(self.tau_clip_min)
                tau_clip_hits_round += 1
            a_i = fednova_normaliser(tau_for_normaliser, m)
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

            # Mechanism diagnostic: capture the client's reported update norm.
            # Clients always include it in fit metrics; we only persist when
            # the runner has provided a sink list.
            if self._update_norm_rows is not None and "update_norm" in fit_res.metrics:
                self._update_norm_rows.append({
                    "round": int(server_round),
                    "client_id": int(fit_res.metrics.get("cid", -1)),
                    "update_norm": float(fit_res.metrics["update_norm"]),
                    "n_samples": int(n),
                    "local_epochs": int(fit_res.metrics.get("local_epochs", -1)),
                    "tau": int(tau),
                })

        # FedNova pseudo-gradient: g_t = a_eff * normalised_delta. This is
        # the un-momentumed global update direction the server would apply
        # to the anchor (`w^{t+1} = w^t - g_t` in vanilla FedNova).
        pseudo_gradient: NDArrays = [
            (a_eff * d).astype(np.float64) for d in normalised_delta
        ]

        # --- Server momentum (default OFF; server_momentum == 0.0). ---
        # When active, maintain the heavy-ball buffer m_t = β·m_{t-1} + g_t
        # and use m_t in place of g_t. Mirrors flwr.server.strategy.FedAvgM
        # (Hsu et al. 2019); the first-round behaviour is m_1 = g_1 so the
        # initial step matches vanilla FedNova exactly.
        if self.server_momentum > 0.0:
            if self._momentum_buffer is None:
                self._momentum_buffer = [g.copy() for g in pseudo_gradient]
            else:
                self._momentum_buffer = [
                    self.server_momentum * mb + g
                    for mb, g in zip(self._momentum_buffer, pseudo_gradient)
                ]
            applied_update = self._momentum_buffer
        else:
            applied_update = pseudo_gradient

        # Aggregated update: w_new = w_anchor - applied_update
        new_global = [
            (anc.astype(np.float64) - upd).astype(anc.dtype)
            for anc, upd in zip(anchor, applied_update)
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
                # Mechanism-diagnostic counters (only populated when the
                # corresponding intervention is active).
                if self.tau_clip_min > 0:
                    metrics_agg["tau_clip_min"] = int(self.tau_clip_min)
                    metrics_agg["tau_clip_hits"] = int(tau_clip_hits_round)
                    self._tau_clip_hits_total += tau_clip_hits_round
                if self.server_momentum > 0.0:
                    metrics_agg["server_momentum"] = float(self.server_momentum)

        return ndarrays_to_parameters(new_global), metrics_agg
