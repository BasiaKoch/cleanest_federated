"""StragglerDroppingFedAvg — Li et al. 2020 §5.2 FedAvg protocol.

The original FedProx paper (Li, Sahu, Talwalkar, Smith; MLSys 2020) §5.2
evaluates FedProx against FedAvg under an **asymmetric** aggregation
protocol that is widely overlooked but central to FedProx's headline
``+22%'' advantage on synthetic data:

  - **FedAvg** drops straggler clients entirely. A straggler is a client
    that completed fewer than ``E_max`` local epochs in a given round
    (``γ``-inexact in the paper's terminology). The straggler's
    parameter update is discarded; aggregation uses only the
    non-straggler subset.
  - **FedProx** includes stragglers as ``γ``-inexact contributors.
    The proximal anchor provides the stability guarantee that allows
    partial-work updates to be aggregated safely.

This strategy implements the FedAvg side of that protocol. Combine it
with an unmodified (proximal-anchored) FedProx run for the canonical
Li 2020 comparison.

Implementation
--------------
Subclasses Flower's ``FedAvg`` strategy. Overrides ``aggregate_fit`` to
filter out client updates whose reported ``local_epochs`` metric is
strictly less than ``E_max``. The filtered subset is then handed off
to the parent's aggregation (size-weighted parameter mean).

Edge cases
----------
- If every client in a round is a straggler, the strategy returns
  ``(None, {...})``, causing Flower's server to skip the parameter
  update for that round (the global model is unchanged until the
  next round produces at least one non-straggler).
- If no clients are stragglers (e.g., under uniform compute mode),
  the strategy behaves exactly like Flower's ``FedAvg`` — the filter
  is a no-op.

This strategy is intentionally limited to FedAvg-style aggregation.
For FedNova-style normalised aggregation under stragglers, a different
strategy is required.

Methodological caveat
---------------------
Flower waits for ALL clients to finish training before invoking
``aggregate_fit``; this strategy then DISCARDS straggler updates from
the aggregation. This faithfully models the *algorithmic* effect of
straggler dropping on accuracy-per-round, but it does NOT model
wall-clock or deadline realism: in a true deadline-bounded deployment,
stragglers would be cut off before completing local training.
Thesis claims should be framed accordingly: this measures the
*aggregation-policy* effect of dropping γ-inexact updates, not the
operational benefit of not waiting for stragglers.

Methodological confound (decomposition note)
--------------------------------------------
When this strategy is used for FedAvg while FedProx aggregates all
clients (the Li 2020 §5.2 comparison), the two algorithms see
DIFFERENT client subsets per round. If FedProx wins, the win
decomposes into two sources:
  (a) FedProx sees more clients per round (those that FedAvg drops);
  (b) the proximal anchor stabilises γ-inexact updates.
The pure proximal effect is given by the symmetric arm — FedProx vs
FedAvg with both algorithms seeing all clients (``system_het_random/``
in this codebase). The asymmetric vs symmetric contrast isolates the
straggler-handling component, while the symmetric arm isolates the
proximal-term component.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from flwr.common import FitRes, Parameters
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class StragglerDroppingFedAvg(FedAvg):
    """FedAvg that drops stragglers (local_epochs < E_max) before aggregation.

    Matches Li et al. 2020 §5.2 FedAvg protocol. Combine with an
    unmodified FedProx run (which includes all clients) to evaluate
    FedProx's γ-inexact-update tolerance.

    Parameters
    ----------
    E_max : int
        The maximum local-epoch budget. Clients reporting
        ``local_epochs < E_max`` in their fit metrics are filtered out
        of the aggregation. All other ``FedAvg`` arguments are
        forwarded.
    """

    def __init__(self, *args, E_max: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.E_max = int(E_max)
        # Diagnostic counters for thesis reporting.
        self._cumulative_dropped = 0
        self._cumulative_kept = 0

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List,
    ) -> Tuple[Optional[Parameters], Dict]:
        if not results:
            return None, {}

        kept: List[Tuple[ClientProxy, FitRes]] = []
        dropped_cids: List[int] = []
        for client_proxy, fit_res in results:
            # Reject any client that did not report local_epochs.
            # Silent fallback to E_max would mask broken client wiring and
            # turn straggler dropping into a no-op without warning. For
            # this special strategy we require explicit metadata.
            if "local_epochs" not in fit_res.metrics:
                raise KeyError(
                    f"StragglerDroppingFedAvg requires every client to "
                    f"report 'local_epochs' in fit metrics (got keys: "
                    f"{list(fit_res.metrics.keys())}). Refusing to fall "
                    f"back to E_max={self.E_max} silently."
                )
            client_E = int(fit_res.metrics["local_epochs"])
            cid = int(fit_res.metrics.get("cid", -1))
            if client_E < self.E_max:
                dropped_cids.append(cid)
                self._cumulative_dropped += 1
            else:
                kept.append((client_proxy, fit_res))
                self._cumulative_kept += 1

        # Edge case: all clients are stragglers this round
        if not kept:
            return None, {
                "n_kept": 0,
                "n_dropped": len(dropped_cids),
                "dropped_cids": ",".join(map(str, dropped_cids)),
            }

        # Delegate to FedAvg's size-weighted aggregation on the filtered subset
        params, metrics = super().aggregate_fit(server_round, kept, failures)
        # Augment metrics with straggler-drop diagnostics
        metrics = dict(metrics or {})
        metrics["n_kept"] = len(kept)
        metrics["n_dropped"] = len(dropped_cids)
        metrics["dropped_cids"] = ",".join(map(str, dropped_cids))
        return params, metrics
