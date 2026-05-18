"""System heterogeneity utilities — per-client per-round local-epoch schedules.

The original FedProx motivation (Li et al., 2020, MLSys) was system
heterogeneity: clients have heterogeneous compute capability, so cannot all
complete the same amount of local work per round. The proximal term
$(\\mu/2)\\|w - w^t\\|^2$ allows clients to submit `γ`-inexact updates --
partial work that doesn't fully minimise the local objective -- while
provably maintaining convergence.

This module generates per-client per-round local-epoch schedules to
simulate stragglers in three modes corresponding to canonical experimental
designs:

  - "uniform":           every client performs E_max epochs every round
                         (the no-system-het baseline).
  - "fixed_stragglers":  a deterministic subset of clients are always slow,
                         performing E_straggler epochs instead of E_max.
                         (Closer to Marija 2025 §3.8.4; predictable but
                         less realistic.)
  - "random_stragglers": every round, half of the participating clients are
                         randomly designated stragglers, with E_i drawn
                         uniformly from [1, E_max - 1]. The other half
                         perform E_max. This follows Li et al. (2020) §5.2
                         exactly.

All schedules are deterministic given a seed, so paired FedAvg/FedProx
runs share an identical schedule and the within-pair Δ reflects only the
algorithmic difference.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class SystemHetConfig:
    """Specification of a system-heterogeneity scenario.

    Attributes
    ----------
    mode : str
        One of "uniform", "fixed_stragglers", "random_stragglers".
    E_max : int
        Maximum local epochs (the non-straggler budget).
    E_straggler : int
        Local epochs for stragglers in "fixed_stragglers" mode.
        Ignored in other modes.
    fixed_straggler_ids : list[int] or None
        For "fixed_stragglers": which client ids are always stragglers.
        Defaults to the last two clients if None.
    random_straggler_fraction : float
        For "random_stragglers": fraction of participating clients per round
        that are designated stragglers. Default 0.5 (Li et al.).
    random_straggler_min_epochs : int
        For "random_stragglers": minimum E_i for a straggler (default 1).
    random_straggler_max_epochs : int or None
        For "random_stragglers": maximum E_i (exclusive of E_max). Default
        E_max - 1.
    """
    mode: str = "uniform"
    E_max: int = 20
    E_straggler: int = 5
    fixed_straggler_ids: Optional[List[int]] = None
    random_straggler_fraction: float = 0.5
    random_straggler_min_epochs: int = 1
    random_straggler_max_epochs: Optional[int] = None

    def to_dict(self) -> Dict:
        return {
            "mode": self.mode,
            "E_max": self.E_max,
            "E_straggler": self.E_straggler,
            "fixed_straggler_ids": self.fixed_straggler_ids,
            "random_straggler_fraction": self.random_straggler_fraction,
            "random_straggler_min_epochs": self.random_straggler_min_epochs,
            "random_straggler_max_epochs": self.random_straggler_max_epochs,
        }


def build_epoch_schedule(
    cfg: SystemHetConfig,
    num_clients: int,
    num_rounds: int,
    seed: int,
) -> np.ndarray:
    """Build a (num_rounds, num_clients) int array of per-(round, cid) local epochs.

    The schedule is fully determined by `seed`, so paired runs across
    algorithms share the same schedule.

    Returns
    -------
    np.ndarray of shape (num_rounds, num_clients) with dtype int.
    Entry [r, k] is the local-epoch count for client k in (1-based) round r+1.
    """
    if cfg.mode == "uniform":
        return np.full((num_rounds, num_clients), cfg.E_max, dtype=int)

    if cfg.mode == "fixed_stragglers":
        straggler_ids = (
            list(range(num_clients - 2, num_clients))
            if cfg.fixed_straggler_ids is None
            else list(cfg.fixed_straggler_ids)
        )
        if any(s < 0 or s >= num_clients for s in straggler_ids):
            raise ValueError(
                f"fixed_straggler_ids contains out-of-range id "
                f"(num_clients={num_clients}, ids={straggler_ids})"
            )
        schedule = np.full((num_rounds, num_clients), cfg.E_max, dtype=int)
        for k in straggler_ids:
            schedule[:, k] = cfg.E_straggler
        return schedule

    if cfg.mode == "random_stragglers":
        # Use a dedicated RNG to avoid interfering with model/data RNGs.
        # Offset is large and odd so it doesn't collide with the offsets used
        # for client sampling and dataloader generators in server_loop.py.
        rng = np.random.default_rng(seed=seed + 7_000_003)
        n_stragglers = int(round(cfg.random_straggler_fraction * num_clients))
        max_e = cfg.random_straggler_max_epochs or (cfg.E_max - 1)
        if max_e >= cfg.E_max:
            raise ValueError(
                f"random_straggler_max_epochs ({max_e}) must be < E_max ({cfg.E_max})"
            )
        if cfg.random_straggler_min_epochs < 1:
            raise ValueError("random_straggler_min_epochs must be >= 1")

        schedule = np.full((num_rounds, num_clients), cfg.E_max, dtype=int)
        for r in range(num_rounds):
            stragglers = rng.choice(num_clients, size=n_stragglers, replace=False)
            for k in stragglers:
                # Uniform in [min_epochs, max_e] inclusive
                schedule[r, k] = int(rng.integers(
                    low=cfg.random_straggler_min_epochs,
                    high=max_e + 1,
                ))
        return schedule

    raise ValueError(f"Unknown system-het mode: {cfg.mode!r}")


def summarise_schedule(schedule: np.ndarray) -> Dict:
    """Compute summary statistics of an epoch schedule for logging."""
    return {
        "shape": list(schedule.shape),
        "mean_epochs_per_client_round": float(schedule.mean()),
        "min_epochs": int(schedule.min()),
        "max_epochs": int(schedule.max()),
        "per_client_mean_epochs": schedule.mean(axis=0).tolist(),
        "frac_straggler_rounds": float((schedule < schedule.max()).mean()),
    }
