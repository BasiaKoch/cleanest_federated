"""Weighted average of model state_dicts (standard FedAvg aggregation).

Used by both FedAvg and FedProx (the proximal term affects only the LOCAL
objective, not the aggregation rule).
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import torch


def weighted_average_state_dicts(
    state_dicts: Sequence[Dict[str, torch.Tensor]],
    weights: Sequence[float],
) -> Dict[str, torch.Tensor]:
    """Compute weighted average over a list of state_dicts.

    For floating-point tensors, applies the size-weighted convex combination.
    For integer buffers (rare but possible — GroupNorm has none, our model has
    none, so this is defensive), takes the value from the FIRST client
    (deterministic fallback).
    """
    if len(state_dicts) == 0:
        raise ValueError("weighted_average_state_dicts: empty input")
    if len(weights) != len(state_dicts):
        raise ValueError(
            f"weights and state_dicts length mismatch: {len(weights)} vs {len(state_dicts)}"
        )
    total = float(sum(weights))
    if total <= 0:
        raise ValueError(f"weights must sum to a positive number; got {total}")
    norm_weights = [w / total for w in weights]

    out: Dict[str, torch.Tensor] = {}
    template = state_dicts[0]
    for key, ref in template.items():
        if ref.dtype.is_floating_point or torch.is_complex(ref):
            acc = torch.zeros_like(ref)
            for w, sd in zip(norm_weights, state_dicts):
                acc = acc + w * sd[key]
            out[key] = acc
        else:
            # Integer-typed buffer (none expected for DermMNISTCNN). Deterministic fallback.
            out[key] = ref.clone()
    return out
