"""Regression tests for paired client RNG seeding."""
from __future__ import annotations

import numpy as np
import torch

from fl_dermamnist.core.seeding import dataloader_generator_seed, numpy_legacy_seed


def test_large_paired_seed_keeps_raw_torch_seed_but_numpy_is_32_bit() -> None:
    raw = dataloader_generator_seed(base_seed=8_675_309, round_num=1, client_id=0)

    assert raw == 86_753_090_100
    assert raw > 2**32 - 1

    np_seed = numpy_legacy_seed(raw)
    assert 0 <= np_seed <= 2**32 - 1

    # The whole bug: NumPy's legacy seed rejects the raw value, while the
    # helper-produced value is accepted. Torch still receives the raw seed to
    # preserve equivalence with the pure-PyTorch reference loop.
    try:
        np.random.seed(raw)
    except ValueError:
        pass
    else:  # pragma: no cover - this would mean NumPy changed its contract
        raise AssertionError("np.random.seed unexpectedly accepted an oversized seed")

    np.random.seed(np_seed)
    torch.Generator().manual_seed(raw)


def test_seed_mapping_is_collision_free_for_registered_hpc_grid() -> None:
    seeds = [42, 123, 456, 789, 999, 2024, 31337, 8_675_309, 161803, 271828]
    seen: set[int] = set()

    for seed in seeds:
        for round_num in range(1, 151):
            for cid in range(7):
                raw = dataloader_generator_seed(seed, round_num, cid)
                np_seed = numpy_legacy_seed(raw)
                assert np_seed not in seen
                seen.add(np_seed)
