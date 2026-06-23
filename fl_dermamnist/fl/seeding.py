"""Shared deterministic seed helpers for paired FL runs."""
from __future__ import annotations


def dataloader_generator_seed(base_seed: int, round_num: int, client_id: int) -> int:
    """Per-(seed, round, client) RNG seed, paired across runtimes."""
    return int(base_seed) * 10_000 + int(round_num) * 100 + int(client_id)


def numpy_legacy_seed(raw_seed: int) -> int:
    """Map an arbitrary integer to NumPy RandomState's 32-bit seed range.

    ``np.random.seed`` accepts only values in ``[0, 2**32 - 1]``. PyTorch's
    manual seed and Python's ``random.seed`` can consume the larger raw seed,
    so callers should use this helper only for NumPy.
    """
    return int(raw_seed) & 0xFFFFFFFF
