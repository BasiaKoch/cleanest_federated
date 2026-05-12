from __future__ import annotations

import numpy as np
from torch.utils.data import WeightedRandomSampler


def get_labels(dataset):
    base = getattr(dataset, 'base', dataset)
    if hasattr(base, 'labels'):
        labels = np.asarray(base.labels).reshape(-1)
    elif hasattr(base, 'targets'):
        labels = np.asarray(base.targets).reshape(-1)
    else:
        labels = np.asarray([int(dataset[i][1]) for i in range(len(dataset))])
    return labels.astype(int)


def create_weighted_sampler(dataset, client_indices, num_classes: int) -> WeightedRandomSampler:
    labels = get_labels(dataset)
    idx = np.asarray(client_indices, dtype=int)
    local_labels = labels[idx]
    counts = np.bincount(local_labels, minlength=num_classes).astype(float)
    counts[counts == 0] = np.inf
    sample_weights = np.array([1.0 / counts[int(y)] for y in local_labels], dtype=float)
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
