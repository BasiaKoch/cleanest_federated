from __future__ import annotations

from typing import Dict, Sequence
import numpy as np
import torch
from data.download import get_labels


def compute_class_weights(class_counts: Dict[int, int] | Sequence[int], strategy: str = 'inverse', beta: float = 0.9999) -> torch.FloatTensor:
    if isinstance(class_counts, dict):
        num_classes = max(class_counts.keys()) + 1 if class_counts else 0
        counts = np.array([class_counts.get(i, 0) for i in range(num_classes)], dtype=float)
    else:
        counts = np.asarray(class_counts, dtype=float)
    original_zero = counts == 0
    safe = np.maximum(counts, 1.0)
    if strategy == 'inverse':
        weights = safe.sum() / (len(safe) * safe)
    elif strategy == 'sqrt_inverse':
        weights = np.sqrt(safe.max() / safe)
    elif strategy == 'effective':
        weights = (1 - beta) / (1 - np.power(beta, safe))
        weights = weights / weights.mean()
    else:
        raise ValueError(f'Unknown weight strategy: {strategy}')
    weights[original_zero] = 0.0
    return torch.tensor(weights, dtype=torch.float32)


def compute_local_class_weights(dataset, client_indices, num_classes: int, strategy: str = 'inverse'):
    labels = get_labels(dataset)
    local = labels[np.asarray(client_indices, dtype=int)]
    counts = np.bincount(local, minlength=num_classes)
    return compute_class_weights(counts, strategy=strategy)


def compute_global_estimated_weights(all_client_distributions_df, num_classes: int, strategy: str = 'inverse'):
    """Estimate global weights from aggregate client counts.

    This shares label-distribution metadata rather than images, which is lower
    risk than raw data sharing but still exposes class-distribution information.
    """
    counts = all_client_distributions_df.sum(axis=0).values[:num_classes]
    return compute_class_weights(counts, strategy=strategy)


if __name__ == '__main__':
    counts = [327, 514, 1099, 115, 1113, 6705, 142]
    w = compute_class_weights(counts)
    print(w)
    assert w[3] > w[5] and w[6] > w[5]
    w0 = compute_class_weights([10, 0, 5])
    assert w0[1].item() == 0.0
    import torch.nn.functional as F
    from losses.focal_loss import FocalLoss
    logits = torch.randn(8, 3)
    targets = torch.randint(0, 3, (8,))
    assert torch.allclose(FocalLoss(gamma=0.0)(logits, targets), F.cross_entropy(logits, targets))
