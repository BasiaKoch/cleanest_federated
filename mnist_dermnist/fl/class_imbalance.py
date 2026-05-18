"""Class-imbalance-aware loss functions for the imbalance-baseline experiment.

Two loss variants are provided, both drop-in replacements for the
standard cross-entropy in `mnist_dermnist.fl.local_train`:

  - `make_class_weighted_ce(labels)`: inverse-frequency class weights
    \\propto 1/n_c, normalised so weights sum to num_classes. The
    canonical "FedAvg + class-weighted CE" baseline.
  - `make_focal_loss(gamma)`: focal loss with focusing parameter γ
    \\citep{Lin et al., 2017}. γ=2 is the default.

The class-weighted CE baseline directly addresses the reviewer concern
that FedProx is an optimisation-level remedy for client drift, not a
class-imbalance correction. By comparing FedAvg vs FedAvg+CW-CE vs
FedProx on the same partition, the thesis can quantify whether
loss-level imbalance correction is a competitive simpler alternative.

These functions take the *training-set* class frequencies as input. In
the FL setting we use the GLOBAL training set's frequencies, computed
once before the experiment and shared with every client (this is
trivially derivable from the partition and not a privacy concern in
simulation).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def class_weights_inverse_freq(
    labels: Sequence[int],
    num_classes: int = 7,
) -> torch.Tensor:
    """Inverse-frequency class weights, normalised to sum to num_classes.

    w_c = (N / n_c) * (1 / num_classes), with safety floor on n_c = 1.
    """
    arr = np.asarray(labels)
    counts = np.bincount(arr, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)  # avoid div-by-zero
    inv_freq = 1.0 / counts
    # Normalise so the weight vector sums to num_classes (a common
    # convention; an alternative is to normalise to mean 1).
    inv_freq = inv_freq * (num_classes / inv_freq.sum())
    return torch.tensor(inv_freq, dtype=torch.float32)


def make_class_weighted_ce(
    labels: Sequence[int],
    num_classes: int = 7,
    device: str = "cpu",
) -> nn.Module:
    """Returns a CrossEntropyLoss configured with inverse-frequency weights."""
    w = class_weights_inverse_freq(labels, num_classes=num_classes).to(device)
    return nn.CrossEntropyLoss(weight=w)


class FocalLoss(nn.Module):
    """Multi-class focal loss \\citep{Lin et al., 2017}.

    L = - alpha_c * (1 - p_c)^gamma * log(p_c)

    where p_c is the predicted probability of the true class c. For
    multi-class classification with no class weighting, we set
    alpha_c = 1 for all c (the focusing parameter γ alone controls the
    relative weighting between easy and hard examples).
    """
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = alpha  # tensor of shape [num_classes] or None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.alpha, reduction="none")
        p_t = torch.exp(-ce)
        focal = (1.0 - p_t) ** self.gamma * ce
        return focal.mean()


def make_focal_loss(
    gamma: float = 2.0,
    labels: Sequence[int] | None = None,
    num_classes: int = 7,
    device: str = "cpu",
) -> nn.Module:
    """Returns a FocalLoss; if `labels` provided, uses inverse-freq alpha."""
    alpha = None
    if labels is not None:
        alpha = class_weights_inverse_freq(labels, num_classes=num_classes).to(device)
    return FocalLoss(gamma=gamma, alpha=alpha)
