"""DermMNISTCNN_BN — BatchNorm variant of DermMNISTCNN for an architecture ablation.

Identical to ``DermMNISTCNN`` in every respect except that the four GroupNorm
layers are replaced by ``nn.BatchNorm2d``. This is the canonical FL-unfriendly
normalization choice: BN's per-batch statistics (running mean/variance) drift
across heterogeneous federated clients and the standard FedAvg aggregator
averages those buffers indiscriminately, which is known to degrade non-IID
performance (Li et al. 2021, "FedBN"; Hsieh et al. 2020, "Non-IID Quagmire").

Purpose: an architecture-normalization ablation. The hypothesis is that
FedProx's parameter-side proximal anchor partially compensates for BN-induced
drift, so the FedProx-over-FedAvg gap should *widen* on the BN variant
relative to the GN headline. This ablation is meant to be run on the
engineered ``balanced_paired_7_clients`` partition at the same hyperparameters
as the headline (E=20, R=150, μ=0.01) for a small subset of paired seeds.

Architecture (identical layer-by-layer to ``DermMNISTCNN`` except norm type):
    Conv(3 → 32)    + BatchNorm2d(32)  + ReLU + MaxPool
    Conv(32 → 64)   + BatchNorm2d(64)  + ReLU + MaxPool
    Conv(64 → 128)  + BatchNorm2d(128) + ReLU + MaxPool
    Conv(128 → 256) + BatchNorm2d(256) + ReLU + AdaptiveAvgPool
    Flatten
    Linear(256 → 128) + ReLU + Dropout(0.2)
    Linear(128 → 7)

Parameter count is essentially identical to the GN variant — the BN affine
parameters (gamma, beta) match GN's, and the BN running buffers (running_mean,
running_var) are not trainable but are aggregated by FedAvg's state-dict
averaging (which is the very mechanism whose interaction with non-IID data
this ablation probes).

Works on 28×28 (and any other size ≥ 8×8) RGB input.
"""
from __future__ import annotations

import torch
from torch import nn


class DermMNISTCNN_BN(nn.Module):
    """BatchNorm variant of the headline DermMNISTCNN.

    See module-level docstring for the FL motivation. The only structural
    difference from ``DermMNISTCNN`` is the substitution of every GroupNorm
    layer with a ``BatchNorm2d`` with matching channel count.
    """

    def __init__(self, num_classes: int = 7, dropout: float = 0.2):
        super().__init__()
        # Block 1: 3 -> 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(32)
        # Block 2: 32 -> 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(64)
        # Block 3: 64 -> 128
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.norm3 = nn.BatchNorm2d(128)
        # Block 4: 128 -> 256
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.norm4 = nn.BatchNorm2d(256)

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.norm1(self.conv1(x))))
        x = self.pool(self.relu(self.norm2(self.conv2(x))))
        x = self.pool(self.relu(self.norm3(self.conv3(x))))
        x = self.gap(self.relu(self.norm4(self.conv4(x))))
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    m = DermMNISTCNN_BN()
    total = sum(p.numel() for p in m.parameters())
    print(f"DermMNISTCNN_BN — parameters: {total:,}")
    for size in (28, 32, 64, 128):
        x = torch.randn(2, 3, size, size)
        print(f"  input {x.shape} -> output {m(x).shape}")
