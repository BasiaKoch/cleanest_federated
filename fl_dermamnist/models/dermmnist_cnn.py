"""DermMNISTCNN — headline FedAvg/FedProx model.

Architecture:
    Conv(3 → 32)   + GroupNorm(4, 32)   + ReLU + MaxPool
    Conv(32 → 64)  + GroupNorm(8, 64)   + ReLU + MaxPool
    Conv(64 → 128) + GroupNorm(16, 128) + ReLU + MaxPool
    Conv(128 → 256)+ GroupNorm(16, 256) + ReLU + AdaptiveAvgPool
    Flatten
    Linear(256 → 128) + ReLU + Dropout(0.2)
    Linear(128 → 7)

GroupNorm (not BatchNorm) is used because BN's per-batch statistics diverge
across heterogeneous federated clients — its running buffers leak distribution
information between rounds and bias the aggregated model. GroupNorm has no
running statistics and is computed independently per sample, making it the
recommended normalization for FL (see Hsieh et al. 2020 "The Non-IID Data
Quagmire of Decentralized ML").

No pretrained weights. No BatchNorm anywhere. Works on 28×28 (and any other
resolution, thanks to AdaptiveAvgPool2d) RGB input.
"""
from __future__ import annotations

import torch
from torch import nn


class DermMNISTCNN(nn.Module):
    def __init__(self, num_classes: int = 7, dropout: float = 0.2):
        super().__init__()
        # Block 1: 3 -> 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups=4, num_channels=32)
        # Block 2: 32 -> 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=64)
        # Block 3: 64 -> 128
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.norm3 = nn.GroupNorm(num_groups=16, num_channels=128)
        # Block 4: 128 -> 256
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.norm4 = nn.GroupNorm(num_groups=16, num_channels=256)

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
    m = DermMNISTCNN()
    total = sum(p.numel() for p in m.parameters())
    print(f"DermMNISTCNN — parameters: {total:,}")
    for size in (28, 32, 64, 128):
        x = torch.randn(2, 3, size, size)
        print(f"  input {x.shape} -> output {m(x).shape}")
