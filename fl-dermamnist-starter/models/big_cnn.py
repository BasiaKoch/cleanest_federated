"""Thesis-grade CNN: 4 conv blocks (32 → 64 → 128 → 256) + 2 FC.

Larger than SimpleCNN so that algorithmic differences between FedAvg and
FedProx aren't masked by model underfitting.
"""
from __future__ import annotations

import torch
from torch import nn


class BigCNN(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 7, dropout: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


if __name__ == '__main__':
    m = BigCNN(3, 7)
    x = torch.randn(2, 3, 28, 28)
    print('Param count:', sum(p.numel() for p in m.parameters()))
    print('Output @ 28×28:', m(x).shape)
    x = torch.randn(2, 3, 64, 64)
    print('Output @ 64×64:', m(x).shape)
