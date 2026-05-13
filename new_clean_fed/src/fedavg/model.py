"""CNN model — 4 conv blocks + 2 FC head.

Conv(3 → 32)  + BatchNorm + ReLU + MaxPool
Conv(32 → 64) + BatchNorm + ReLU + MaxPool
Conv(64 → 128) + BatchNorm + ReLU + MaxPool
Conv(128 → 256) + BatchNorm + ReLU + AdaptiveAvgPool
FC(256 → 128) + ReLU + Dropout(0.2)
FC(128 → 7)
"""
from __future__ import annotations

import torch
from torch import nn


class CNNClassifier(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 7, dropout: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 3 -> 32
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4: 128 -> 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
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


if __name__ == "__main__":
    m = CNNClassifier(3, 7, dropout=0.2)
    print("Parameters:", sum(p.numel() for p in m.parameters()))
    print("Output @ 64×64:", m(torch.randn(2, 3, 64, 64)).shape)
    print("Output @ 28×28:", m(torch.randn(2, 3, 28, 28)).shape)
