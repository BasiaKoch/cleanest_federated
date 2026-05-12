from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


def get_resnet18(in_channels: int = 3, num_classes: int = 7, pretrained: bool = False, image_size: int = 28):
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    if image_size in [28, 64] and not pretrained:
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    else:
        if in_channels != 3:
            old = model.conv1
            new = nn.Conv2d(in_channels, old.out_channels, kernel_size=old.kernel_size,
                            stride=old.stride, padding=old.padding, bias=False)
            with torch.no_grad():
                if pretrained:
                    if in_channels == 1:
                        new.weight.copy_(old.weight.mean(dim=1, keepdim=True))
                    else:
                        new.weight[:, :min(3, in_channels)].copy_(old.weight[:, :min(3, in_channels)])
                model.conv1 = new
            model.conv1 = new

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
