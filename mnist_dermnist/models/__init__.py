"""Model factory for the mnist_dermnist experiment.

Existing model names must not be displaced when new ones are added; this is
a registry-based dispatch.
"""
from __future__ import annotations

from torch import nn

from .dermmnist_cnn import DermMNISTCNN


_REGISTRY = {
    "dermmnist_cnn": lambda cfg: DermMNISTCNN(
        num_classes=int(cfg.get("num_classes", 7)),
        dropout=float(cfg.get("dropout", 0.2)),
    ),
}


def get_model(name: str, **cfg) -> nn.Module:
    """Build a model by name.

    Args:
        name: model identifier as listed in `_REGISTRY` (e.g. "dermmnist_cnn").
        **cfg: keyword arguments forwarded to the model factory
               (e.g. num_classes, dropout).
    """
    key = str(name).lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown model name '{name}'. Known: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[key](cfg)


__all__ = ["DermMNISTCNN", "get_model"]
