"""Model factory for the mnist_dermnist experiment.

Existing model names must not be displaced when new ones are added; this is
a registry-based dispatch.

Variant naming convention:
    "dermmnist_cnn"    — headline GroupNorm CNN, used for all primary results.
    "dermmnist_cnn_bn" — BatchNorm variant, used for the architecture ablation
                         (see ``mnist_dermnist.scripts.runpod_arch_ablation_bn``).
"""
from __future__ import annotations

from torch import nn

from .dermmnist_cnn import DermMNISTCNN
from .dermmnist_cnn_bn import DermMNISTCNN_BN


_REGISTRY = {
    "dermmnist_cnn": lambda cfg: DermMNISTCNN(
        num_classes=int(cfg.get("num_classes", 7)),
        dropout=float(cfg.get("dropout", 0.2)),
    ),
    "dermmnist_cnn_bn": lambda cfg: DermMNISTCNN_BN(
        num_classes=int(cfg.get("num_classes", 7)),
        dropout=float(cfg.get("dropout", 0.2)),
    ),
}


# Short-name -> registry-key alias used by CLI runners' --model-variant flag.
# Keeps the registry key explicit ("dermmnist_cnn_bn") while letting users
# pass the conventional short label ("gn" or "bn") at the command line.
_VARIANT_ALIAS = {
    "gn": "dermmnist_cnn",
    "bn": "dermmnist_cnn_bn",
}


def get_model(name: str, **cfg) -> nn.Module:
    """Build a model by name.

    Args:
        name: model identifier as listed in ``_REGISTRY`` (e.g. ``"dermmnist_cnn"``).
        **cfg: keyword arguments forwarded to the model factory
               (e.g. ``num_classes``, ``dropout``).
    """
    key = str(name).lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown model name '{name}'. Known: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[key](cfg)


def resolve_variant(variant: str) -> str:
    """Map a short variant label (gn|bn) to its registry key.

    Pass-through for names already in the registry, so callers can use either
    the alias ``"bn"`` or the full key ``"dermmnist_cnn_bn"``.
    """
    v = str(variant).lower()
    if v in _VARIANT_ALIAS:
        return _VARIANT_ALIAS[v]
    if v in _REGISTRY:
        return v
    raise ValueError(
        f"Unknown model variant '{variant}'. "
        f"Aliases: {sorted(_VARIANT_ALIAS.keys())}; "
        f"registry keys: {sorted(_REGISTRY.keys())}"
    )


__all__ = ["DermMNISTCNN", "DermMNISTCNN_BN", "get_model", "resolve_variant"]
