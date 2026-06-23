"""Tests for DermMNISTCNN_BN — the BatchNorm architecture-ablation variant.

Mirrors test_dermmnist_cnn.py but inverts the normalisation expectations:
the BN variant MUST use BatchNorm2d and MUST NOT use GroupNorm, and is
otherwise architecturally identical to DermMNISTCNN.
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

from fl_dermamnist.models import (
    DermMNISTCNN,
    DermMNISTCNN_BN,
    get_model,
    resolve_variant,
)


# ----- core architecture / shape -----

def test_bn_output_shape_for_batch_4x3x28x28():
    """Batch [4, 3, 28, 28] -> output [4, 7], same as the GN variant."""
    m = DermMNISTCNN_BN()
    m.eval()
    x = torch.randn(4, 3, 28, 28)
    y = m(x)
    assert y.shape == (4, 7), f"expected (4, 7), got {tuple(y.shape)}"


def test_bn_works_on_multiple_resolutions():
    """AdaptiveAvgPool2d should preserve resolution-agnosticism."""
    m = DermMNISTCNN_BN()
    m.eval()
    for size in (28, 32, 64, 128):
        out = m(torch.randn(2, 3, size, size))
        assert out.shape == (2, 7), f"size={size}: got {tuple(out.shape)}"


# ----- normalization layer contract -----

def test_bn_variant_uses_batchnorm_not_groupnorm():
    """The BN ablation MUST use BatchNorm2d everywhere, no GroupNorm."""
    m = DermMNISTCNN_BN()
    bn_modules = [name for name, mod in m.named_modules()
                  if isinstance(mod, nn.BatchNorm2d)]
    gn_modules = [name for name, mod in m.named_modules()
                  if isinstance(mod, nn.GroupNorm)]
    assert gn_modules == [], f"GroupNorm forbidden in BN variant, found: {gn_modules}"
    assert len(bn_modules) == 4, f"expected 4 BatchNorm2d layers, got {len(bn_modules)}"


def test_bn_channel_counts_match_spec():
    """BatchNorm2d channels must match (32, 64, 128, 256) in order."""
    m = DermMNISTCNN_BN()
    expected = [32, 64, 128, 256]
    actual = [bn.num_features
              for _, bn in m.named_modules() if isinstance(bn, nn.BatchNorm2d)]
    assert actual == expected, f"expected {expected}, got {actual}"


# ----- FC head + dropout: identical to GN variant -----

def test_bn_fc_head_shapes():
    m = DermMNISTCNN_BN()
    assert m.fc1.in_features == 256 and m.fc1.out_features == 128
    assert m.fc2.in_features == 128 and m.fc2.out_features == 7


def test_bn_dropout_default_is_0_2():
    m = DermMNISTCNN_BN()
    assert abs(m.dropout.p - 0.2) < 1e-9


def test_bn_param_count_close_to_gn():
    """BN has the same number of trainable parameters as GN (gamma + beta
    per channel, matching GN's affine params). Running buffers do not count."""
    bn = DermMNISTCNN_BN()
    gn = DermMNISTCNN()
    bn_trainable = sum(p.numel() for p in bn.parameters() if p.requires_grad)
    gn_trainable = sum(p.numel() for p in gn.parameters() if p.requires_grad)
    assert bn_trainable == gn_trainable, (
        f"BN ({bn_trainable:,}) and GN ({gn_trainable:,}) should match "
        f"in trainable parameter count"
    )


def test_bn_has_running_stats_buffers():
    """BN-specific: must have running_mean and running_var buffers per layer.
    This is the very feature that makes BN FL-unfriendly under non-IID data."""
    m = DermMNISTCNN_BN()
    buffer_names = [name for name, _ in m.named_buffers()]
    expected_prefixes = ["norm1.running_mean", "norm1.running_var",
                          "norm2.running_mean", "norm2.running_var",
                          "norm3.running_mean", "norm3.running_var",
                          "norm4.running_mean", "norm4.running_var"]
    for expected in expected_prefixes:
        assert any(b.startswith(expected) for b in buffer_names), (
            f"missing BN buffer {expected!r}; have {buffer_names}"
        )


# ----- factory integration -----

def test_factory_constructs_bn_variant_via_registry_key():
    m = get_model("dermmnist_cnn_bn", num_classes=7, dropout=0.2)
    assert isinstance(m, DermMNISTCNN_BN)


def test_resolve_variant_alias_gn():
    assert resolve_variant("gn") == "dermmnist_cnn"


def test_resolve_variant_alias_bn():
    assert resolve_variant("bn") == "dermmnist_cnn_bn"


def test_resolve_variant_pass_through_registry_key():
    """Callers can also pass the full registry key directly."""
    assert resolve_variant("dermmnist_cnn") == "dermmnist_cnn"
    assert resolve_variant("dermmnist_cnn_bn") == "dermmnist_cnn_bn"


def test_resolve_variant_rejects_unknown():
    with pytest.raises(ValueError):
        resolve_variant("definitely_not_a_real_variant")


# ----- gradient flow -----

def test_bn_backward_pass_runs():
    """Forward + backward should run cleanly on a small batch."""
    m = DermMNISTCNN_BN()
    m.train()
    x = torch.randn(8, 3, 28, 28)   # batch >= 2 for BN to compute stats
    y = torch.randint(0, 7, (8,))
    loss = nn.functional.cross_entropy(m(x), y)
    loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, f"missing gradient on {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad on {name}"


# ----- back-compatibility: GN variant unchanged -----

def test_gn_default_unchanged_by_bn_addition():
    """Existing 'dermmnist_cnn' must still return GroupNorm-only network."""
    m = get_model("dermmnist_cnn", num_classes=7, dropout=0.2)
    assert isinstance(m, DermMNISTCNN)
    bn_modules = [n for n, mod in m.named_modules()
                  if isinstance(mod, nn.BatchNorm2d)]
    assert bn_modules == [], (
        f"GN variant must not contain BatchNorm; found {bn_modules}"
    )
