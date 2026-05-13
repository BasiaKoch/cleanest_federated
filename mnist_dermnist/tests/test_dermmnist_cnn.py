"""Tests for DermMNISTCNN — architecture, GroupNorm-only, shape, and
selection through the config factory."""
from __future__ import annotations

import torch
from torch import nn

from mnist_dermnist.models import DermMNISTCNN, get_model
from mnist_dermnist.configs.loader import load_config, build_model_from_config


# ----- core architecture / shape -----

def test_output_shape_for_batch_4x3x28x28():
    """Headline spec: batch [4, 3, 28, 28] -> output [4, 7]."""
    m = DermMNISTCNN()
    m.eval()
    x = torch.randn(4, 3, 28, 28)
    y = m(x)
    assert y.shape == (4, 7), f"expected (4, 7), got {tuple(y.shape)}"


def test_works_on_multiple_resolutions():
    """AdaptiveAvgPool2d should make the model resolution-agnostic."""
    m = DermMNISTCNN()
    m.eval()
    for size in (28, 32, 64, 128):
        out = m(torch.randn(2, 3, size, size))
        assert out.shape == (2, 7), f"size={size}: got {tuple(out.shape)}"


# ----- normalization layer contract -----

def test_uses_groupnorm_not_batchnorm():
    """Spec: no BatchNorm anywhere; GroupNorm only."""
    m = DermMNISTCNN()
    bn_modules = [name for name, mod in m.named_modules()
                  if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))]
    gn_modules = [name for name, mod in m.named_modules()
                  if isinstance(mod, nn.GroupNorm)]
    assert bn_modules == [], f"BatchNorm modules forbidden, found: {bn_modules}"
    assert len(gn_modules) == 4, f"expected 4 GroupNorm layers, got {len(gn_modules)}"


def test_groupnorm_num_groups_match_spec():
    """GroupNorm(4, 32), (8, 64), (16, 128), (16, 256) — in order."""
    m = DermMNISTCNN()
    expected = [(4, 32), (8, 64), (16, 128), (16, 256)]
    actual = [(gn.num_groups, gn.num_channels)
              for _, gn in m.named_modules() if isinstance(gn, nn.GroupNorm)]
    assert actual == expected, f"expected {expected}, got {actual}"


# ----- weight / FC head shapes -----

def test_fc_head_shapes():
    """Linear(256 -> 128) then Linear(128 -> 7)."""
    m = DermMNISTCNN()
    assert m.fc1.in_features == 256 and m.fc1.out_features == 128
    assert m.fc2.in_features == 128 and m.fc2.out_features == 7


def test_dropout_default_is_0_2():
    m = DermMNISTCNN()
    assert abs(m.dropout.p - 0.2) < 1e-9


def test_param_count_is_in_reasonable_range():
    """Sanity: ~430K-440K trainable parameters for this architecture."""
    m = DermMNISTCNN()
    total = sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert 350_000 < total < 500_000, f"unexpected param count {total:,}"


# ----- no pretrained weights -----

def test_no_pretrained_weights_loaded():
    """Fresh-init expectation: two independent instances differ before training."""
    torch.manual_seed(0)
    a = DermMNISTCNN()
    torch.manual_seed(1)
    b = DermMNISTCNN()
    diffs = []
    for (_, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        if pa.requires_grad:
            diffs.append((pa - pb).abs().max().item())
    assert max(diffs) > 1e-3, "Two fresh-seeded models are suspiciously identical"


# ----- config-driven instantiation -----

def test_factory_constructs_dermmnist_cnn():
    m = get_model("dermmnist_cnn", num_classes=7, dropout=0.2)
    assert isinstance(m, DermMNISTCNN)


def test_factory_rejects_unknown_model():
    import pytest
    with pytest.raises(ValueError):
        get_model("definitely_not_a_real_model")


def test_base_yaml_selects_dermmnist_cnn():
    """Spec: the model is selectable from config.yaml."""
    cfg = load_config()
    assert cfg["model"]["name"] == "dermmnist_cnn"
    m = build_model_from_config(cfg)
    assert isinstance(m, DermMNISTCNN)
    # quick functional check
    m.eval()
    out = m(torch.randn(2, 3, 28, 28))
    assert out.shape == (2, 7)


# ----- gradient flow -----

def test_backward_pass_runs():
    """Forward + backward should run cleanly on a small batch."""
    m = DermMNISTCNN()
    m.train()
    x = torch.randn(4, 3, 28, 28)
    y = torch.randint(0, 7, (4,))
    loss = nn.functional.cross_entropy(m(x), y)
    loss.backward()
    # Every trainable parameter should have a non-None gradient
    for name, p in m.named_parameters():
        assert p.grad is not None, f"missing gradient on {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad on {name}"
