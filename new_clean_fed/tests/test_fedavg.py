"""Smoke tests for FedAvg client and the CNN architecture."""
import torch
import pytest

from new_clean_fed.src.fedavg.model import CNNClassifier
from new_clean_fed.src.fedavg.client import (
    state_dict_to_numpy, numpy_to_state_dict,
)


def test_cnn_forward_shapes():
    m = CNNClassifier(3, 7, dropout=0.2)
    for size in [28, 32, 64, 128]:
        x = torch.randn(2, 3, size, size)
        out = m(x)
        assert out.shape == (2, 7), f"size={size}: got {out.shape}"


def test_cnn_param_count_matches_spec():
    """Spec: 4 conv blocks + FC(256→128) + FC(128→7). Roughly ~430K parameters."""
    m = CNNClassifier(3, 7, dropout=0.2)
    total = sum(p.numel() for p in m.parameters())
    assert 200_000 < total < 800_000, f"Unexpected param count {total}"


def test_state_dict_roundtrip():
    """Serialization round-trip preserves weights exactly."""
    torch.manual_seed(0)
    a = CNNClassifier(3, 7)
    arr = state_dict_to_numpy(a)
    torch.manual_seed(1)   # different seed: ensure b starts different
    b = CNNClassifier(3, 7)
    numpy_to_state_dict(b, arr)
    for (k, va), (_, vb) in zip(a.state_dict().items(), b.state_dict().items()):
        assert torch.allclose(va.float(), vb.float(), atol=1e-7), f"Mismatch in {k}"


def test_state_dict_includes_bn_buffers():
    """BatchNorm running stats must round-trip too (not just trainable params)."""
    m = CNNClassifier(3, 7)
    sd = m.state_dict()
    bn_keys = [k for k in sd.keys() if 'running_mean' in k or 'running_var' in k or 'num_batches_tracked' in k]
    assert len(bn_keys) >= 4, f"Expected ≥4 BN buffer keys, got: {bn_keys}"
    arrays = state_dict_to_numpy(m)
    assert len(arrays) == len(sd), f"Parameter count mismatch: {len(arrays)} vs {len(sd)}"
