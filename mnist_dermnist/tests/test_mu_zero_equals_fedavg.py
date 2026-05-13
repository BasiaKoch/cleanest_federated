"""μ=0 sanity test: FedProx(μ=0) must produce bit-identical results to FedAvg.

When `proximal_mu == 0`, the proximal-term branch in `local_train` is gated
off entirely (`if proximal_mu > 0:`). With identical:
  - seed → identical model init (DermMNISTCNN with GroupNorm, no BN buffers)
  - partition (identical sample assignment per client)
  - client sampling schedule (sampling_rng derived from cfg.seed)
  - DataLoader generators (seed = base*10000 + round*100 + cid)
  - optimizer hyperparameters

…the two runs should produce numerically identical per-round metrics AND
final aggregated state_dicts.

We test this directly on synthetic data — no DermMNIST dependency — to keep
the test deterministic and fast (<5s).
"""
from __future__ import annotations

import copy
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from mnist_dermnist.fl.server_loop import FLConfig, run_fl
from mnist_dermnist.models import DermMNISTCNN


# ----- Fixtures -----------------------------------------------------------

NUM_CLIENTS = 3
N_TRAIN = 60     # 20 / client
N_VAL = 30
N_TEST = 30
IMG_SIZE = 28


def _synthetic_dataset(n: int, *, seed: int) -> TensorDataset:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, IMG_SIZE, IMG_SIZE, generator=g)
    y = torch.randint(0, 7, (n,), generator=g)
    return TensorDataset(x, y)


def _partition_round_robin(n: int, num_clients: int) -> list[list[int]]:
    out: list[list[int]] = [[] for _ in range(num_clients)]
    for i in range(n):
        out[i % num_clients].append(i)
    return out


def _model_builder():
    return DermMNISTCNN()


# ----- Helpers ------------------------------------------------------------

def _run(cfg: FLConfig, train_ds, val_loader, test_loader, partitions):
    return run_fl(
        cfg,
        model_builder=_model_builder,
        train_dataset=train_ds,
        val_loader=val_loader,
        test_loader=test_loader,
        client_indices=partitions,
    )


def _flatten_state(sd):
    return torch.cat([v.flatten() for v in sd.values()])


# ----- Tests --------------------------------------------------------------

@pytest.fixture
def fixture():
    """Identical setup for both algorithms."""
    train_ds = _synthetic_dataset(N_TRAIN, seed=100)
    val_ds = _synthetic_dataset(N_VAL, seed=200)
    test_ds = _synthetic_dataset(N_TEST, seed=300)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
    partitions = _partition_round_robin(N_TRAIN, NUM_CLIENTS)
    return train_ds, val_loader, test_loader, partitions


def _make_cfg(algo: str, mu: float) -> FLConfig:
    return FLConfig(
        seed=42,
        algorithm=algo,
        mu=mu,
        num_rounds=3,
        local_epochs=2,
        fraction_fit=1.0,
        lr=0.05,
        momentum=0.9,
        weight_decay=0.0,
        batch_size=8,
        num_classes=7,
        device="cpu",
    )


def test_per_round_history_is_identical(fixture):
    """Per-round train_loss, val_loss, val_macro_f1 etc. should match exactly."""
    train_ds, val_loader, test_loader, partitions = fixture

    a = _run(_make_cfg("fedavg",  mu=0.0),
             train_ds, val_loader, test_loader, partitions)
    b = _run(_make_cfg("fedprox", mu=0.0),
             train_ds, val_loader, test_loader, partitions)

    # Compare per-round metric columns
    cols = ["round", "train_loss", "val_loss", "val_accuracy",
            "val_balanced_accuracy", "val_macro_f1"] + [f"val_f1_class_{i}" for i in range(7)]
    for col in cols:
        diffs = (a["history"][col].values - b["history"][col].values)
        max_abs = float(np.abs(diffs).max())
        # Allow a sub-1e-8 numerical slack; bit-identical in practice
        assert max_abs < 1e-8, (
            f"Column '{col}' differs by {max_abs:.2e} between FedAvg and FedProx(μ=0); "
            f"FedAvg={a['history'][col].tolist()}  FedProx(μ=0)={b['history'][col].tolist()}"
        )


def test_final_aggregated_state_dict_is_identical(fixture):
    """The final global model after all rounds must match bit-by-bit."""
    train_ds, val_loader, test_loader, partitions = fixture

    a = _run(_make_cfg("fedavg",  mu=0.0), train_ds, val_loader, test_loader, partitions)
    b = _run(_make_cfg("fedprox", mu=0.0), train_ds, val_loader, test_loader, partitions)

    # `best_state_dict` is the model selected by best val_macro_f1.
    # Since metrics are identical, the same round wins → same state_dict.
    fa, fp = a["best_state_dict"], b["best_state_dict"]
    assert fa is not None and fp is not None, "best checkpoint missing"
    assert fa.keys() == fp.keys()
    for k in fa:
        max_abs = float((fa[k] - fp[k]).abs().max().item())
        assert max_abs < 1e-8, f"state_dict['{k}']: max diff {max_abs:.2e}"


def test_test_metrics_at_best_val_are_identical(fixture):
    """Test-set evaluation at the best-val checkpoint should match."""
    train_ds, val_loader, test_loader, partitions = fixture

    a = _run(_make_cfg("fedavg",  mu=0.0), train_ds, val_loader, test_loader, partitions)
    b = _run(_make_cfg("fedprox", mu=0.0), train_ds, val_loader, test_loader, partitions)

    fa = a["test_metrics"]; fp = b["test_metrics"]
    assert fa is not None and fp is not None
    assert fa["selected_round"] == fp["selected_round"]
    for k in ("loss", "accuracy", "balanced_accuracy", "macro_f1"):
        assert abs(fa[k] - fp[k]) < 1e-8, f"test_metrics['{k}']: |Δ| = {abs(fa[k]-fp[k]):.2e}"


def test_mu_positive_changes_outcome(fixture):
    """Sanity: with μ > 0, the proximal term must actually influence training.
    (Otherwise the gating would be a no-op everywhere.)"""
    train_ds, val_loader, test_loader, partitions = fixture

    a = _run(_make_cfg("fedavg",  mu=0.0), train_ds, val_loader, test_loader, partitions)
    b = _run(_make_cfg("fedprox", mu=1.0), train_ds, val_loader, test_loader, partitions)

    fa = _flatten_state(a["best_state_dict"])
    fp = _flatten_state(b["best_state_dict"])
    max_abs = float((fa - fp).abs().max().item())
    assert max_abs > 1e-4, (
        f"FedProx(μ=1.0) produced identical weights to FedAvg — proximal term inactive (max diff {max_abs:.2e})"
    )


def test_best_round_selection_via_val_macro_f1(fixture):
    """The selected `best_round` must match argmax(val_macro_f1)."""
    train_ds, val_loader, test_loader, partitions = fixture
    result = _run(_make_cfg("fedavg", mu=0.0), train_ds, val_loader, test_loader, partitions)
    history = result["history"]
    expected_best_round = int(history.loc[history["val_macro_f1"].idxmax(), "round"])
    actual_best_round = int(result["test_metrics"]["selected_round"])
    assert expected_best_round == actual_best_round, \
        f"best_round mismatch: expected {expected_best_round}, got {actual_best_round}"


def test_history_has_required_columns(fixture):
    """All columns required by the spec must be present in the per-round history."""
    train_ds, val_loader, test_loader, partitions = fixture
    result = _run(_make_cfg("fedavg", mu=0.0), train_ds, val_loader, test_loader, partitions)
    required = {
        "seed", "algorithm", "mu", "local_epochs", "round",
        "train_loss", "val_loss", "val_accuracy", "val_balanced_accuracy", "val_macro_f1",
    } | {f"val_f1_class_{i}" for i in range(7)}
    missing = required - set(result["history"].columns)
    assert not missing, f"history is missing columns: {sorted(missing)}"


def test_no_test_columns_in_per_round_history(fixture):
    """Per the spec: test metrics only at the SELECTED best-val checkpoint, not per round."""
    train_ds, val_loader, test_loader, partitions = fixture
    result = _run(_make_cfg("fedavg", mu=0.0), train_ds, val_loader, test_loader, partitions)
    test_cols = [c for c in result["history"].columns if c.startswith("test_")]
    assert test_cols == [], f"test columns leaked into history: {test_cols}"
