"""Unit tests for the DermMNIST partitioner."""
from __future__ import annotations

import numpy as np
import pytest

from mnist_dermnist.data.partition import (
    NUM_CLASSES,
    SIMPLE_PATHOLOGICAL_3_CLIENTS,
    MEDICAL_SKEW_7_CLIENTS,
    QUANTITY_SKEW_IMPROVED_SPEC,
    BALANCED_PAIRED_7_CLIENTS_SPEC,
    simple_pathological_3_clients,
    medical_skew_7_clients,
    balanced_specialist_7_clients,
    balanced_paired_7_clients,
    quantity_skew_improved,
    class_count_table,
)


def _synthetic_labels() -> np.ndarray:
    """DermMNIST-like proportions at 1/2 scale (10015/2 ≈ 5007 samples)."""
    counts = [114, 180, 384, 40, 390, 2346, 50]   # halved from official
    labels = []
    for c, n in enumerate(counts):
        labels.extend([c] * n)
    rng = np.random.default_rng(0)
    rng.shuffle(labels)
    return np.asarray(labels, dtype=np.int64)


# ----- shared invariants for both modes -----

@pytest.fixture
def labels():
    return _synthetic_labels()


@pytest.mark.parametrize("partition_fn,k", [
    (simple_pathological_3_clients, 3),
    (medical_skew_7_clients, 7),
    (balanced_specialist_7_clients, 7),
])
def test_no_duplicate_sample_assignment(labels, partition_fn, k):
    clients, _ = partition_fn(labels, seed=42)
    flat = [i for cl in clients for i in cl]
    assert len(flat) == len(set(flat)), \
        f"{partition_fn.__name__}: {len(flat) - len(set(flat))} duplicate assignments"


@pytest.mark.parametrize("partition_fn,k", [
    (simple_pathological_3_clients, 3),
    (medical_skew_7_clients, 7),
    (balanced_specialist_7_clients, 7),
])
def test_no_missing_samples(labels, partition_fn, k):
    clients, _ = partition_fn(labels, seed=42)
    flat = [i for cl in clients for i in cl]
    assert set(flat) == set(range(len(labels))), \
        f"{partition_fn.__name__}: assigned indices do not cover [0, N)"


@pytest.mark.parametrize("partition_fn,k", [
    (simple_pathological_3_clients, 3),
    (medical_skew_7_clients, 7),
    (balanced_specialist_7_clients, 7),
])
def test_no_missing_classes_globally(labels, partition_fn, k):
    """Every class must appear in at least one client."""
    clients, _ = partition_fn(labels, seed=42)
    seen = set()
    for cl in clients:
        seen.update(int(labels[i]) for i in cl)
    assert seen == set(range(NUM_CLASSES)), f"Classes missing: {set(range(NUM_CLASSES)) - seen}"


@pytest.mark.parametrize("partition_fn,k", [
    (simple_pathological_3_clients, 3),
    (medical_skew_7_clients, 7),
    (balanced_specialist_7_clients, 7),
])
def test_returns_correct_number_of_clients(labels, partition_fn, k):
    clients, _ = partition_fn(labels, seed=42)
    assert len(clients) == k


@pytest.mark.parametrize("partition_fn,k", [
    (simple_pathological_3_clients, 3),
    (medical_skew_7_clients, 7),
    (balanced_specialist_7_clients, 7),
])
def test_deterministic_same_seed(labels, partition_fn, k):
    a, _ = partition_fn(labels, seed=42)
    b, _ = partition_fn(labels, seed=42)
    for ca, cb in zip(a, b):
        assert ca == cb, f"{partition_fn.__name__}: not deterministic with same seed"


@pytest.mark.parametrize("partition_fn,k", [
    (simple_pathological_3_clients, 3),
    (medical_skew_7_clients, 7),
    (balanced_specialist_7_clients, 7),
])
def test_different_seeds_produce_different_assignments(labels, partition_fn, k):
    """Different seeds must produce different ordering (even if class sets are identical)."""
    a, _ = partition_fn(labels, seed=42)
    b, _ = partition_fn(labels, seed=123)
    # Same total class membership per client, but ordering / per-sample assignment differs
    assert any(set(ca) != set(cb) for ca, cb in zip(a, b)) or \
        any(ca != cb for ca, cb in zip(a, b)), \
        f"{partition_fn.__name__}: same partition across seeds 42 and 123"


@pytest.mark.parametrize("partition_fn,k", [
    (simple_pathological_3_clients, 3),
    (medical_skew_7_clients, 7),
    (balanced_specialist_7_clients, 7),
])
def test_long_form_df_has_required_columns(labels, partition_fn, k):
    _, df = partition_fn(labels, seed=42)
    assert list(df.columns) == ["sample_index", "client_id", "class_id"]
    assert len(df) == len(labels)
    # No duplicates by sample_index
    assert df["sample_index"].is_unique


# ----- mode-specific invariants -----

def test_simple_mode_obeys_assigned_class_sets(labels):
    """In mode 1, each client must contain ONLY samples of its assigned classes."""
    clients, _ = simple_pathological_3_clients(labels, seed=42)
    for cid, assigned in enumerate(SIMPLE_PATHOLOGICAL_3_CLIENTS):
        for i in clients[cid]:
            assert int(labels[i]) in assigned, \
                f"client {cid} got sample with class {labels[i]} (not in {assigned})"


def test_medical_mode_specialists_are_modal(labels):
    """Specialists 3-6 should have their dominant class as the MOST frequent."""
    clients, _ = medical_skew_7_clients(labels, seed=42)
    spec_map = {3: 0, 4: 1, 5: 3, 6: 6}
    for cid, dom_class in spec_map.items():
        counts = np.bincount([int(labels[i]) for i in clients[cid]], minlength=NUM_CLASSES)
        modal = int(np.argmax(counts))
        assert modal == dom_class, \
            f"specialist {cid}: expected modal class {dom_class}, got {modal}. counts={counts.tolist()}"


def test_medical_mode_hospitals_are_largest(labels):
    """Hospitals 0,1 should each be larger than every specialist client."""
    clients, _ = medical_skew_7_clients(labels, seed=42)
    sizes = [len(c) for c in clients]
    hospital_min = min(sizes[0], sizes[1])
    specialist_max = max(sizes[3], sizes[4], sizes[5], sizes[6])
    assert hospital_min > specialist_max, \
        f"hospitals (min={hospital_min}) should exceed specialists (max={specialist_max})"


def test_class_count_table_shape(labels):
    clients, _ = medical_skew_7_clients(labels, seed=42)
    tbl = class_count_table(clients, labels)
    assert tbl.shape == (7, NUM_CLASSES + 1)
    assert tbl["total"].sum() == len(labels)


def test_balanced_mode_has_uniform_client_sizes(labels):
    """The balanced partition's FedProx-favourable property: max/min size ratio < 2×."""
    clients, _ = balanced_specialist_7_clients(labels, seed=42)
    sizes = [len(c) for c in clients]
    assert max(sizes) / min(sizes) < 2.0, \
        f"balanced partition size ratio is {max(sizes)/min(sizes):.2f}x, expected < 2x. Sizes: {sizes}"


def test_balanced_paired_spec_matches_real_dermmnist_counts():
    """The hardcoded per-class allocation must sum to DermMNIST's training counts."""
    actual_counts = {0: 228, 1: 359, 2: 769, 3: 80, 4: 779, 5: 4693, 6: 99}
    spec_sums = {c: 0 for c in range(NUM_CLASSES)}
    for entry in BALANCED_PAIRED_7_CLIENTS_SPEC:
        for c, n in entry["per_class"].items():
            spec_sums[c] += int(n)
    for c in range(NUM_CLASSES):
        assert spec_sums[c] == actual_counts[c], (
            f"class {c}: spec sums to {spec_sums[c]}, real DermMNIST has {actual_counts[c]}"
        )


def test_balanced_paired_every_minority_class_in_at_least_two_clients():
    """The defining property: every non-mel_nevi class is owned by ≥2 clients."""
    counts = [228, 359, 769, 80, 779, 4693, 99]
    labels = []
    for c, n in enumerate(counts):
        labels.extend([c] * n)
    labels = np.asarray(labels, dtype=np.int64)
    np.random.default_rng(0).shuffle(labels)
    clients, _ = balanced_paired_7_clients(labels, seed=42)
    for c in range(NUM_CLASSES):
        if c == 5:   # mel_nevi background — present everywhere
            continue
        owners = sum(1 for cl in clients
                     if any(int(labels[i]) == c for i in cl))
        assert owners >= 2, f"class {c} ({['actinic','basal','benign','dermato','melanoma','nevi','vascular'][c]}) is owned by {owners} clients, expected ≥ 2"


def test_quantity_skew_improved_spec_matches_real_dermmnist_counts():
    """The hardcoded per-class allocation in QUANTITY_SKEW_IMPROVED_SPEC
    must sum to DermMNIST's real per-class training counts."""
    actual_counts = {0: 228, 1: 359, 2: 769, 3: 80, 4: 779, 5: 4693, 6: 99}
    spec_sums = {c: 0 for c in range(NUM_CLASSES)}
    for entry in QUANTITY_SKEW_IMPROVED_SPEC:
        for c, n in entry["per_class"].items():
            spec_sums[c] += int(n)
    for c in range(NUM_CLASSES):
        assert spec_sums[c] == actual_counts[c], (
            f"class {c}: spec sums to {spec_sums[c]}, real DermMNIST has {actual_counts[c]}"
        )


def test_quantity_skew_improved_aborts_when_spec_mismatches_dataset():
    """If labels don't match the hardcoded spec, must abort loudly."""
    # Tiny synthetic with wrong class proportions
    bad_labels = np.array([0, 1, 5, 5, 5], dtype=np.int64)
    with pytest.raises(ValueError):
        quantity_skew_improved(bad_labels, seed=42)


def test_quantity_skew_improved_per_client_totals_real_dermmnist():
    """Verifies the per-client totals from the user spec on real-size labels."""
    counts = [228, 359, 769, 80, 779, 4693, 99]
    labels = []
    for c, n in enumerate(counts):
        labels.extend([c] * n)
    labels = np.asarray(labels, dtype=np.int64)
    np.random.default_rng(0).shuffle(labels)

    clients, _ = quantity_skew_improved(labels, seed=42)
    sizes = [len(c) for c in clients]
    expected = [2420, 2050, 1331, 348, 509, 150, 199]
    assert sizes == expected, f"client totals {sizes} != spec {expected}"


def test_balanced_mode_each_client_has_mel_nevi_background(labels):
    """Every client should hold roughly N/7 samples of class 5 (mel_nevi)."""
    clients, _ = balanced_specialist_7_clients(labels, seed=42)
    total_nevi = int((labels == 5).sum())
    expected = total_nevi // 7
    for cid, idxs in enumerate(clients):
        nevi_count = int(np.sum(labels[np.asarray(idxs, dtype=int)] == 5))
        assert abs(nevi_count - expected) <= 1, \
            f"client {cid}: mel_nevi count {nevi_count}, expected ~{expected}"


def test_simple_mode_aborts_if_definition_misses_a_class():
    """If a class is not in any client's assigned set, the function must abort."""
    # We can't easily mutate SIMPLE_PATHOLOGICAL_3_CLIENTS in a test (it's a tuple
    # constant), but we CAN feed it labels that include an out-of-range value.
    bogus_labels = np.array([0, 1, 2, 3, 4, 5, 6, 99], dtype=np.int64)
    with pytest.raises(ValueError):
        simple_pathological_3_clients(bogus_labels, seed=42)
