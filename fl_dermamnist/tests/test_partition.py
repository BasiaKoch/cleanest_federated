"""Unit tests for the DermMNIST partitioner."""
from __future__ import annotations

import numpy as np
import pytest

from fl_dermamnist.data.partition import (
    NUM_CLASSES,
    SIMPLE_PATHOLOGICAL_3_CLIENTS,
    MEDICAL_SKEW_7_CLIENTS,
    QUANTITY_SKEW_IMPROVED_SPEC,
    BALANCED_PAIRED_7_CLIENTS_SPEC,
    TWO_CLIENT_90_10_RARE_STRESS_SPEC,
    simple_pathological_3_clients,
    medical_skew_7_clients,
    balanced_specialist_7_clients,
    balanced_paired_7_clients,
    quantity_skew_improved,
    two_client_90_10_rare_stress,
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
        if c == 5:   # mel_nevi background - present everywhere
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


def _real_dermmnist_labels() -> np.ndarray:
    """Real DermMNIST training-set class proportions (exactly 7007 samples)."""
    counts = [228, 359, 769, 80, 779, 4693, 99]
    labels = []
    for c, n in enumerate(counts):
        labels.extend([c] * n)
    labels = np.asarray(labels, dtype=np.int64)
    np.random.default_rng(0).shuffle(labels)
    return labels


def test_two_client_stress_spec_matches_real_dermmnist_counts():
    """Spec must sum to exact DermMNIST training-set class counts."""
    actual_counts = {0: 228, 1: 359, 2: 769, 3: 80, 4: 779, 5: 4693, 6: 99}
    spec_sums = {c: 0 for c in range(NUM_CLASSES)}
    for entry in TWO_CLIENT_90_10_RARE_STRESS_SPEC:
        for c, n in entry["per_class"].items():
            spec_sums[c] += int(n)
    for c in range(NUM_CLASSES):
        assert spec_sums[c] == actual_counts[c], (
            f"class {c}: spec sums to {spec_sums[c]}, real DermMNIST has {actual_counts[c]}"
        )


def test_two_client_stress_returns_exactly_two_clients():
    clients, _ = two_client_90_10_rare_stress(_real_dermmnist_labels(), seed=42)
    assert len(clients) == 2


def test_two_client_stress_split_is_approximately_90_10():
    """Client 0 ≈ 86%, Client 1 ≈ 14%. Within the 'approximately 90/10' band."""
    labels = _real_dermmnist_labels()
    clients, _ = two_client_90_10_rare_stress(labels, seed=42)
    total = len(labels)
    s0 = len(clients[0]) / total
    s1 = len(clients[1]) / total
    assert 0.80 <= s0 <= 0.95, f"Client 0 fraction {s0:.3f} out of [0.80, 0.95]"
    assert 0.05 <= s1 <= 0.20, f"Client 1 fraction {s1:.3f} out of [0.05, 0.20]"
    assert abs(s0 + s1 - 1.0) < 1e-9


def test_two_client_stress_is_class_disjoint():
    """The defining property: every class lives on exactly one client."""
    labels = _real_dermmnist_labels()
    clients, _ = two_client_90_10_rare_stress(labels, seed=42)
    classes_per_client = [
        set(int(labels[i]) for i in cl) for cl in clients
    ]
    assert classes_per_client[0] == {0, 1, 2, 5}, classes_per_client[0]
    assert classes_per_client[1] == {3, 4, 6}, classes_per_client[1]
    assert classes_per_client[0].isdisjoint(classes_per_client[1])


def test_two_client_stress_client0_has_no_melanoma_dermato_vascular():
    """Engineered stress test: Client 0 must hold ZERO critical/rare class samples."""
    labels = _real_dermmnist_labels()
    clients, _ = two_client_90_10_rare_stress(labels, seed=42)
    for crit_class in (3, 4, 6):
        n = sum(1 for i in clients[0] if int(labels[i]) == crit_class)
        assert n == 0, f"Client 0 holds {n} class-{crit_class} samples; expected 0"


def test_two_client_stress_client1_holds_all_melanoma():
    """All 779 melanoma samples must land on Client 1."""
    labels = _real_dermmnist_labels()
    clients, _ = two_client_90_10_rare_stress(labels, seed=42)
    n_mel = sum(1 for i in clients[1] if int(labels[i]) == 4)
    assert n_mel == 779, f"Client 1 holds {n_mel}/779 melanoma; expected 779"
    n_derm = sum(1 for i in clients[1] if int(labels[i]) == 3)
    assert n_derm == 80
    n_vasc = sum(1 for i in clients[1] if int(labels[i]) == 6)
    assert n_vasc == 99


def test_two_client_stress_no_duplicates_and_full_coverage():
    labels = _real_dermmnist_labels()
    clients, _ = two_client_90_10_rare_stress(labels, seed=42)
    flat = [i for cl in clients for i in cl]
    assert len(flat) == len(labels), "wrong total sample count"
    assert len(set(flat)) == len(labels), "duplicates across clients"
    assert set(flat) == set(range(len(labels))), "missing some sample indices"


def test_two_client_stress_deterministic_for_same_seed():
    labels = _real_dermmnist_labels()
    a, _ = two_client_90_10_rare_stress(labels, seed=42)
    b, _ = two_client_90_10_rare_stress(labels, seed=42)
    for ca, cb in zip(a, b):
        assert ca == cb


def test_two_client_stress_seeds_shuffle_within_class():
    """Different seeds → different within-class sample ORDER.

    Because the partition is class-disjoint and consumes 100% of each
    class, the SET of sample indices on each client is identical across
    seeds - only the order within each client's index list changes.
    This is the invariant we care about for paired-fair runs: the same
    seed yields the same dataloader iteration order, so FedAvg and
    FedProx see the same batches in the same order.
    """
    labels = _real_dermmnist_labels()
    a, _ = two_client_90_10_rare_stress(labels, seed=42)
    b, _ = two_client_90_10_rare_stress(labels, seed=123)
    # Sample memberships are identical (consequence of class-disjoint + full-class consumption).
    assert set(a[0]) == set(b[0]) and set(a[1]) == set(b[1])
    # But order differs - seed actually affects the per-class shuffle.
    assert a[0] != b[0] or a[1] != b[1], \
        "seed-42 and seed-123 produced identical within-class orderings"


def test_two_client_stress_aborts_when_dataset_class_counts_differ():
    """If labels do not match the hardcoded spec, must abort loudly."""
    bad_labels = np.array([0, 1, 5, 5, 5], dtype=np.int64)
    with pytest.raises(ValueError):
        two_client_90_10_rare_stress(bad_labels, seed=42)


def test_simple_mode_aborts_if_definition_misses_a_class():
    """If a class is not in any client's assigned set, the function must abort."""
    # We can't easily mutate SIMPLE_PATHOLOGICAL_3_CLIENTS in a test (it's a tuple
    # constant), but we CAN feed it labels that include an out-of-range value.
    bogus_labels = np.array([0, 1, 2, 3, 4, 5, 6, 99], dtype=np.int64)
    with pytest.raises(ValueError):
        simple_pathological_3_clients(bogus_labels, seed=42)


# =====================================================================
# Heterogeneity-ladder partitions (Levels 0-3)
# =====================================================================
#
# Tests run against synthetic labels constructed at the REAL DermMNIST
# training-set proportions (228+359+769+80+779+4693+99 = 7007). This is
# the same fixture the Level-4 (90/10) test block uses below for
# consistency; some of the new partitions hard-code quotas keyed to the
# 7007-total dataset, so the test must use the same totals.

from fl_dermamnist.data.partition import (
    two_client_50_50_label_skew_only,
    two_client_50_50_stratified_iid,
    two_client_70_30_rare_enriched,
    two_client_86_14_quantity_only_stratified,
)

LADDER_PARTITIONS = [
    ("two_client_50_50_stratified_iid", two_client_50_50_stratified_iid),
    ("two_client_86_14_quantity_only_stratified",
     two_client_86_14_quantity_only_stratified),
    ("two_client_50_50_label_skew_only", two_client_50_50_label_skew_only),
    ("two_client_70_30_rare_enriched", two_client_70_30_rare_enriched),
]

RARE_CLASSES = (3, 4, 6)        # dermato, melanoma, vascular


@pytest.fixture
def real_dermmnist_labels():
    """Synthetic labels at real DermMNIST training-set counts (7007)."""
    counts = [228, 359, 769, 80, 779, 4693, 99]
    labels = np.concatenate([[c] * n for c, n in enumerate(counts)])
    rng = np.random.default_rng(0)
    rng.shuffle(labels)
    return labels


# --- Shared invariants across all four ladder partitions -------------

@pytest.mark.parametrize("name,fn", LADDER_PARTITIONS)
def test_ladder_returns_exactly_two_clients(real_dermmnist_labels, name, fn):
    clients, _ = fn(real_dermmnist_labels, seed=42)
    assert len(clients) == 2, f"{name}: expected 2 clients, got {len(clients)}"


@pytest.mark.parametrize("name,fn", LADDER_PARTITIONS)
def test_ladder_full_coverage(real_dermmnist_labels, name, fn):
    """Every training-set sample is assigned to exactly one client."""
    clients, _ = fn(real_dermmnist_labels, seed=42)
    flat = sorted(int(i) for c in clients for i in c)
    expected = list(range(len(real_dermmnist_labels)))
    assert flat == expected, f"{name}: missing or duplicated indices"


@pytest.mark.parametrize("name,fn", LADDER_PARTITIONS)
def test_ladder_no_overlap(real_dermmnist_labels, name, fn):
    clients, _ = fn(real_dermmnist_labels, seed=42)
    c0_set = set(clients[0])
    c1_set = set(clients[1])
    assert len(c0_set & c1_set) == 0, f"{name}: clients overlap"


@pytest.mark.parametrize("name,fn", LADDER_PARTITIONS)
def test_ladder_global_class_counts_preserved(real_dermmnist_labels, name, fn):
    clients, _ = fn(real_dermmnist_labels, seed=42)
    expected = np.bincount(real_dermmnist_labels, minlength=NUM_CLASSES)
    actual = np.zeros(NUM_CLASSES, dtype=np.int64)
    for c in clients:
        for i in c:
            actual[real_dermmnist_labels[i]] += 1
    assert (actual == expected).all(), f"{name}: class counts altered"


@pytest.mark.parametrize("name,fn", LADDER_PARTITIONS)
def test_ladder_deterministic_same_seed(real_dermmnist_labels, name, fn):
    a, _ = fn(real_dermmnist_labels, seed=42)
    b, _ = fn(real_dermmnist_labels, seed=42)
    assert [list(map(int, c)) for c in a] == [list(map(int, c)) for c in b], \
        f"{name}: same seed produced different partitions"


@pytest.mark.parametrize("name,fn", LADDER_PARTITIONS)
def test_ladder_different_seeds_differ_within_class_order(
        real_dermmnist_labels, name, fn):
    """The per-class C0/C1 quotas are fixed, but the within-class shuffle
    is seeded. Different seeds must produce different concrete index
    sequences on C1 (the smaller client; checked there for speed)."""
    a, _ = fn(real_dermmnist_labels, seed=42)
    b, _ = fn(real_dermmnist_labels, seed=43)
    # Quotas are identical (same dataset), so the *sets* may overlap a
    # lot. But the sorted-index sequence on Client 1 should differ.
    assert sorted(a[1]) != sorted(b[1]) or sorted(a[0]) != sorted(b[0]), \
        f"{name}: different seeds produced bit-identical partitions"


# --- Level-0 specific tests ------------------------------------------

def test_level0_is_approximately_50_50(real_dermmnist_labels):
    clients, _ = two_client_50_50_stratified_iid(real_dermmnist_labels, seed=42)
    n0, n1 = len(clients[0]), len(clients[1])
    total = n0 + n1
    # Allow ±0.5% slack for parity rounding on odd class counts.
    assert abs(n0 / total - 0.5) < 0.005, f"C0 share {n0/total:.4f} ≠ 0.5"
    assert abs(n1 / total - 0.5) < 0.005, f"C1 share {n1/total:.4f} ≠ 0.5"


def test_level0_is_class_stratified(real_dermmnist_labels):
    """Every class is split ~50/50, not just the totals."""
    clients, _ = two_client_50_50_stratified_iid(real_dermmnist_labels, seed=42)
    for cls in range(NUM_CLASSES):
        c0_count = sum(1 for i in clients[0] if real_dermmnist_labels[i] == cls)
        c1_count = sum(1 for i in clients[1] if real_dermmnist_labels[i] == cls)
        total = c0_count + c1_count
        if total < 2:
            continue   # tiny class - skip stratification check
        # ceil/floor allowed: |c0 - c1| should be ≤ 1.
        assert abs(c0_count - c1_count) <= 1, \
            f"class {cls}: C0={c0_count}, C1={c1_count} not ceil/floor split"


# --- Level-1 specific tests ------------------------------------------

def test_level1_is_approximately_86_14(real_dermmnist_labels):
    clients, _ = two_client_86_14_quantity_only_stratified(
        real_dermmnist_labels, seed=42)
    n0, n1 = len(clients[0]), len(clients[1])
    total = n0 + n1
    assert abs(n0 / total - 0.8633) < 0.01, f"C0 share {n0/total:.4f}"
    assert abs(n1 / total - 0.1367) < 0.01, f"C1 share {n1/total:.4f}"


def test_level1_is_class_stratified(real_dermmnist_labels):
    """Every class is split ~86/14, not just totals."""
    clients, _ = two_client_86_14_quantity_only_stratified(
        real_dermmnist_labels, seed=42)
    for cls in range(NUM_CLASSES):
        c0 = sum(1 for i in clients[0] if real_dermmnist_labels[i] == cls)
        c1 = sum(1 for i in clients[1] if real_dermmnist_labels[i] == cls)
        total = c0 + c1
        if total < 5:
            continue
        share = c0 / total
        assert 0.80 <= share <= 0.92, (
            f"class {cls}: C0 share {share:.3f} not within 80-92% "
            f"(quantity-only stratification check)")


# --- Level-2 specific tests ------------------------------------------

def test_level2_is_approximately_50_50(real_dermmnist_labels):
    clients, _ = two_client_50_50_label_skew_only(
        real_dermmnist_labels, seed=42)
    n0, n1 = len(clients[0]), len(clients[1])
    total = n0 + n1
    assert abs(n0 / total - 0.5) < 0.01, f"C0 share {n0/total:.4f}"


def test_level2_rare_classes_disjoint_to_client1(real_dermmnist_labels):
    """100% of dermato/melanoma/vascular live on Client 1."""
    clients, _ = two_client_50_50_label_skew_only(
        real_dermmnist_labels, seed=42)
    for cls in RARE_CLASSES:
        c0 = sum(1 for i in clients[0] if real_dermmnist_labels[i] == cls)
        c1 = sum(1 for i in clients[1] if real_dermmnist_labels[i] == cls)
        assert c0 == 0, f"class {cls} present on Client 0 (count={c0})"
        assert c1 == int((real_dermmnist_labels == cls).sum()), \
            f"class {cls} not 100% on Client 1"


def test_level2_common_classes_0_1_2_disjoint_to_client0(
        real_dermmnist_labels):
    """100% of actinic/basal/benign_keratosis live on Client 0."""
    clients, _ = two_client_50_50_label_skew_only(
        real_dermmnist_labels, seed=42)
    for cls in (0, 1, 2):
        c1 = sum(1 for i in clients[1] if real_dermmnist_labels[i] == cls)
        assert c1 == 0, (
            f"class {cls} should be 100% on Client 0; "
            f"found {c1} on Client 1.")


# --- Level-3 specific tests ------------------------------------------

def test_level3_is_approximately_70_30(real_dermmnist_labels):
    clients, _ = two_client_70_30_rare_enriched(
        real_dermmnist_labels, seed=42)
    n0, n1 = len(clients[0]), len(clients[1])
    total = n0 + n1
    assert abs(n0 / total - 0.70) < 0.01, f"C0 share {n0/total:.4f}"
    assert abs(n1 / total - 0.30) < 0.01, f"C1 share {n1/total:.4f}"


def test_level3_rare_classes_on_client1(real_dermmnist_labels):
    """100% of rare classes on Client 1."""
    clients, _ = two_client_70_30_rare_enriched(
        real_dermmnist_labels, seed=42)
    for cls in RARE_CLASSES:
        c0 = sum(1 for i in clients[0] if real_dermmnist_labels[i] == cls)
        assert c0 == 0, f"class {cls} present on Client 0 (count={c0})"


# --- Cross-level integrity: ladder monotonicity ----------------------

def test_ladder_quantity_skew_is_monotone(real_dermmnist_labels):
    """C1 share decreases monotonically Level 0 -> 4: 50% -> 30% -> 14%.

    Level 2 (50%) and Level 0 (50%) tie, which is expected because both
    are 50/50 by construction; Level 3 is 30% and Level 4 (existing) is
    14%.
    """
    from fl_dermamnist.data.partition import two_client_90_10_rare_stress
    fns = [
        ("L0", two_client_50_50_stratified_iid),
        ("L2", two_client_50_50_label_skew_only),
        ("L3", two_client_70_30_rare_enriched),
        ("L4", two_client_90_10_rare_stress),
    ]
    shares = []
    for tag, fn in fns:
        c, _ = fn(real_dermmnist_labels, seed=42)
        shares.append((tag, len(c[1]) / sum(len(x) for x in c)))
    # Monotone non-increasing with a 1% slack for parity-rounding
    # disagreements between levels (L0 ceils on C0, L2 ceils on total
    # so L2's C1 share is 50.01% vs L0's 49.96% - a 0.05% nuisance).
    SLACK = 0.01
    for (a_tag, a), (b_tag, b) in zip(shares, shares[1:]):
        assert a + SLACK >= b, (
            f"Ladder C1 share not monotone: {a_tag}={a:.3f} -> {b_tag}={b:.3f}")

