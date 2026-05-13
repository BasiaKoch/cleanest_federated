"""DermMNIST non-IID partitioner inspired by the MNIST pathological split.

Two modes:
  1. simple_pathological_3_clients
     - Client 0: classes [5, 4]
     - Client 1: classes [2, 1, 0]
     - Client 2: classes [3, 6]

  2. medical_skew_7_clients
     - Client 0: mostly class 5, some class 4 and 2  (big general hospital A)
     - Client 1: mostly class 5, some class 4 and 2  (big general hospital B)
     - Client 2: mostly class 4 and 2                (cancer-focused referral centre)
     - Client 3: specialist for class 0              (actinic keratoses clinic)
     - Client 4: specialist for class 1              (basal cell carcinoma clinic)
     - Client 5: specialist for class 3              (dermatofibroma clinic)
     - Client 6: specialist for class 6              (vascular lesions clinic)

Class IDs (DermMNIST):
  0 = actinic keratoses
  1 = basal cell carcinoma
  2 = benign keratosis-like lesions
  3 = dermatofibroma
  4 = melanoma
  5 = melanocytic nevi
  6 = vascular lesions

Contract:
- Deterministic given a seed.
- Different seeds produce different shuffled assignments.
- Every training sample is assigned to exactly one client.
- Every class must appear in at least one client.
- Aborts loudly (ValueError) if any class would end up missing globally.
- Returns a partition AND a long-form pandas DataFrame
  (columns: sample_index, client_id, class_id) that can be saved as CSV.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


# DermMNIST class metadata (single source of truth)
CLASS_NAMES: Tuple[str, ...] = (
    "actinic_keratoses",
    "basal_cell_carcinoma",
    "benign_keratosis_like_lesions",
    "dermatofibroma",
    "melanoma",
    "melanocytic_nevi",
    "vascular_lesions",
)
NUM_CLASSES = len(CLASS_NAMES)


# Mode 1 — pathological split: each client gets a fixed set of classes ONLY.
SIMPLE_PATHOLOGICAL_3_CLIENTS: Tuple[Tuple[int, ...], ...] = (
    (5, 4),         # client 0
    (2, 1, 0),      # client 1
    (3, 6),         # client 2
)


# Mode 5 — fully paired balanced partition.
#
# Refinement of Mode 3 (balanced_specialist_7_clients) in which EVERY minority
# class is owned by exactly two clients, eliminating singleton specialists.
# Designed to maximise FedProx's expected advantage: the proximal term only
# delivers gains when multiple clients carry the same class but disagree —
# under the previous mode, half the rare classes had a single owner, denying
# FedProx the mechanism it needs.
#
# Per-client composition (matches the spec, hardcoded counts to make the
# pairing exact):
BALANCED_PAIRED_7_CLIENTS_SPEC: List[Dict] = [
    {"id": 0, "per_class": {0: 114, 1: 180, 5: 670}},   # actinic-a + basal-a + nevi
    {"id": 1, "per_class": {0: 114, 1: 179, 5: 670}},   # actinic-b + basal-b + nevi
    {"id": 2, "per_class": {2: 385, 3:  40, 5: 670}},   # benign-a  + dermato-a + nevi
    {"id": 3, "per_class": {2: 384, 3:  40, 5: 670}},   # benign-b  + dermato-b + nevi
    {"id": 4, "per_class": {4: 390, 6:  50, 5: 670}},   # melanoma-a + vascular-a + nevi
    {"id": 5, "per_class": {4: 389, 6:  49, 5: 670}},   # melanoma-b + vascular-b + nevi
    {"id": 6, "per_class": {5: 673}},                    # general nevi-only client
]


# Mode 4 — quantity-skewed specialist 7 clients.
#
# Realistic medical referral network: 3 size-skewed hospital-style clients
# (C0–C2) hold most of the volume but share several common classes; 4 small
# specialty clinics (C3–C6) hold a unique rare class plus a small mel_nevi
# background to prevent single-class collapse during local training.
#
# Per-class counts are HARDCODED per the user-specified spec; the partitioner
# verifies these exactly match the actual class sizes in the dataset.
QUANTITY_SKEW_IMPROVED_SPEC: List[Dict] = [
    # Each entry: {client_id, name, per_class_counts: {class_id: count}}
    {"id": 0, "name": "nevi_hospital_large",
     "per_class": {5: 2300, 4:   80, 2:   40}},                # 2420
    {"id": 1, "name": "nevi_benign_hospital",
     "per_class": {5: 1700, 2:  250, 4:  100}},                # 2050
    {"id": 2, "name": "melanoma_benign_referral",
     "per_class": {4:  579, 2:  429, 5:  323}},                # 1331
    {"id": 3, "name": "actinic_clinic",
     "per_class": {0:  228, 5:  120}},                          # 348
    {"id": 4, "name": "basal_clinic",
     "per_class": {1:  359, 5:  150}},                          # 509
    {"id": 5, "name": "dermatofibroma_clinic",
     "per_class": {3:   80, 5:   50, 2:   20}},                 # 150
    {"id": 6, "name": "vascular_clinic",
     "per_class": {6:   99, 5:   50, 2:   30, 4:   20}},        # 199
]


# Mode 3 — FedProx-favourable balanced partition.
#
# 7 equal-size clients (~1000 samples each), all sharing the same background
# of class 5 (melanocytic_nevi), each specialising in ONE other class. Designed
# so each client carries roughly the same aggregation weight (1/7 ≈ 14.3%):
# this makes per-client drift matter, which is the regime where FedProx's
# proximal-term advantage is largest.
BALANCED_SPECIALIST_7_CLIENTS: List[Dict] = [
    # (specialty_class, share_of_specialty_class)
    {"id": 0, "specialty": 0, "share": 1.00},   # all 228 actinic
    {"id": 1, "specialty": 1, "share": 1.00},   # all 359 basal
    {"id": 2, "specialty": 2, "share": 0.50},   # half of 769 benign keratosis
    {"id": 3, "specialty": 2, "share": 1.00},   # the rest
    {"id": 4, "specialty": 4, "share": 0.50},   # half of 779 melanoma
    {"id": 5, "specialty": 4, "share": 1.00},   # the rest
    {"id": 6, "specialty": (3, 6), "share": 1.00},  # all 80 dermato + 99 vascular (both rare)
]


# Mode 2 — medical skew with both DOMINANT and SECONDARY classes.
#
# For each client we list:
#   (dominant_classes, secondary_classes, target_total_fraction_of_dominant)
#
# dominant_classes  : these are the bulk; the client takes a large fraction of the
#                     globally-available samples of each.
# secondary_classes : the client gets a small share of these to provide overlap
#                     (so training is not single-class collapse).
# dominant_share    : fraction of each dominant class's globally-remaining pool
#                     that this client receives.
#
# We process clients top-to-bottom and consume from the class pools as we go,
# so order matters. Hospitals (high quantity) take their share first; specialists
# pick up what's left.
MEDICAL_SKEW_7_CLIENTS: List[Dict] = [
    # Two large general hospitals: mostly mel_nevi (5), some melanoma (4) and benign_kerat (2)
    {"id": 0, "name": "hospital_A",
     "dominant": [5], "dominant_share": 0.40,
     "secondary": [4, 2], "secondary_share": 0.20},
    {"id": 1, "name": "hospital_B",
     "dominant": [5], "dominant_share": 0.40,
     "secondary": [4, 2], "secondary_share": 0.20},
    # Cancer-focused centre: melanoma + benign_kerat
    {"id": 2, "name": "cancer_referral",
     "dominant": [4, 2], "dominant_share": 0.45,
     "secondary": [5], "secondary_share": 0.10},
    # Specialists for the four rare classes
    {"id": 3, "name": "specialist_actinic",
     "dominant": [0], "dominant_share": 0.75,
     "secondary": [], "secondary_share": 0.0},
    {"id": 4, "name": "specialist_basal",
     "dominant": [1], "dominant_share": 0.75,
     "secondary": [], "secondary_share": 0.0},
    {"id": 5, "name": "specialist_dermatofibroma",
     "dominant": [3], "dominant_share": 0.75,
     "secondary": [], "secondary_share": 0.0},
    {"id": 6, "name": "specialist_vascular",
     "dominant": [6], "dominant_share": 0.75,
     "secondary": [], "secondary_share": 0.0},
]


# ----------------------------------------------------------------------------
# Core builders
# ----------------------------------------------------------------------------

def _class_pools(labels: np.ndarray, seed: int) -> Dict[int, np.ndarray]:
    """Return {class_id: shuffled np.ndarray of sample indices} keyed by class.

    The per-class shuffle uses a seed-derived child generator so different
    seeds produce different orderings (and therefore different shuffled
    partitions).
    """
    rng = np.random.default_rng(seed)
    pools: Dict[int, np.ndarray] = {}
    for c in range(NUM_CLASSES):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        pools[c] = idx
    return pools


def simple_pathological_3_clients(
    labels: Sequence[int],
    seed: int = 42,
) -> Tuple[List[List[int]], pd.DataFrame]:
    """Mode 1 — each client gets a fixed disjoint set of classes.

    Args:
        labels: length-N array of integer class IDs (0..6).
        seed: deterministic shuffle seed.

    Returns:
        (client_indices, df)
          client_indices: List[List[int]] of length 3
          df: long-form DataFrame [sample_index, client_id, class_id]
    """
    labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    _assert_labels_valid(labels_arr)

    pools = _class_pools(labels_arr, seed)
    clients: List[List[int]] = [[] for _ in range(len(SIMPLE_PATHOLOGICAL_3_CLIENTS))]

    # Each class belongs to exactly one client in this mode.
    class_to_client: Dict[int, int] = {}
    for cid, classes in enumerate(SIMPLE_PATHOLOGICAL_3_CLIENTS):
        for c in classes:
            if c in class_to_client:
                raise ValueError(
                    f"Class {c} is assigned to multiple clients in mode 1 "
                    f"({class_to_client[c]} and {cid}). Definition is invalid."
                )
            class_to_client[c] = cid

    missing = [c for c in range(NUM_CLASSES) if c not in class_to_client]
    if missing:
        raise ValueError(
            f"Mode 1 leaves classes {missing} unassigned. Every class must appear "
            f"in at least one client by spec."
        )

    for c, cid in class_to_client.items():
        clients[cid].extend(pools[c].tolist())

    df = _build_long_form(clients, labels_arr)
    _validate(clients, labels_arr)
    return clients, df


def medical_skew_7_clients(
    labels: Sequence[int],
    seed: int = 42,
) -> Tuple[List[List[int]], pd.DataFrame]:
    """Mode 2 — realistic medical-style skew with overlap.

    Each client gets a controlled share of one or two dominant classes plus a
    smaller share of secondary classes. Remaining unallocated samples are
    distributed round-robin to ensure full coverage and that every class
    appears globally.
    """
    labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    _assert_labels_valid(labels_arr)

    pools = _class_pools(labels_arr, seed)
    # Each pool is a position cursor: take from the front.
    pool_cursors: Dict[int, int] = {c: 0 for c in range(NUM_CLASSES)}
    clients: List[List[int]] = [[] for _ in range(len(MEDICAL_SKEW_7_CLIENTS))]

    def take(c: int, n: int) -> np.ndarray:
        cur = pool_cursors[c]
        end = min(cur + n, len(pools[c]))
        chunk = pools[c][cur:end]
        pool_cursors[c] = end
        return chunk

    for spec in MEDICAL_SKEW_7_CLIENTS:
        cid = spec["id"]
        # Dominant classes — take a large share of what's still available
        for c in spec["dominant"]:
            remaining = len(pools[c]) - pool_cursors[c]
            n = int(round(remaining * spec["dominant_share"]))
            clients[cid].extend(take(c, n).tolist())
        # Secondary classes — small share to give the client some overlap
        for c in spec["secondary"]:
            remaining = len(pools[c]) - pool_cursors[c]
            n = int(round(remaining * spec["secondary_share"]))
            clients[cid].extend(take(c, n).tolist())

    # Route unallocated samples to HOSPITAL clients only (0, 1, 2). This
    # preserves specialist clients' modal class — otherwise the abundance of
    # mel_nevi (class 5) would swamp every specialist's small share.
    hospital_ids = [0, 1, 2]
    rng_left = np.random.default_rng(seed + 9001)
    for c in range(NUM_CLASSES):
        leftover_c = pools[c][pool_cursors[c]:].tolist()
        rng_left.shuffle(leftover_c)
        # Round-robin across hospitals so they're roughly equal in size
        for i, idx in enumerate(leftover_c):
            clients[hospital_ids[i % len(hospital_ids)]].append(int(idx))

    df = _build_long_form(clients, labels_arr)
    _validate(clients, labels_arr)
    return clients, df


def balanced_specialist_7_clients(
    labels: Sequence[int],
    seed: int = 42,
) -> Tuple[List[List[int]], pd.DataFrame]:
    """Mode 3 — equal-size specialists. FedProx-favourable partition.

    7 clients, each ~1000 samples. Composition per client:
      - 1/7 share of class 5 (mel_nevi) — uniform background
      - Their full allocated share of one specialty class (or pair, for C6)

    Aggregation weights are ~uniform across clients (each ~14%), so per-client
    drift contributes meaningfully to the aggregated model. This is the regime
    where FedProx's proximal-term advantage is largest (paper §5.1).
    """
    labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    _assert_labels_valid(labels_arr)

    K = len(BALANCED_SPECIALIST_7_CLIENTS)
    pools = _class_pools(labels_arr, seed)
    clients: List[List[int]] = [[] for _ in range(K)]

    # 1) Split mel_nevi (class 5) equally across all 7 clients.
    nevi = pools[5]
    nevi_chunks = np.array_split(nevi, K)
    for cid in range(K):
        clients[cid].extend(int(i) for i in nevi_chunks[cid])

    # 2) Distribute specialty classes per the BALANCED_SPECIALIST_7_CLIENTS spec.
    # Cursors track how much of each non-mel_nevi class has been claimed.
    cursors: Dict[int, int] = {c: 0 for c in range(num_classes_helper())}

    def take(c: int, share: float) -> np.ndarray:
        """Take `share` fraction of class c's REMAINING samples."""
        remaining = pools[c][cursors[c]:]
        n = int(round(len(remaining) * share))
        chunk = remaining[:n]
        cursors[c] += n
        return chunk

    for spec in BALANCED_SPECIALIST_7_CLIENTS:
        cid = spec["id"]
        specialty = spec["specialty"]
        share = spec["share"]
        if isinstance(specialty, tuple):
            for c in specialty:
                clients[cid].extend(int(i) for i in take(c, share))
        else:
            clients[cid].extend(int(i) for i in take(specialty, share))

    # 3) Sweep up any straggler samples (rounding errors, unallocated rare classes)
    # via round-robin onto clients to satisfy "every sample assigned exactly once".
    leftover: List[int] = []
    for c in range(NUM_CLASSES):
        if c == 5:
            continue   # mel_nevi already fully consumed in step 1
        leftover.extend(pools[c][cursors[c]:].tolist())
    rng_left = np.random.default_rng(seed + 9999)
    rng_left.shuffle(leftover)
    for i, idx in enumerate(leftover):
        clients[i % K].append(int(idx))

    df = _build_long_form(clients, labels_arr)
    _validate(clients, labels_arr)
    return clients, df


def num_classes_helper() -> int:
    return NUM_CLASSES


def balanced_paired_7_clients(
    labels: Sequence[int],
    seed: int = 42,
) -> Tuple[List[List[int]], pd.DataFrame]:
    """Mode 5 — every minority class is owned by exactly 2 clients.

    Each client gets a near-equal mel_nevi background plus two paired-specialty
    fragments (or only nevi for the general client). Designed to maximise
    FedProx's advantage on the per-class drift mechanism: with every rare
    class held by two clients, the proximal term has consensus-finding work
    to do everywhere — no singleton ownership.
    """
    labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    _assert_labels_valid(labels_arr)

    # Verify spec consistency
    actual = {c: int((labels_arr == c).sum()) for c in range(NUM_CLASSES)}
    spec_totals: Dict[int, int] = {c: 0 for c in range(NUM_CLASSES)}
    for entry in BALANCED_PAIRED_7_CLIENTS_SPEC:
        for c, n in entry["per_class"].items():
            spec_totals[c] += int(n)
    for c in range(NUM_CLASSES):
        if spec_totals[c] != actual[c]:
            raise ValueError(
                f"balanced_paired_7_clients: spec sums to {spec_totals[c]} for class "
                f"{c} ({CLASS_NAMES[c]}) but the training set has {actual[c]}."
            )

    pools = _class_pools(labels_arr, seed)
    cursors: Dict[int, int] = {c: 0 for c in range(NUM_CLASSES)}

    K = len(BALANCED_PAIRED_7_CLIENTS_SPEC)
    clients: List[List[int]] = [[] for _ in range(K)]
    for entry in BALANCED_PAIRED_7_CLIENTS_SPEC:
        cid = entry["id"]
        for c, n in entry["per_class"].items():
            start = cursors[c]
            end = start + int(n)
            if end > len(pools[c]):
                raise ValueError(
                    f"balanced_paired_7_clients: ran out of class {c} during client {cid}."
                )
            clients[cid].extend(int(i) for i in pools[c][start:end])
            cursors[c] = end

    df = _build_long_form(clients, labels_arr)
    _validate(clients, labels_arr)
    return clients, df


def quantity_skew_improved(
    labels: Sequence[int],
    seed: int = 42,
) -> Tuple[List[List[int]], pd.DataFrame]:
    """Mode 4 — quantity-skewed specialist 7-client partition.

    Realistic medical referral network: 3 large hospital-style clients (C0–C2)
    own most of the volume but share common classes (nevi, melanoma, benign);
    4 small specialty clinics (C3–C6) each own a unique rare class plus a
    mel_nevi background so local training doesn't collapse to one class.

    The per-class allocations are HARDCODED via `QUANTITY_SKEW_IMPROVED_SPEC`.
    The function:
      - validates that the spec's per-class sums match the dataset's class
        sizes exactly (raises ValueError if not — abort loudly per spec);
      - shuffles each class deterministically by seed;
      - takes the first n samples of each class for each client (in spec order).

    Different seeds therefore produce different SAMPLE-LEVEL assignments but
    the same per-class structure (sizes are fixed by the spec).
    """
    labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    _assert_labels_valid(labels_arr)

    # 1) Verify the spec is consistent with the dataset.
    actual = {c: int((labels_arr == c).sum()) for c in range(NUM_CLASSES)}
    spec_totals: Dict[int, int] = {c: 0 for c in range(NUM_CLASSES)}
    for entry in QUANTITY_SKEW_IMPROVED_SPEC:
        for c, n in entry["per_class"].items():
            spec_totals[c] += int(n)
    for c in range(NUM_CLASSES):
        if spec_totals[c] != actual[c]:
            raise ValueError(
                f"quantity_skew_improved: spec sums to {spec_totals[c]} for class "
                f"{c} ({CLASS_NAMES[c]}) but the training set has {actual[c]}. "
                f"The spec is incompatible with this dataset."
            )

    # 2) Pre-shuffle each class deterministically.
    pools = _class_pools(labels_arr, seed)
    cursors: Dict[int, int] = {c: 0 for c in range(NUM_CLASSES)}

    # 3) Allocate per spec order.
    K = len(QUANTITY_SKEW_IMPROVED_SPEC)
    clients: List[List[int]] = [[] for _ in range(K)]
    for entry in QUANTITY_SKEW_IMPROVED_SPEC:
        cid = entry["id"]
        for c, n in entry["per_class"].items():
            start = cursors[c]
            end = start + int(n)
            if end > len(pools[c]):
                raise ValueError(
                    f"quantity_skew_improved: ran out of class {c} during client {cid} "
                    f"allocation (requested {n}, only {len(pools[c]) - start} remain)."
                )
            clients[cid].extend(int(i) for i in pools[c][start:end])
            cursors[c] = end

    # 4) Sanity-check no leftovers (spec totals == dataset totals so this should be empty).
    leftover_total = sum(len(pools[c]) - cursors[c] for c in range(NUM_CLASSES))
    if leftover_total != 0:
        raise ValueError(
            f"quantity_skew_improved: {leftover_total} samples left unassigned after spec "
            f"applied. Sums don't match — please check QUANTITY_SKEW_IMPROVED_SPEC."
        )

    df = _build_long_form(clients, labels_arr)
    _validate(clients, labels_arr)
    return clients, df


# ----------------------------------------------------------------------------
# Validation, table, IO
# ----------------------------------------------------------------------------

def _assert_labels_valid(labels: np.ndarray) -> None:
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1-D (got shape {labels.shape})")
    present = np.unique(labels)
    if present.min() < 0 or present.max() >= NUM_CLASSES:
        raise ValueError(
            f"labels must be in [0, {NUM_CLASSES - 1}]; got min={present.min()}, max={present.max()}"
        )


def _validate(clients: List[List[int]], labels: np.ndarray) -> None:
    """Abort loudly on any partition invariant violation."""
    n = len(labels)
    all_assigned = [i for cl in clients for i in cl]
    if len(all_assigned) != n:
        raise ValueError(
            f"Partition assigns {len(all_assigned)} samples but dataset has {n}. "
            f"Missing or extra indices."
        )
    if len(set(all_assigned)) != n:
        dupes = len(all_assigned) - len(set(all_assigned))
        raise ValueError(
            f"Partition contains {dupes} duplicate sample assignments. "
            f"Every sample must belong to exactly one client."
        )
    if set(all_assigned) != set(range(n)):
        raise ValueError("Assigned indices do not match range(N) exactly.")
    classes_seen = set(int(labels[i]) for i in all_assigned)
    missing = sorted(set(range(NUM_CLASSES)) - classes_seen)
    if missing:
        raise ValueError(
            f"Classes missing globally across all clients: {missing} "
            f"(class names: {[CLASS_NAMES[c] for c in missing]})"
        )


def _build_long_form(clients: List[List[int]], labels: np.ndarray) -> pd.DataFrame:
    """Long-form table with one row per sample."""
    rows = []
    for cid, idxs in enumerate(clients):
        for i in idxs:
            rows.append({
                "sample_index": int(i),
                "client_id": int(cid),
                "class_id": int(labels[i]),
            })
    # Sort by sample_index for stable, diff-friendly output
    return pd.DataFrame(rows).sort_values("sample_index").reset_index(drop=True)


def class_count_table(clients: List[List[int]], labels: np.ndarray) -> pd.DataFrame:
    """Return a clients × classes count table (rows = clients, cols = classes)."""
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    rows = []
    for cid, idxs in enumerate(clients):
        counts = np.bincount(
            labels[np.asarray(idxs, dtype=int)] if len(idxs) else np.array([], dtype=int),
            minlength=NUM_CLASSES,
        )
        row = {f"class_{c}_{CLASS_NAMES[c][:14]}": int(counts[c]) for c in range(NUM_CLASSES)}
        row["client_id"] = int(cid)
        row["total"] = int(counts.sum())
        rows.append(row)
    df = pd.DataFrame(rows)
    df = df.set_index("client_id")
    cols = [f"class_{c}_{CLASS_NAMES[c][:14]}" for c in range(NUM_CLASSES)] + ["total"]
    return df[cols]


def save_partition(
    partition_df: pd.DataFrame,
    count_table_df: pd.DataFrame,
    out_dir: str | Path,
    mode: str,
    seed: int,
) -> Tuple[Path, Path]:
    """Write partition CSV + count-table CSV to `out_dir`.

    Returns (partition_csv_path, count_table_csv_path).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p_csv = out / f"partition_{mode}_seed{seed}.csv"
    t_csv = out / f"partition_{mode}_seed{seed}_counts.csv"
    partition_df.to_csv(p_csv, index=False)
    count_table_df.to_csv(t_csv)
    return p_csv, t_csv


# ----------------------------------------------------------------------------
# CLI entry point — load DermMNIST, partition, print + save
# ----------------------------------------------------------------------------

def _load_dermmnist_labels(npz_path: str | Path) -> np.ndarray:
    """Load only training labels from the DermMNIST npz (avoids loading images)."""
    data = np.load(npz_path)
    if "train_labels" not in data.files:
        raise ValueError(f"NPZ {npz_path} missing 'train_labels'. Files: {data.files}")
    return np.asarray(data["train_labels"]).reshape(-1).astype(np.int64)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="/Users/basiakoch/cleanest_federated/dermamnist_64.npz",
                    help="Path to dermamnist .npz file")
    ap.add_argument("--mode",
                    choices=["simple_pathological_3_clients",
                             "medical_skew_7_clients",
                             "balanced_specialist_7_clients",
                             "balanced_paired_7_clients",
                             "quantity_skew_improved"],
                    default="medical_skew_7_clients")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="mnist_dermnist/results/partitions")
    args = ap.parse_args()

    labels = _load_dermmnist_labels(args.npz)
    print(f"Loaded {len(labels)} training labels from {args.npz}")
    print(f"Global class counts: {np.bincount(labels, minlength=NUM_CLASSES).tolist()}")
    print()

    if args.mode == "simple_pathological_3_clients":
        clients, df = simple_pathological_3_clients(labels, seed=args.seed)
    elif args.mode == "balanced_specialist_7_clients":
        clients, df = balanced_specialist_7_clients(labels, seed=args.seed)
    elif args.mode == "balanced_paired_7_clients":
        clients, df = balanced_paired_7_clients(labels, seed=args.seed)
    elif args.mode == "quantity_skew_improved":
        clients, df = quantity_skew_improved(labels, seed=args.seed)
    else:
        clients, df = medical_skew_7_clients(labels, seed=args.seed)

    count_table = class_count_table(clients, labels)
    print(f"=== {args.mode} (seed={args.seed}) ===")
    print(count_table.to_string())
    print()

    p_csv, t_csv = save_partition(df, count_table, args.out, mode=args.mode, seed=args.seed)
    print(f"Wrote: {p_csv}")
    print(f"Wrote: {t_csv}")


if __name__ == "__main__":
    main()
