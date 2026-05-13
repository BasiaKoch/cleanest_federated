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
    ap.add_argument("--mode", choices=["simple_pathological_3_clients", "medical_skew_7_clients"],
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
