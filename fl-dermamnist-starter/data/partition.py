from __future__ import annotations

from typing import Dict, List, Tuple
import warnings
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from scipy.stats import entropy as scipy_entropy


def get_labels(dataset: Dataset) -> np.ndarray:
    base = getattr(dataset, 'base', dataset)
    if hasattr(base, 'labels'):
        labels = np.asarray(base.labels).reshape(-1)
    elif hasattr(base, 'targets'):
        labels = np.asarray(base.targets).reshape(-1)
    else:
        labels = np.asarray([int(dataset[i][1]) for i in range(len(dataset))])
    return labels.astype(int)


def _validate_partition(client_indices: List[List[int]], n: int) -> None:
    all_idx = [int(i) for client in client_indices for i in client]
    assert len(all_idx) == n, f'Assigned {len(all_idx)} samples, expected {n}'
    assert len(set(all_idx)) == n, 'Duplicate or missing sample indices detected'
    assert set(all_idx) == set(range(n)), 'Partition does not cover exactly all dataset indices'
    assert all(len(c) > 0 for c in client_indices), 'At least one client is empty'


def iid_partition(dataset: Dataset, num_clients: int, seed: int = 42) -> List[List[int]]:
    """Shuffle all dataset indices and split them evenly across clients."""
    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset))
    rng.shuffle(indices)
    splits = np.array_split(indices, num_clients)
    clients = [s.astype(int).tolist() for s in splits]
    _validate_partition(clients, len(dataset))
    return clients


def dirichlet_partition(
    dataset: Dataset,
    num_clients: int,
    alpha: float,
    min_samples_per_client: int = 10,
    seed: int = 42,
    max_retries: int = 100,
) -> List[List[int]]:
    """Partition by label skew using class-wise Dirichlet draws.

    Args:
        dataset: Dataset with labels available via .labels, .targets, or __getitem__.
        num_clients: Number of federated clients.
        alpha: Dirichlet concentration; lower values create stronger label skew.
        min_samples_per_client: Retry until every client has at least this many samples.
        seed: Random seed.
        max_retries: Maximum attempts before raising ValueError.
    """
    labels = get_labels(dataset)
    classes = np.unique(labels)
    rng = np.random.default_rng(seed)

    for _ in range(max_retries):
        client_indices = [[] for _ in range(num_clients)]
        for c in classes:
            idx_c = np.where(labels == c)[0]
            rng.shuffle(idx_c)
            props = rng.dirichlet(alpha * np.ones(num_clients))
            cut_points = (np.cumsum(props)[:-1] * len(idx_c)).astype(int)
            splits = np.split(idx_c, cut_points)
            for cid, split in enumerate(splits):
                client_indices[cid].extend(split.astype(int).tolist())
        if min(len(x) for x in client_indices) >= min_samples_per_client:
            for x in client_indices:
                rng.shuffle(x)
            _validate_partition(client_indices, len(dataset))
            return client_indices

    raise ValueError(
        f'Could not satisfy min_samples={min_samples_per_client} with alpha={alpha} '
        f'and {num_clients} clients after {max_retries} retries. Try larger alpha, '
        f'fewer clients, or smaller min_samples.'
    )


def pathological_partition(
    dataset: Dataset,
    num_clients: int,
    classes_per_client: int = 2,
    seed: int = 42,
    return_assignment: bool = False,
):
    """Assign each client exactly ``classes_per_client`` classes.

    Pass ``return_assignment=True`` to also return the class-client assignment
    matrix with shape (num_clients, num_classes).
    """
    labels = get_labels(dataset)
    classes = np.unique(labels).astype(int)
    num_classes = len(classes)
    rng = np.random.default_rng(seed)

    total_slots = num_clients * classes_per_client
    base = total_slots // num_classes
    extra = total_slots % num_classes
    class_slots = {c: base + (1 if i < extra else 0) for i, c in enumerate(classes)}

    counts = {int(c): int((labels == c).sum()) for c in classes}
    for c, slots in class_slots.items():
        if counts[c] < 50 and slots > 2:
            warnings.warn(f'Rare class {c} with {counts[c]} samples assigned to {slots} clients')

    client_classes = [[] for _ in range(num_clients)]
    # Fill slots by repeatedly assigning the least-filled eligible clients.
    for c in rng.permutation(classes):
        for _ in range(class_slots[int(c)]):
            eligible = [i for i in range(num_clients) if len(client_classes[i]) < classes_per_client and int(c) not in client_classes[i]]
            if not eligible:
                eligible = [i for i in range(num_clients) if len(client_classes[i]) < classes_per_client]
            cid = min(eligible, key=lambda i: len(client_classes[i]))
            client_classes[cid].append(int(c))

    assignment = np.zeros((num_clients, num_classes), dtype=int)
    for cid, cs in enumerate(client_classes):
        assert len(set(cs)) == classes_per_client, 'Client does not have requested distinct classes'
        for c in cs:
            assignment[cid, int(c)] = 1

    client_indices = [[] for _ in range(num_clients)]
    for c in classes:
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        assigned_clients = np.where(assignment[:, int(c)] == 1)[0]
        splits = np.array_split(idx_c, len(assigned_clients))
        for cid, split in zip(assigned_clients, splits):
            client_indices[int(cid)].extend(split.astype(int).tolist())

    for x in client_indices:
        rng.shuffle(x)
    _validate_partition(client_indices, len(dataset))
    if return_assignment:
        return client_indices, assignment
    return client_indices


def quantity_skew_partition(dataset: Dataset, num_clients: int, alpha: float = 0.5, seed: int = 42) -> List[List[int]]:
    """Partition IID labels with unequal client dataset sizes."""
    rng = np.random.default_rng(seed)
    n = len(dataset)
    proportions = rng.dirichlet(alpha * np.ones(num_clients))
    sizes = np.floor(proportions * n).astype(int)
    while sizes.sum() < n:
        sizes[np.argmin(sizes)] += 1
    while sizes.sum() > n:
        sizes[np.argmax(sizes)] -= 1
    indices = np.arange(n)
    rng.shuffle(indices)
    splits = []
    start = 0
    for size in sizes:
        splits.append(indices[start:start + size].astype(int).tolist())
        start += size
    _validate_partition(splits, n)
    return splits


def dirichlet_quantity_partition(
    dataset: Dataset,
    num_clients: int,
    label_alpha: float = 0.5,
    quantity_alpha: float = 0.5,
    seed: int = 42,
    min_samples_per_client: int = 10,
    max_retries: int = 100,
) -> List[List[int]]:
    """Combine quantity skew and label skew while preserving each sample once.

    Quantity skew is applied FIRST (target client sizes ~ Dir(quantity_alpha));
    THEN label skew is applied per class so each client's allocation reflects a
    Dir(label_alpha) preference, weighted by its target size.
    """
    rng = np.random.default_rng(seed)
    labels = get_labels(dataset)
    n = len(dataset)
    num_classes = int(labels.max()) + 1

    for _ in range(max_retries):
        # Step 1 — quantity skew: target sample count per client.
        quantity_props = rng.dirichlet(quantity_alpha * np.ones(num_clients))
        targets = np.maximum(np.floor(quantity_props * n).astype(int), 1)
        while targets.sum() < n:
            targets[int(np.argmin(targets))] += 1
        while targets.sum() > n:
            j = int(np.argmax(targets))
            if targets[j] > 1:
                targets[j] -= 1
            else:
                break

        # Step 2 — label skew per client: each client gets a Dirichlet label preference.
        # Weight by target size so a client's expected total matches its quantity target.
        client_label_props = rng.dirichlet(label_alpha * np.ones(num_classes), size=num_clients)
        demand = client_label_props * targets[:, None]  # (num_clients, num_classes)

        class_indices = {c: np.where(labels == c)[0].copy() for c in range(num_classes)}
        for c in class_indices:
            rng.shuffle(class_indices[c])

        client_indices: List[List[int]] = [[] for _ in range(num_clients)]
        for c in range(num_classes):
            idx = class_indices[c]
            if len(idx) == 0:
                continue
            class_demand = demand[:, c]
            if class_demand.sum() <= 0:
                class_demand = targets.astype(float)
            props = class_demand / class_demand.sum()
            splits = (np.cumsum(props) * len(idx)).astype(int)[:-1]
            chunks = np.split(idx, splits)
            for i, chunk in enumerate(chunks):
                client_indices[i].extend(int(x) for x in chunk)

        if all(len(c) >= min_samples_per_client for c in client_indices):
            for x in client_indices:
                rng.shuffle(x)
            _validate_partition(client_indices, n)
            return client_indices

    raise ValueError(
        f"Could not satisfy min_samples={min_samples_per_client} with "
        f"label_alpha={label_alpha}, quantity_alpha={quantity_alpha} and "
        f"{num_clients} clients after {max_retries} retries. Try larger alphas, "
        f"fewer clients, or smaller min_samples."
    )


def get_client_class_distribution(dataset: Dataset, client_indices: List[int]) -> Dict[int, int]:
    """Return class counts for a single client's index list."""
    labels = get_labels(dataset)
    subset_labels = labels[np.asarray(client_indices, dtype=int)]
    vals, counts = np.unique(subset_labels, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, counts)}


def get_all_client_distributions(dataset: Dataset, all_client_indices: List[List[int]]) -> pd.DataFrame:
    """Return a clients x classes DataFrame of local class counts."""
    labels = get_labels(dataset)
    classes = np.unique(labels).astype(int)
    rows = []
    for cid, idxs in enumerate(all_client_indices):
        dist = get_client_class_distribution(dataset, idxs)
        rows.append({int(c): int(dist.get(int(c), 0)) for c in classes})
    df = pd.DataFrame(rows)
    df.index.name = 'client_id'
    return df


def compute_distribution_entropy(client_distribution) -> float:
    """Return Shannon entropy of a client's normalised class distribution."""
    arr = np.asarray(client_distribution, dtype=float)
    total = arr.sum()
    if total <= 0:
        return 0.0
    p = arr / total
    return float(scipy_entropy(p, base=2))


def specialist_partition(dataset: Dataset, num_clients: int, seed: int = 42) -> List[List[int]]:
    """Specialist clients: each client gets all samples of exactly one class.

    Setup follows Zhao et al. 2018 ("Federated Learning with Non-IID Data") and
    the extreme pathological case in Li et al. 2020 (FedProx). Each client
    `c` receives every training sample with label `c`. num_clients must equal
    the number of classes in the dataset.

    The natural class imbalance is preserved (no subsampling): client sizes
    follow the global class counts. On DermaMNIST this means client sizes
    differ by up to 58×, with the Melanocytic-Nevi specialist holding ~67%
    of all training samples. This is the most extreme case for FedAvg's
    size-weighted aggregation.
    """
    labels = get_labels(dataset)
    num_classes = int(labels.max()) + 1
    if num_clients != num_classes:
        raise ValueError(
            f'specialist_partition requires num_clients == num_classes '
            f'(got num_clients={num_clients}, num_classes={num_classes}).'
        )
    rng = np.random.default_rng(seed)
    client_indices: List[List[int]] = []
    for c in range(num_classes):
        idxs = np.where(labels == c)[0]
        rng.shuffle(idxs)
        client_indices.append(idxs.astype(int).tolist())
    _validate_partition(client_indices, len(dataset))
    return client_indices


def mixed_type_partition(dataset: Dataset, num_clients: int = 10, seed: int = 42,
                         hospital_ids=(0, 1, 2),
                         specialist_assignments=((3, 0), (4, 1), (5, 3), (6, 6)),
                         mixed_ids=(7, 8, 9),
                         specialist_dominance: float = 0.70,
                         hospital_common_share: float = 0.90,
                         common_classes=(2, 4, 5),
                         min_samples_per_client: int = 10) -> List[List[int]]:
    """Mixed-type partition for the thesis FedProx stress experiment.

    Creates exactly 10 clients with three roles:
      - 3 large hospital clients (IDs 0,1,2): mostly common classes (2,4,5)
      - 4 specialist clinics (IDs 3-6): each dominated by one rare class
        Default assignments: client 3 -> class 0 (actinic keratoses),
                             client 4 -> class 1 (basal cell carcinoma),
                             client 5 -> class 3 (dermatofibroma),
                             client 6 -> class 6 (vascular lesions).
      - 3 mixed clients (IDs 7,8,9): receive leftover samples from all classes

    Procedure:
      1. For each specialist, take `specialist_dominance` (default 70%) of its
         dominant class.
      2. For each common class, give `hospital_common_share` (default 90%) of
         remaining samples split evenly across the 3 hospitals.
      3. Round-robin remaining samples to mixed clients (and specialists as
         filler) so every class is non-empty globally and every client has
         at least `min_samples_per_client` samples.

    The partition is deterministic given a seed but produces different
    partitions for different seeds.
    """
    assert num_clients == 10, 'mixed_type_partition currently requires exactly 10 clients'
    rng = np.random.default_rng(seed)
    labels = get_labels(dataset)
    num_classes = int(labels.max()) + 1
    assert num_classes == 7, f'mixed_type_partition assumes 7 classes (got {num_classes})'

    # Permute class indices reproducibly
    class_idxs: Dict[int, np.ndarray] = {}
    for c in range(num_classes):
        idxs = np.where(labels == c)[0]
        rng.shuffle(idxs)
        class_idxs[c] = idxs

    clients: List[List[int]] = [[] for _ in range(num_clients)]

    # Step 1 — specialists take their dominant class share
    for client_id, dominant_class in specialist_assignments:
        idxs = class_idxs[dominant_class]
        n_take = max(1, int(len(idxs) * specialist_dominance))
        clients[client_id].extend(idxs[:n_take].astype(int).tolist())
        class_idxs[dominant_class] = idxs[n_take:]

    # Step 2 — hospitals split common-class samples evenly.
    # IMPORTANT: common-class leftovers go BACK to hospitals (not to specialists)
    # to avoid drowning specialists' dominant class with majority-class filler.
    for c in common_classes:
        idxs = class_idxs[c]
        # Split ALL common-class samples evenly across hospitals
        chunks = np.array_split(idxs, len(hospital_ids))
        for h, chunk in zip(hospital_ids, chunks):
            clients[h].extend(chunk.astype(int).tolist())
        class_idxs[c] = np.array([], dtype=int)   # fully consumed

    # Step 3 — distribute remaining rare-class samples round-robin to
    # mixed clients + specialists. These are minority-heavy by construction.
    remaining = []
    for c in range(num_classes):
        for idx in class_idxs[c]:
            remaining.append(int(idx))
    rng.shuffle(remaining)

    target_clients = list(mixed_ids) + [c for c, _ in specialist_assignments]
    for i, idx in enumerate(remaining):
        clients[target_clients[i % len(target_clients)]].append(idx)

    # Final validations
    _validate_partition(clients, len(dataset))

    # Per-spec: every class must appear globally, every client must have enough samples
    for c in range(num_classes):
        if not any(c in [int(labels[i]) for i in cl] for cl in clients):
            raise ValueError(f'Class {c} missing globally — partition invalid')
    for cid, cl in enumerate(clients):
        if len(cl) < min_samples_per_client:
            raise ValueError(
                f'Client {cid} has only {len(cl)} samples (< min_samples_per_client={min_samples_per_client}). '
                f'Try smaller specialist_dominance or hospital_common_share.'
            )

    return clients


def print_partition_table(dataset: Dataset, client_indices: List[List[int]], class_names=None) -> pd.DataFrame:
    """Pretty-print a clients × classes count table. Returns the DataFrame."""
    labels = get_labels(dataset)
    num_classes = int(labels.max()) + 1
    rows = []
    for cid, cl in enumerate(client_indices):
        cl_labels = labels[np.asarray(cl, dtype=int)]
        row = {f'class_{c}': int((cl_labels == c).sum()) for c in range(num_classes)}
        row['client_id'] = cid
        row['total'] = len(cl)
        rows.append(row)
    df = pd.DataFrame(rows).set_index('client_id')
    cols_order = [f'class_{c}' for c in range(num_classes)] + ['total']
    df = df[cols_order]
    if class_names is not None:
        df.columns = list(class_names) + ['total']
    return df


def make_partition(dataset: Dataset, strategy: str, num_clients: int, seed: int = 42, **kwargs) -> List[List[int]]:
    strategy = strategy.lower()
    if strategy == 'iid':
        return iid_partition(dataset, num_clients, seed=seed)
    if strategy == 'dirichlet':
        return dirichlet_partition(dataset, num_clients, alpha=float(kwargs.get('alpha', 0.5)), seed=seed)
    if strategy == 'pathological':
        return pathological_partition(dataset, num_clients, classes_per_client=int(kwargs.get('classes_per_client', 2)), seed=seed)
    if strategy == 'quantity_skew':
        return quantity_skew_partition(dataset, num_clients, alpha=float(kwargs.get('quantity_alpha', kwargs.get('alpha', 0.5))), seed=seed)
    if strategy in ['dirichlet_quantity', 'dirichlet_quantity_skew']:
        return dirichlet_quantity_partition(
            dataset,
            num_clients,
            label_alpha=float(kwargs.get('label_alpha', kwargs.get('alpha', 0.5))),
            quantity_alpha=float(kwargs.get('quantity_alpha', 0.5)),
            seed=seed,
        )
    if strategy == 'specialist':
        return specialist_partition(dataset, num_clients, seed=seed)
    if strategy in ('mixed_type', 'mixed'):
        return mixed_type_partition(
            dataset,
            num_clients,
            seed=seed,
            specialist_dominance=float(kwargs.get('specialist_dominance', 0.70)),
            hospital_common_share=float(kwargs.get('hospital_common_share', 0.90)),
            min_samples_per_client=int(kwargs.get('min_samples_per_client', 10)),
        )
    raise ValueError(f'Unknown partition strategy: {strategy}')
