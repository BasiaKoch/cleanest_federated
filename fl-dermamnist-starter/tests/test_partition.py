import numpy as np
import pytest
from torch.utils.data import Dataset
from data.partition import (
    iid_partition, dirichlet_partition, pathological_partition, quantity_skew_partition,
    dirichlet_quantity_partition, get_all_client_distributions, compute_distribution_entropy,
)


class SyntheticDataset(Dataset):
    def __init__(self):
        counts = [400, 150, 120, 30, 130, 100, 70]
        self.labels = np.concatenate([np.full(c, i) for i, c in enumerate(counts)])
        self.targets = self.labels.tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return np.zeros((3, 28, 28), dtype=np.float32), int(self.labels[idx])


def assert_valid(ds, parts):
    all_idx = [i for p in parts for i in p]
    assert len(all_idx) == len(ds)
    assert len(set(all_idx)) == len(ds)
    assert all(len(p) > 0 for p in parts)
    df = get_all_client_distributions(ds, parts)
    global_counts = np.bincount(ds.labels, minlength=7)
    assert np.all(df.sum(axis=0).values == global_counts)


@pytest.mark.parametrize('fn,kwargs', [
    (iid_partition, {}),
    (dirichlet_partition, {'alpha': 0.5}),
    (pathological_partition, {'classes_per_client': 2}),
    (quantity_skew_partition, {'alpha': 0.5}),
    (dirichlet_quantity_partition, {'label_alpha': 0.5, 'quantity_alpha': 0.5}),
])
def test_partitions_valid(fn, kwargs):
    ds = SyntheticDataset()
    parts = fn(ds, 10, seed=42, **kwargs)
    assert_valid(ds, parts)


@pytest.mark.parametrize('fn,kwargs', [
    (iid_partition, {}),
    (dirichlet_partition, {'alpha': 0.5}),
    (pathological_partition, {'classes_per_client': 2}),
    (quantity_skew_partition, {'alpha': 0.5}),
    (dirichlet_quantity_partition, {'label_alpha': 0.5, 'quantity_alpha': 0.5}),
])
def test_reproducibility(fn, kwargs):
    ds = SyntheticDataset()
    p1 = fn(ds, 10, seed=123, **kwargs)
    p2 = fn(ds, 10, seed=123, **kwargs)
    assert p1 == p2


def test_dirichlet_high_alpha_approx_iid():
    ds = SyntheticDataset()
    parts = dirichlet_partition(ds, 10, alpha=100.0, seed=42)
    df = get_all_client_distributions(ds, parts)
    props = df.div(df.sum(axis=1), axis=0).values
    global_props = np.bincount(ds.labels, minlength=7) / len(ds)
    assert np.max(np.abs(props - global_props)) < 0.12


def test_dirichlet_low_alpha_more_heterogeneous_than_high_alpha():
    ds = SyntheticDataset()
    low = dirichlet_partition(ds, 10, alpha=0.1, seed=42)
    high = dirichlet_partition(ds, 10, alpha=1.0, seed=42)
    low_df = get_all_client_distributions(ds, low)
    high_df = get_all_client_distributions(ds, high)
    low_ent = np.mean([compute_distribution_entropy(row.values) for _, row in low_df.iterrows()])
    high_ent = np.mean([compute_distribution_entropy(row.values) for _, row in high_df.iterrows()])
    assert low_ent < high_ent
    assert (low_df.div(low_df.sum(axis=1), axis=0).max(axis=1) > 0.60).any()


def test_dirichlet_min_samples_enforced():
    ds = SyntheticDataset()
    parts = dirichlet_partition(ds, 10, alpha=0.5, min_samples_per_client=10, seed=42)
    assert min(len(p) for p in parts) >= 10


def test_dirichlet_impossible_raises():
    ds = SyntheticDataset()
    with pytest.raises(ValueError):
        dirichlet_partition(ds, 1000, alpha=0.001, min_samples_per_client=100, seed=42, max_retries=2)


def test_pathological_class_count():
    ds = SyntheticDataset()
    parts = pathological_partition(ds, 10, classes_per_client=2, seed=42)
    for part in parts:
        assert len(set(ds.labels[part])) == 2


def test_pathological_rare_class_warning():
    ds = SyntheticDataset()
    with pytest.warns(UserWarning, match='Rare class'):
        pathological_partition(ds, 10, classes_per_client=2, seed=42)
