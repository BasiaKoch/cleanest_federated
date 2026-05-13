"""Tests for the thesis mixed-type partition."""
from __future__ import annotations

import numpy as np
import pytest
from torch.utils.data import Dataset

from data.partition import mixed_type_partition, get_labels, print_partition_table


class SyntheticDataset(Dataset):
    """Mimics DermaMNIST class proportions at 1/2 scale."""
    def __init__(self):
        counts = [114, 180, 384, 40, 390, 2346, 50]
        labels = []
        for c, n in enumerate(counts):
            labels.extend([c] * n)
        rng = np.random.default_rng(0)
        rng.shuffle(labels)
        self.labels = np.array(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return idx, int(self.labels[idx])


@pytest.fixture
def dataset():
    return SyntheticDataset()


def test_partition_assigns_all_samples_exactly_once(dataset):
    parts = mixed_type_partition(dataset, num_clients=10, seed=42)
    all_idx = [i for cl in parts for i in cl]
    assert len(all_idx) == len(dataset)
    assert len(set(all_idx)) == len(all_idx), 'duplicate indices'
    assert set(all_idx) == set(range(len(dataset)))


def test_partition_has_10_clients(dataset):
    parts = mixed_type_partition(dataset, num_clients=10, seed=42)
    assert len(parts) == 10


def test_every_class_appears_globally(dataset):
    parts = mixed_type_partition(dataset, num_clients=10, seed=42)
    labels = get_labels(dataset)
    classes_seen = set()
    for cl in parts:
        classes_seen.update(int(labels[i]) for i in cl)
    assert classes_seen == set(range(7)), f'classes missing globally: {set(range(7)) - classes_seen}'


def test_every_client_meets_min_samples(dataset):
    parts = mixed_type_partition(dataset, num_clients=10, seed=42, min_samples_per_client=10)
    for cid, cl in enumerate(parts):
        assert len(cl) >= 10, f'client {cid} has only {len(cl)} samples'


def test_specialist_clients_have_their_assigned_class_as_modal(dataset):
    """Clients 3-6 should have their assigned class as the MOST FREQUENT class.

    Note: for very rare classes (dermatofibroma n=80 on real DermaMNIST),
    even 70% of all available samples is dwarfed by round-robin filler from
    leftover samples across 7 classes. The defensible test is that the
    assigned class is the *modal* class for the specialist, not a strict
    percentage threshold.
    """
    parts = mixed_type_partition(dataset, num_clients=10, seed=42)
    labels = get_labels(dataset)
    spec_map = {3: 0, 4: 1, 5: 3, 6: 6}
    for client_id, dom_class in spec_map.items():
        cl = parts[client_id]
        counts = np.bincount([int(labels[i]) for i in cl], minlength=7)
        modal_class = int(np.argmax(counts))
        assert modal_class == dom_class, \
            f'specialist {client_id}: modal class is {modal_class}, expected {dom_class}. counts={counts.tolist()}'


def test_hospitals_have_more_samples_than_specialists(dataset):
    parts = mixed_type_partition(dataset, num_clients=10, seed=42)
    sizes = [len(p) for p in parts]
    hosp_avg = np.mean([sizes[0], sizes[1], sizes[2]])
    spec_avg = np.mean([sizes[3], sizes[4], sizes[5], sizes[6]])
    assert hosp_avg > spec_avg * 2, f'hospitals should be ≥2× specialists (got {hosp_avg:.0f} vs {spec_avg:.0f})'


def test_different_seeds_produce_different_partitions(dataset):
    a = mixed_type_partition(dataset, num_clients=10, seed=42)
    b = mixed_type_partition(dataset, num_clients=10, seed=123)
    # At least one client should have different membership
    assert any(set(a[i]) != set(b[i]) for i in range(10)), 'partitions are identical across seeds'


def test_same_seed_produces_identical_partition(dataset):
    a = mixed_type_partition(dataset, num_clients=10, seed=42)
    b = mixed_type_partition(dataset, num_clients=10, seed=42)
    for ca, cb in zip(a, b):
        assert ca == cb, 'reproducibility violated'


def test_print_partition_table_shape(dataset):
    parts = mixed_type_partition(dataset, num_clients=10, seed=42)
    df = print_partition_table(dataset, parts)
    assert df.shape == (10, 8), f'expected 10×8 table, got {df.shape}'
    assert df['total'].sum() == len(dataset)
