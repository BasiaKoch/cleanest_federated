"""DermaMNIST loading + Dirichlet partitioning.

Mirrors prepare_data.py from the reference repo (which did breast-MRI loading +
split_data) but adapted for DermaMNIST classification with a non-IID partition.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DermaMNISTDataset(Dataset):
    """Standard MedMNIST .npz format wrapper.

    Each item returns (image_chw_float, label_long_scalar). Normalization uses
    ImageNet statistics — fine for natural-image-like medical photographs.
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray, normalize: bool = True):
        self.images = np.asarray(images)
        self.labels = np.asarray(labels).reshape(-1).astype(np.int64)
        self.normalize = normalize
        if normalize:
            self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
            self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        # HWC uint8 → CHW float32 / 255
        x = torch.as_tensor(self.images[idx]).permute(2, 0, 1).float() / 255.0
        if self.normalize:
            x = (x - self.mean) / self.std
        y = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return x, y


def load_dermamnist(npz_path: str) -> Tuple[DermaMNISTDataset, DermaMNISTDataset, DermaMNISTDataset]:
    """Returns (train, val, test) datasets."""
    path = Path(npz_path)
    if not path.exists():
        raise FileNotFoundError(f"DermaMNIST npz not found at {path}")
    data = np.load(path)
    required = ["train_images", "train_labels", "val_images", "val_labels", "test_images", "test_labels"]
    missing = [k for k in required if k not in data.files]
    if missing:
        raise ValueError(f"NPZ missing arrays: {missing}")
    return (
        DermaMNISTDataset(data["train_images"], data["train_labels"]),
        DermaMNISTDataset(data["val_images"],   data["val_labels"]),
        DermaMNISTDataset(data["test_images"],  data["test_labels"]),
    )


def dirichlet_partition(
    dataset: DermaMNISTDataset,
    num_clients: int,
    alpha: float,
    seed: int = 42,
    min_samples_per_client: int = 5,
    max_retries: int = 100,
) -> List[List[int]]:
    """Non-IID label-skew partition (Hsu et al. 2019).

    For each class c, draw proportions p_c ~ Dir(alpha * 1_K) and assign that
    class's samples to clients accordingly. Retries up to `max_retries` if any
    client ends up with fewer than `min_samples_per_client`.
    """
    labels = dataset.labels
    num_classes = int(labels.max()) + 1

    for _ in range(max_retries):
        rng = np.random.default_rng(seed)
        client_idxs: List[List[int]] = [[] for _ in range(num_clients)]
        for c in range(num_classes):
            cls_idxs = np.where(labels == c)[0]
            rng.shuffle(cls_idxs)
            props = rng.dirichlet(alpha * np.ones(num_clients))
            splits = (np.cumsum(props) * len(cls_idxs)).astype(int)[:-1]
            for k, chunk in enumerate(np.split(cls_idxs, splits)):
                client_idxs[k].extend(chunk.astype(int).tolist())
        if all(len(cl) >= min_samples_per_client for cl in client_idxs):
            # validate
            all_idx = [i for cl in client_idxs for i in cl]
            assert len(all_idx) == len(labels) and len(set(all_idx)) == len(all_idx)
            return client_idxs
        seed += 1  # bump seed and retry

    raise ValueError(
        f"Could not satisfy min_samples_per_client={min_samples_per_client} with "
        f"alpha={alpha}, num_clients={num_clients} in {max_retries} retries."
    )


def print_distribution(dataset: DermaMNISTDataset, client_idxs: List[List[int]]) -> None:
    """Print a clients × classes count table."""
    labels = dataset.labels
    num_classes = int(labels.max()) + 1
    print(f"{'client':>7} | " + " ".join(f"{c:>5}" for c in range(num_classes)) + " | total")
    print("-" * (10 + 6 * num_classes + 10))
    for k, idxs in enumerate(client_idxs):
        counts = np.bincount(labels[np.asarray(idxs, dtype=int)], minlength=num_classes)
        print(f"     C{k} | " + " ".join(f"{int(n):>5}" for n in counts) + f" | {sum(counts):>5}")
