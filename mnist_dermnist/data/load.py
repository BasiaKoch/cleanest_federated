"""DermMNIST loader — reads the 64×64 npz and (optionally) resizes to 28×28.

Returns `(train_ds, val_ds, test_ds)` with deterministic transforms. Labels
are returned as torch long scalars (not `[N,1]`).
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DermMNISTDataset(Dataset):
    """In-memory wrapper around the standard MedMNIST npz arrays.

    Args:
        images: uint8 NHWC array.
        labels: int label array, shape (N,) or (N, 1).
        target_size: if set and different from the source spatial size, bilinear
                     resize is applied per-item.
        normalize: apply ImageNet mean/std after scaling to [0,1].
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray, *,
                 target_size: int | None = None, normalize: bool = True):
        self.images = np.asarray(images)
        self.labels = np.asarray(labels).reshape(-1).astype(np.int64)
        self.target_size = int(target_size) if target_size else None
        self.normalize = normalize
        if normalize:
            self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
            self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        img = torch.as_tensor(self.images[idx]).permute(2, 0, 1).float() / 255.0  # CHW
        if self.target_size is not None and img.shape[-1] != self.target_size:
            img = F.interpolate(img.unsqueeze(0), size=self.target_size,
                                mode="bilinear", align_corners=False).squeeze(0)
        if self.normalize:
            img = (img - self.mean) / self.std
        y = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return img, y


def load_dermmnist(npz_path: str | Path, *, image_size: int = 28
                   ) -> Tuple[DermMNISTDataset, DermMNISTDataset, DermMNISTDataset]:
    """Load (train, val, test) DermMNIST splits with the given image_size."""
    p = Path(npz_path)
    if not p.exists():
        raise FileNotFoundError(f"DermMNIST npz not found at {p}")
    data = np.load(p)
    return (
        DermMNISTDataset(data["train_images"], data["train_labels"], target_size=image_size),
        DermMNISTDataset(data["val_images"],   data["val_labels"],   target_size=image_size),
        DermMNISTDataset(data["test_images"],  data["test_labels"],  target_size=image_size),
    )
