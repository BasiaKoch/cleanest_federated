from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from data.download import MedMNISTNPZDataset, load_dermamnist, load_dermamnist_npz


def _write_npz(path: Path) -> None:
    rng = np.random.default_rng(123)
    np.savez(
        path,
        train_images=rng.integers(0, 256, size=(3, 4, 5, 3), dtype=np.uint8),
        train_labels=np.array([[0], [1], [2]], dtype=np.uint8),
        val_images=rng.integers(0, 256, size=(2, 4, 5, 3), dtype=np.uint8),
        val_labels=np.array([[3], [4]], dtype=np.uint8),
        test_images=rng.integers(0, 256, size=(1, 4, 5, 3), dtype=np.uint8),
        test_labels=np.array([[5]], dtype=np.uint8),
    )


def test_medmnist_npz_dataset_returns_chw_float_image_and_long_label():
    images = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(1, 4, 5, 3)
    labels = np.array([[6]], dtype=np.uint8)
    dataset = MedMNISTNPZDataset(images, labels, normalize=False)

    image, label = dataset[0]

    assert image.dtype == torch.float32
    assert image.shape == (3, 4, 5)
    assert torch.allclose(image, torch.as_tensor(images[0]).permute(2, 0, 1).float() / 255.0)
    assert label.dtype == torch.long
    assert label.ndim == 0
    assert label.item() == 6
    assert dataset.labels.tolist() == [6]


def test_load_dermamnist_npz_supports_standard_medmnist_keys(tmp_path):
    npz_path = tmp_path / 'dermamnist.npz'
    _write_npz(npz_path)

    train, val, test = load_dermamnist_npz(npz_path, normalize=False)

    assert len(train) == 3
    assert len(val) == 2
    assert len(test) == 1
    assert train[0][0].shape == (3, 4, 5)
    assert train.labels.tolist() == [0, 1, 2]
    assert val.labels.tolist() == [3, 4]
    assert test.labels.tolist() == [5]


def test_load_dermamnist_can_use_npz_source(tmp_path):
    npz_path = tmp_path / 'dermamnist.npz'
    _write_npz(npz_path)

    train, _, _ = load_dermamnist(source='npz', npz_path=npz_path)

    image, label = train[0]
    assert image.shape == (3, 4, 5)
    assert label.item() == 0


def test_batch_size_one_labels_are_safe_with_view(tmp_path):
    npz_path = tmp_path / 'dermamnist.npz'
    _write_npz(npz_path)
    train, _, _ = load_dermamnist_npz(npz_path, normalize=False)
    loader = DataLoader(train, batch_size=1, shuffle=False)

    _, labels = next(iter(loader))
    labels = labels.view(-1).long()

    assert labels.shape == (1,)
    assert labels.dtype == torch.long
    assert labels.item() == 0


def test_load_dermamnist_npz_rejects_missing_required_arrays(tmp_path):
    npz_path = tmp_path / 'bad.npz'
    np.savez(npz_path, train_images=np.zeros((1, 4, 5, 3), dtype=np.uint8))

    with pytest.raises(ValueError, match='missing required MedMNIST arrays'):
        load_dermamnist_npz(npz_path)
