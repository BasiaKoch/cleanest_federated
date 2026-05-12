from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset


DERMAMNIST_NPZ_PATH = Path('datasets/medmnist/dermamnist.npz')
MEDMNIST_NPZ_KEYS = (
    'train_images',
    'train_labels',
    'val_images',
    'val_labels',
    'test_images',
    'test_labels',
)


def _dermamnist_stats() -> Tuple[List[float], List[float]]:
    # Consistent normalization for all supported DermaMNIST sizes.
    return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def _image_to_chw_float_tensor(image) -> torch.Tensor:
    tensor = torch.as_tensor(image)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 3:
        if tensor.shape[-1] in (1, 3, 4):
            tensor = tensor.permute(2, 0, 1)
        elif tensor.shape[0] not in (1, 3, 4):
            raise ValueError(f'Unsupported image shape: {tuple(tensor.shape)}')
    else:
        raise ValueError(f'Unsupported image shape: {tuple(tensor.shape)}')

    tensor = tensor.contiguous().float()
    if not torch.is_floating_point(torch.as_tensor(image)):
        tensor = tensor / 255.0
    return tensor


def _normalize_image_tensor(image: torch.Tensor, mean: List[float], std: List[float]) -> torch.Tensor:
    if image.ndim != 3:
        raise ValueError(f'Expected image tensor with shape (C, H, W), got {tuple(image.shape)}')
    channels = image.shape[0]
    if channels > len(mean) or channels > len(std):
        raise ValueError(f'No normalization statistics available for {channels} channels')
    mean_tensor = torch.as_tensor(mean[:channels], dtype=image.dtype, device=image.device).view(channels, 1, 1)
    std_tensor = torch.as_tensor(std[:channels], dtype=image.dtype, device=image.device).view(channels, 1, 1)
    return (image - mean_tensor) / std_tensor


class SqueezeLabelDataset(Dataset):
    """Wrap a dataset so labels are returned as scalar long tensors."""

    def __init__(self, base: Dataset):
        self.base = base
        for attr in ['labels', 'targets']:
            if hasattr(base, attr):
                setattr(self, attr, getattr(base, attr))

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        y = torch.as_tensor(y).view(-1)[0].long()
        return x, y


class MedMNISTNPZDataset(Dataset):
    """Dataset for standard MedMNIST .npz split arrays."""

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        *,
        normalize: bool = True,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
    ):
        if len(images) != len(labels):
            raise ValueError(f'Image/label length mismatch: {len(images)} images, {len(labels)} labels')
        self.images = np.asarray(images)
        self.labels = np.asarray(labels).reshape(-1).astype(np.int64)
        self.targets = self.labels.tolist()
        self.normalize = normalize
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        x = _image_to_chw_float_tensor(self.images[idx])
        if self.normalize:
            if self.mean is None or self.std is None:
                raise ValueError('mean and std are required when normalize=True')
            x = _normalize_image_tensor(x, self.mean, self.std)
        y = torch.as_tensor(self.labels[idx]).long()
        return x, y


def load_mnist(root: str = './data_store') -> Tuple[Dataset, Dataset]:
    from torchvision import datasets, transforms
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train = datasets.MNIST(root=root, train=True, download=True, transform=tfm)
    test = datasets.MNIST(root=root, train=False, download=True, transform=tfm)
    return train, test


def _validate_medmnist_npz(data: np.lib.npyio.NpzFile, path: Path) -> None:
    missing = [key for key in MEDMNIST_NPZ_KEYS if key not in data.files]
    if missing:
        raise ValueError(f'{path} is missing required MedMNIST arrays: {missing}')


def load_dermamnist_npz(
    npz_path: str | Path = DERMAMNIST_NPZ_PATH,
    *,
    normalize: bool = True,
) -> Tuple[Dataset, Dataset, Dataset]:
    path = Path(npz_path)
    mean, std = _dermamnist_stats()
    with np.load(path) as data:
        _validate_medmnist_npz(data, path)
        train = MedMNISTNPZDataset(data['train_images'], data['train_labels'], normalize=normalize, mean=mean, std=std)
        val = MedMNISTNPZDataset(data['val_images'], data['val_labels'], normalize=normalize, mean=mean, std=std)
        test = MedMNISTNPZDataset(data['test_images'], data['test_labels'], normalize=normalize, mean=mean, std=std)
    return train, val, test


def load_dermamnist(
    size: int = 28,
    root: str = './data_store',
    *,
    source: str = 'package',
    npz_path: str | Path = DERMAMNIST_NPZ_PATH,
    download: bool = True,
) -> Tuple[Dataset, Dataset, Dataset]:
    source = source.lower()
    if source not in {'package', 'npz', 'auto'}:
        raise ValueError("source must be one of {'package', 'npz', 'auto'}")
    if source == 'npz' or (source == 'auto' and Path(npz_path).exists()):
        return load_dermamnist_npz(npz_path)

    from torchvision import transforms
    from medmnist import DermaMNIST

    assert size in [28, 64, 128, 224], 'size must be one of [28, 64, 128, 224]'
    Path(root).mkdir(parents=True, exist_ok=True)
    mean, std = _dermamnist_stats()
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    train = SqueezeLabelDataset(DermaMNIST(split='train', transform=tfm, download=download, size=size, root=root))
    val = SqueezeLabelDataset(DermaMNIST(split='val', transform=tfm, download=download, size=size, root=root))
    test = SqueezeLabelDataset(DermaMNIST(split='test', transform=tfm, download=download, size=size, root=root))
    return train, val, test


def get_labels(dataset: Dataset) -> np.ndarray:
    base = getattr(dataset, 'base', dataset)
    if hasattr(base, 'labels'):
        labels = np.asarray(base.labels).reshape(-1)
    elif hasattr(base, 'targets'):
        labels = np.asarray(base.targets).reshape(-1)
    else:
        labels = np.asarray([int(dataset[i][1]) for i in range(len(dataset))])
    return labels.astype(int)


def get_class_distribution(dataset: Dataset) -> Dict[int, int]:
    labels = get_labels(dataset)
    values, counts = np.unique(labels, return_counts=True)
    return {int(v): int(c) for v, c in zip(values, counts)}


def get_dataset_info(dataset_name: str) -> Dict:
    dataset_name = dataset_name.lower()
    if dataset_name == 'dermamnist':
        from medmnist import INFO
        info = INFO['dermamnist']
        labels = info['label']
        class_names = [labels[str(i)] for i in range(len(labels))]
        return {
            'num_classes': len(class_names),
            'class_names': class_names,
            'input_channels': 3,
            'supported_sizes': [28, 64, 128, 224],
        }
    if dataset_name == 'mnist':
        return {
            'num_classes': 10,
            'class_names': [str(i) for i in range(10)],
            'input_channels': 1,
            'supported_sizes': [28],
        }
    raise ValueError(f'Unknown dataset: {dataset_name}')


if __name__ == '__main__':
    train, val, test = load_dermamnist(size=28)
    info = get_dataset_info('dermamnist')
    dist = get_class_distribution(train)
    print(f'DermaMNIST - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}')
    print('Expected official split: 7007 / 1003 / 2005')
    for idx, name in enumerate(info['class_names']):
        count = dist.get(idx, 0)
        print(f'{idx}: {name}: {count}')
    counts = np.array([dist.get(i, 0) for i in range(info['num_classes'])])
    print(f'Imbalance ratio: {counts.max() / max(counts.min(), 1):.2f}')
    melanoma_idx = [i for i, n in enumerate(info['class_names']) if 'melanoma' in n.lower()][0]
    print(f'Melanoma count: {dist.get(melanoma_idx, 0)}')
    mnist_train, mnist_test = load_mnist()
    print(f'MNIST - Train: {len(mnist_train)}, Test: {len(mnist_test)}')
