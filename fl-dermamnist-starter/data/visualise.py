from __future__ import annotations

from pathlib import Path
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ModuleNotFoundError:
    sns = None
from data.download import get_class_distribution, get_labels


def _save(fig, save_path):
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig


def plot_global_class_distribution(dataset, class_names, save_path, figsize=(7, 4)):
    dist = get_class_distribution(dataset)
    counts = np.array([dist.get(i, 0) for i in range(len(class_names))])
    pct = counts / counts.sum() * 100
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.Blues((counts - counts.min()) / (counts.max() - counts.min() + 1e-12))
    y = np.arange(len(class_names))
    ax.barh(y, counts, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(class_names)
    ax.invert_yaxis()
    ax.set_xlabel('Samples')
    ax.set_title('Global Class Distribution')
    for i, (c, p) in enumerate(zip(counts, pct)):
        ax.text(c, i, f' {c} ({p:.1f}%)', va='center')
    fig.tight_layout()
    return _save(fig, save_path)


def plot_client_distributions(client_distributions_df, class_names, save_path, title='', figsize=(8, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    df = client_distributions_df.copy()
    df.columns = class_names[:len(df.columns)]
    df.plot(kind='bar', stacked=True, ax=ax, colormap='tab10')
    ax.set_xlabel('Client')
    ax.set_ylabel('Samples')
    ax.set_title(title or 'Client Class Distributions')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_client_heatmap(client_distributions_df, class_names, save_path, title='', figsize=(8, 4)):
    counts = client_distributions_df.copy()
    props = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)
    fig, ax = plt.subplots(figsize=figsize)
    if sns is not None:
        sns.heatmap(props, annot=counts.astype(int), fmt='d', cmap='viridis', ax=ax,
                    xticklabels=class_names[:counts.shape[1]])
    else:
        im = ax.imshow(props.values, aspect='auto', vmin=0, vmax=1, cmap='viridis')
        ax.set_xticks(np.arange(counts.shape[1]))
        ax.set_xticklabels(class_names[:counts.shape[1]], rotation=45, ha='right')
        ax.set_yticks(np.arange(counts.shape[0]))
        ax.set_yticklabels(counts.index)
        fig.colorbar(im, ax=ax)
    ax.set_xlabel('Class')
    ax.set_ylabel('Client')
    ax.set_title(title or 'Client Class Proportions')
    fig.tight_layout()
    return _save(fig, save_path)


def plot_partition_comparison(distributions_dict: Dict[str, object], class_names, save_path, figsize=(12, 7)):
    n = len(distributions_dict)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=figsize, constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for ax, (name, df) in zip(axes, distributions_dict.items()):
        props = df.div(df.sum(axis=1).replace(0, np.nan), axis=0)
        if sns is not None:
            sns.heatmap(props, cmap='viridis', cbar=False, ax=ax,
                        xticklabels=class_names[:df.shape[1]], yticklabels=True)
        else:
            ax.imshow(props.values, aspect='auto', vmin=0, vmax=1, cmap='viridis')
            ax.set_xticks(np.arange(df.shape[1]))
            ax.set_xticklabels(class_names[:df.shape[1]], rotation=45, ha='right')
            ax.set_yticks(np.arange(df.shape[0]))
        ax.set_title(name)
        ax.tick_params(axis='x', rotation=45)
    for ax in axes[n:]:
        ax.axis('off')
    return _save(fig, save_path)


def plot_sample_images(dataset, class_names, num_per_class=5, save_path=None, figsize=(10, 8)):
    labels = get_labels(dataset)
    fig, axes = plt.subplots(len(class_names), num_per_class, figsize=figsize, constrained_layout=True)
    for c, name in enumerate(class_names):
        idxs = np.where(labels == c)[0][:num_per_class]
        for j in range(num_per_class):
            ax = axes[c, j]
            ax.axis('off')
            if j < len(idxs):
                img, _ = dataset[int(idxs[j])]
                arr = img.detach().cpu().numpy()
                if arr.shape[0] in [1, 3]:
                    arr = np.transpose(arr, (1, 2, 0))
                arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-12)
                ax.imshow(arr.squeeze())
            if j == 0:
                ax.set_ylabel(f'{name}\n(N={(labels == c).sum()})', fontsize=8)
    return _save(fig, save_path)
