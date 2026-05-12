from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.download import load_dermamnist, get_dataset_info, get_class_distribution
from data.partition import iid_partition, dirichlet_partition, pathological_partition, get_all_client_distributions, compute_distribution_entropy
from data.visualise import plot_global_class_distribution, plot_sample_images, plot_client_distributions, plot_client_heatmap, plot_partition_comparison


def main():
    out = Path('results/exploration')
    out.mkdir(parents=True, exist_ok=True)
    train, val, test = load_dermamnist(28)
    info = get_dataset_info('dermamnist')
    names = info['class_names']
    print(f'DermaMNIST sizes: train={len(train)}, val={len(val)}, test={len(test)}')
    dist = get_class_distribution(train)
    counts = np.array([dist.get(i, 0) for i in range(len(names))])
    for i, name in enumerate(names):
        print(f'{i}: {name}: {counts[i]} ({counts[i] / counts.sum() * 100:.2f}%)')
    print(f'Imbalance ratio: {counts.max() / max(counts.min(), 1):.2f}')
    melanoma_idx = [i for i, n in enumerate(names) if 'melanoma' in n.lower()][0]
    print(f'Melanoma: {counts[melanoma_idx]} ({counts[melanoma_idx] / counts.sum() * 100:.2f}%)')
    plot_global_class_distribution(train, names, out / 'global_class_distribution.png')
    plot_sample_images(train, names, 8, out / 'dermamnist_samples.png')

    parts = {
        'IID': iid_partition(train, 10, seed=42),
        'Dir(0.1)': dirichlet_partition(train, 10, 0.1, seed=42),
        'Dir(0.3)': dirichlet_partition(train, 10, 0.3, seed=42),
        'Dir(0.5)': dirichlet_partition(train, 10, 0.5, seed=42),
        'Dir(1.0)': dirichlet_partition(train, 10, 1.0, seed=42),
        'Pathological(2)': pathological_partition(train, 10, 2, seed=42),
    }
    dfs = {}
    for label, p in parts.items():
        df = get_all_client_distributions(train, p)
        dfs[label] = df
        safe = label.replace('(', '').replace(')', '').replace('.', '').replace(' ', '_')
        plot_client_distributions(df, names, out / f'{safe}_stacked.png', title=label)
        plot_client_heatmap(df, names, out / f'{safe}_heatmap.png', title=label)
        print(f'\n{label}')
        print(f'Clients with zero Dermatofibroma: {(df[3] == 0).sum()}')
        print(f'Clients with zero Vascular Lesions: {(df[6] == 0).sum()}')
        print(f'Min/max client sizes: {df.sum(axis=1).min()} / {df.sum(axis=1).max()}')
        ents = [compute_distribution_entropy(row.values) for _, row in df.iterrows()]
        print(f'Mean entropy: {np.mean(ents):.3f}')
    plot_partition_comparison(dfs, names, out / 'partition_comparison.png')


if __name__ == '__main__':
    main()
