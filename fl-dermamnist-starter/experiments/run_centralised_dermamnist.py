from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.download import load_dermamnist, get_dataset_info, get_class_distribution
from losses.weighted_ce import compute_class_weights
from models.simple_cnn import SimpleCNN
from trainers.centralised import CentralisedTrainer
from metrics.evaluation import plot_confusion_matrix, plot_per_class_metrics, plot_training_curves
from utils.seed import set_all_seeds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    set_all_seeds(args.seed)
    device = torch.device(args.device if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    train_ds, val_ds, test_ds = load_dermamnist(size=28)
    info = get_dataset_info('dermamnist')
    class_names = info['class_names']
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    class_counts = get_class_distribution(train_ds)
    class_weights = compute_class_weights([class_counts.get(i, 0) for i in range(info['num_classes'])], strategy='inverse')
    runs = [
        ('ce', 'ce', None),
        ('weighted_ce', 'weighted_ce', class_weights),
        ('focal', 'focal', class_weights),
    ]
    rows = []
    histories = {}
    comparison_dir = Path('results/centralised/dermamnist/comparison')
    comparison_dir.mkdir(parents=True, exist_ok=True)
    for name, loss_fn, weights in runs:
        out_dir = Path('results/centralised/dermamnist') / name
        model = SimpleCNN(in_channels=3, num_classes=info['num_classes'])
        trainer = CentralisedTrainer(model, train_loader, val_loader, test_loader, device, info['num_classes'],
                                     lr=args.lr, loss_fn=loss_fn, class_weights=weights, focal_gamma=2.0)
        result = trainer.train(args.epochs, out_dir, class_names=class_names)
        metrics = result['test_metrics']
        histories[name] = result['history']
        plot_confusion_matrix(metrics['confusion_matrix'], class_names, out_dir / 'confusion_matrix.png')
        plot_per_class_metrics(metrics, class_names, out_dir / 'per_class_metrics.png')
        rows.append({
            'loss_type': name,
            'accuracy': metrics['accuracy'],
            'balanced_accuracy': metrics['balanced_accuracy'],
            'macro_f1': metrics['macro_f1'],
            'worst_f1': metrics['worst_class_f1'],
            'melanoma_recall': metrics['per_class_recall'][4],
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(comparison_dir / 'summary.csv', index=False)
    for name, history in histories.items():
        plot_training_curves(history, comparison_dir / f'{name}_training_curve.png', title=name)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
