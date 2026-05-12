from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset, random_split
import flwr as fl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from client.flower_client import create_client_fn, set_state_dict_from_numpy
from data.download import load_mnist
from data.partition import make_partition, get_all_client_distributions
from data.visualise import plot_client_distributions
from metrics.evaluation import evaluate_model, plot_training_curves
from models.simple_cnn import SimpleCNN
from server.flower_server import SaveModelFedAvg, _history_to_dataframe
from utils.seed import set_all_seeds


def _loaders(dataset, parts, batch_size, seed):
    train_loaders, val_loaders = [], []
    generator = torch.Generator().manual_seed(seed)
    for idxs in parts:
        subset = Subset(dataset, idxs)
        n_val = max(1, int(0.1 * len(subset)))
        train_subset, val_subset = random_split(subset, [len(subset) - n_val, n_val], generator=generator)
        train_loaders.append(DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2))
        val_loaders.append(DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2))
    return train_loaders, val_loaders


def run_one(name, train_ds, test_ds, cfg, device, seed):
    out = Path('results/fl_mnist') / name
    out.mkdir(parents=True, exist_ok=True)
    parts = make_partition(train_ds, cfg['partition'], cfg['clients'], seed=seed, alpha=cfg.get('alpha', 0.5))
    dist = get_all_client_distributions(train_ds, parts)
    dist.to_csv(out / 'client_distribution.csv')
    plot_client_distributions(dist, [str(i) for i in range(10)], out / 'client_distribution.png', title=name)
    train_loaders, val_loaders = _loaders(train_ds, parts, 64, seed)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=2)

    def model_fn():
        return SimpleCNN(in_channels=1, num_classes=10)

    strategy = SaveModelFedAvg()

    def eval_fn(server_round, parameters_ndarrays, config):
        model = model_fn().to(device)
        set_state_dict_from_numpy(model, parameters_ndarrays)
        metrics = evaluate_model(model, test_loader, device, 10)
        strategy.update_best(metrics['balanced_accuracy'], parameters_ndarrays)
        return metrics['loss'], {'accuracy': metrics['accuracy'], 'balanced_accuracy': metrics['balanced_accuracy']}

    strategy.evaluate_fn = eval_fn
    client_fn = create_client_fn(model_fn, train_loaders, val_loaders, device, 10, {
        'local_epochs': cfg['local_epochs'],
        'lr': 0.01,
        'momentum': 0.9,
        'loss_fn': 'ce',
        'proximal_mu': cfg['proximal_mu'],
    })
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg['clients'],
        config=fl.server.ServerConfig(num_rounds=cfg['rounds']),
        strategy=strategy,
        client_resources={'num_cpus': 1, 'num_gpus': 0.1 if device.type == 'cuda' else 0.0},
    )
    history_df = _history_to_dataframe(history)
    history_df.to_csv(out / 'metrics_history.csv', index=False)
    plot_training_curves(history_df, out / 'accuracy_loss_curve.png', title=name)
    return history_df.assign(experiment=name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    set_all_seeds(args.seed)
    device = torch.device(args.device if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    train_ds, test_ds = load_mnist()
    clients, rounds = (2, 3) if args.debug else (5, 30)
    configs = [
        ('fedavg_iid', {'partition': 'iid', 'clients': clients, 'rounds': rounds, 'local_epochs': 5, 'proximal_mu': 0.0}),
        ('fedavg_dir05', {'partition': 'dirichlet', 'alpha': 0.5, 'clients': clients, 'rounds': rounds, 'local_epochs': 5, 'proximal_mu': 0.0}),
        ('fedprox_dir05', {'partition': 'dirichlet', 'alpha': 0.5, 'clients': clients, 'rounds': rounds, 'local_epochs': 5, 'proximal_mu': 0.01}),
    ]
    histories = [run_one(name, train_ds, test_ds, cfg, device, args.seed) for name, cfg in configs]
    combined = pd.concat(histories, ignore_index=True)
    combined.to_csv('results/fl_mnist/comparison_history.csv', index=False)
    print(combined.groupby('experiment').tail(1)[['experiment', 'accuracy', 'balanced_accuracy']].to_string(index=False))


if __name__ == '__main__':
    main()
