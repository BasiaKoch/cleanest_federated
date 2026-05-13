from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import math
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset, random_split
import flwr as fl
from flwr.common import parameters_to_ndarrays

from client.flower_client import create_client_fn, set_state_dict_from_numpy
from data.download import load_dermamnist, get_dataset_info
from data.partition import make_partition, get_all_client_distributions, get_client_class_distribution
from data.visualise import plot_client_distributions, plot_client_heatmap
from data.augmentation import create_weighted_sampler
from losses.weighted_ce import compute_local_class_weights, compute_global_estimated_weights
from metrics.evaluation import evaluate_model, plot_confusion_matrix, plot_per_class_metrics, plot_training_curves
from metrics.per_class_tracker import PerClassTracker
from metrics.drift_analysis import compute_cosine_similarity, compute_l2_distance
from models import get_model
from utils.experiment_tracker import make_jsonable


class SaveModelFedAvg(fl.server.strategy.FedAvg):
    """FedAvg with checkpointing and optional client drift tracking."""

    def __init__(self, *args, track_drift: bool = False,
                 system_het_epochs=None, system_het_seed: int = 42,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.best_parameters_ndarrays = None
        self.best_balanced_accuracy = -1.0
        self.latest_parameters = None
        self.track_drift = track_drift
        self.drift_history = []
        self._previous_global_params = None
        # System heterogeneity: list of candidate local-epoch values to sample
        # from per-client per-round. If None / empty, behaves like vanilla FedAvg
        # (constant local_epochs from the YAML).
        self.system_het_epochs = list(system_het_epochs) if system_het_epochs else None
        self._sys_het_rng = np.random.default_rng(system_het_seed)
        self.epochs_history = []  # list of {round, client_id, local_epochs}

    def configure_fit(self, server_round, parameters, client_manager):
        instructions = super().configure_fit(server_round, parameters, client_manager)
        if not self.system_het_epochs:
            return instructions
        new_instructions = []
        for client, fit_ins in instructions:
            assigned = int(self._sys_het_rng.choice(self.system_het_epochs))
            cfg = dict(fit_ins.config) if fit_ins.config else {}
            cfg['local_epochs'] = assigned
            new_instructions.append((client, fl.common.FitIns(fit_ins.parameters, cfg)))
            self.epochs_history.append({
                'round': int(server_round),
                'client_proxy': getattr(client, 'cid', 'unknown'),
                'local_epochs': assigned,
            })
        return new_instructions

    def aggregate_fit(self, server_round, results, failures):
        if self.track_drift and self._previous_global_params is not None:
            for _, fit_res in results:
                client_params = parameters_to_ndarrays(fit_res.parameters)
                self.drift_history.append({
                    'round': int(server_round),
                    'client_id': str(fit_res.metrics.get('cid', 'unknown')),
                    'cosine_similarity': compute_cosine_similarity(self._previous_global_params, client_params),
                    'l2_distance': compute_l2_distance(self._previous_global_params, client_params),
                    'num_samples': int(fit_res.num_examples),
                })
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None:
            self.latest_parameters = aggregated_parameters
            self._previous_global_params = parameters_to_ndarrays(aggregated_parameters)
        return aggregated_parameters, metrics

    def update_best(self, balanced_accuracy: float, parameters_ndarrays: List[np.ndarray]) -> None:
        if balanced_accuracy > self.best_balanced_accuracy:
            self.best_balanced_accuracy = float(balanced_accuracy)
            self.best_parameters_ndarrays = [np.array(p, copy=True) for p in parameters_ndarrays]


def create_centralised_evaluate_fn(model_fn, val_loader, device, num_classes, class_names, strategy_ref: SaveModelFedAvg):
    """Evaluate global model on validation data after each FL round.

    Important: this uses validation data, not test data. The test set is used only
    once after training is complete.
    """

    def evaluate(server_round, parameters_ndarrays, config):
        model = model_fn().to(device)
        set_state_dict_from_numpy(model, parameters_ndarrays)
        metrics = evaluate_model(model, val_loader, device, num_classes)
        strategy_ref.update_best(metrics['balanced_accuracy'], parameters_ndarrays)
        result_metrics = {
            'accuracy': float(metrics['accuracy']),
            'balanced_accuracy': float(metrics['balanced_accuracy']),
            'macro_f1': float(metrics['macro_f1']),
            'worst_class_f1': float(metrics['worst_class_f1']),
            'worst_class_recall': float(metrics['worst_class_recall']),
        }
        for i, name in enumerate(class_names):
            short = name[:20].replace(' ', '_')
            result_metrics[f'f1_{short}'] = float(metrics['per_class_f1'][i])
            result_metrics[f'recall_{short}'] = float(metrics['per_class_recall'][i])
        return float(metrics['loss']), result_metrics

    return evaluate


def _history_to_dataframe(history) -> pd.DataFrame:
    rows = {}
    for metric_name, values in history.metrics_centralized.items():
        for rnd, val in values:
            rows.setdefault(int(rnd), {'round': int(rnd)})[metric_name] = float(val)
    if hasattr(history, 'losses_centralized'):
        for rnd, val in history.losses_centralized:
            rows.setdefault(int(rnd), {'round': int(rnd)})['loss'] = float(val)
    return pd.DataFrame([rows[k] for k in sorted(rows)])


def _save_model_from_ndarrays(model_fn, params_ndarrays, save_path: Path) -> None:
    model = model_fn()
    set_state_dict_from_numpy(model, params_ndarrays)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)


def _make_loaders_for_partition(dataset, partitions, batch_size, seed, num_classes, use_weighted_sampler=False):
    train_loaders, val_loaders = [], []
    generator = torch.Generator().manual_seed(seed)
    for indices in partitions:
        subset = Subset(dataset, indices)
        n_val = max(1, int(0.1 * len(subset)))
        n_train = len(subset) - n_val
        train_subset, val_subset = random_split(subset, [n_train, n_val], generator=generator)
        if use_weighted_sampler:
            # Map random_split subset positions back to original dataset indices.
            original_train_indices = [indices[i] for i in train_subset.indices]
            sampler = create_weighted_sampler(dataset, original_train_indices, num_classes)
            train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=sampler, num_workers=2)
        else:
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)
        train_loaders.append(train_loader)
        val_loaders.append(val_loader)
    return train_loaders, val_loaders


def evaluate_per_client(model_fn, params_ndarrays, client_test_loaders, device, num_classes, train_partitions, test_partitions, test_dataset):
    rows = []
    for cid, loader in enumerate(client_test_loaders):
        model = model_fn().to(device)
        set_state_dict_from_numpy(model, params_ndarrays)
        metrics = evaluate_model(model, loader, device, num_classes)
        row = {
            'client_id': cid,
            'num_train_samples': len(train_partitions[cid]),
            'num_test_samples': len(test_partitions[cid]),
            'local_class_distribution': json.dumps(get_client_class_distribution(test_dataset, test_partitions[cid])),
            'accuracy': metrics['accuracy'],
            'balanced_accuracy': metrics['balanced_accuracy'],
            'macro_f1': metrics['macro_f1'],
            'worst_class_f1': metrics['worst_class_f1'],
            'worst_class_recall': metrics['worst_class_recall'],
            'per_class_recall_json': json.dumps(metrics['per_class_recall']),
            'per_class_f1_json': json.dumps(metrics['per_class_f1']),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def run_simulation(config: Dict):
    dataset_cfg = config['dataset']
    fed_cfg = config['federation']
    train_cfg = config['training']
    part_cfg = config['partition']
    model_cfg = config['model']
    loss_cfg = config.get('loss', {})
    obj_cfg = config.get('client_objective', {})
    misc_cfg = config.get('misc', {})
    aug_cfg = config.get('augmentation', {})

    seed = int(misc_cfg.get('seed', 42))
    device = torch.device(misc_cfg.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    save_dir = Path(misc_cfg.get('resolved_save_dir', misc_cfg.get('save_dir', 'results')))
    save_dir.mkdir(parents=True, exist_ok=True)

    if dataset_cfg['name'] != 'dermamnist':
        raise NotImplementedError('run_simulation currently supports dermamnist. Use run_fl_mnist.py for MNIST validation.')
    train_ds, val_ds, test_ds = load_dermamnist(
        size=int(dataset_cfg.get('size', 28)),
        source=str(dataset_cfg.get('source', 'package')),
        npz_path=dataset_cfg.get('npz_path', 'datasets/medmnist/dermamnist.npz'),
    )
    if config.get('debug_subset'):
        n = int(config['debug_subset'])
        train_ds = Subset(train_ds, list(range(min(n, len(train_ds)))))
        # Subset lacks labels; for debug, partition functions will fall back to __getitem__.

    info = get_dataset_info('dermamnist')
    class_names = info['class_names']
    num_clients = int(fed_cfg['num_clients'])
    num_classes = int(model_cfg['num_classes'])
    batch_size = int(train_cfg['batch_size'])

    part_kwargs = dict(part_cfg)
    strategy_name = part_kwargs.pop('strategy')
    train_partitions = make_partition(train_ds, strategy_name, num_clients, seed=seed, **part_kwargs)
    # Simulated per-client test partitions from official test split. Test is used only at the end.
    test_partitions = make_partition(test_ds, strategy_name, num_clients, seed=seed + 999, **part_kwargs)

    train_dist = get_all_client_distributions(train_ds, train_partitions)
    (save_dir / 'data_distributions').mkdir(exist_ok=True, parents=True)
    train_dist.to_csv(save_dir / 'data_distributions' / 'client_train_distributions.csv')
    plot_client_distributions(train_dist, class_names, save_dir / 'data_distributions' / 'client_distribution.png', title=strategy_name)
    plot_client_heatmap(train_dist, class_names, save_dir / 'data_distributions' / 'client_heatmap.png', title=strategy_name)

    use_weighted_sampler = bool(aug_cfg.get('use_weighted_sampler', False))
    train_loaders, val_loaders = _make_loaders_for_partition(train_ds, train_partitions, batch_size, seed, num_classes, use_weighted_sampler)
    global_val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    global_test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    client_test_loaders = [DataLoader(Subset(test_ds, idxs), batch_size=batch_size, shuffle=False, num_workers=2) for idxs in test_partitions]

    def model_fn():
        return get_model(
            model_cfg['name'],
            in_channels=int(model_cfg['in_channels']),
            num_classes=num_classes,
            image_size=int(dataset_cfg.get('size', 28)),
            pretrained=bool(model_cfg.get('pretrained', False)),
        )

    client_config = {
        'local_epochs': int(fed_cfg['local_epochs']),
        'lr': float(train_cfg['lr']),
        'momentum': float(train_cfg.get('momentum', 0.9)),
        'weight_decay': float(train_cfg.get('weight_decay', 0.0)),
        'loss_fn': loss_cfg.get('type', 'ce'),
        'focal_gamma': float(loss_cfg.get('focal_gamma', 2.0)),
        'proximal_mu': float(obj_cfg.get('proximal_mu', 0.0)),
    }

    client_class_weights = None
    weight_strategy = loss_cfg.get('weight_strategy', 'none')
    if weight_strategy and weight_strategy != 'none':
        if weight_strategy.startswith('local'):
            method = 'inverse' if 'inverse' in weight_strategy else 'sqrt_inverse'
            client_class_weights = [compute_local_class_weights(train_ds, idxs, num_classes, strategy=method) for idxs in train_partitions]
        elif weight_strategy == 'global_estimated':
            w = compute_global_estimated_weights(train_dist, num_classes, strategy='inverse')
            client_class_weights = [w for _ in range(num_clients)]

    strategy = SaveModelFedAvg(track_drift=bool(config.get('track_drift', False)))
    evaluate_fn = create_centralised_evaluate_fn(model_fn, global_val_loader, device, num_classes, class_names, strategy)
    fraction_fit = float(fed_cfg.get('fraction_fit', 1.0))
    fraction_evaluate = float(fed_cfg.get('fraction_evaluate', 1.0))
    min_fit_clients = max(1, math.ceil(num_clients * fraction_fit))
    min_eval_clients = max(1, math.ceil(num_clients * fraction_evaluate))
    system_het_epochs = fed_cfg.get('system_heterogeneity', None)
    strategy = SaveModelFedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_eval_clients,
        min_available_clients=num_clients,
        evaluate_fn=evaluate_fn,
        track_drift=bool(config.get('track_drift', False)),
        system_het_epochs=system_het_epochs,
        system_het_seed=int(config.get('misc', {}).get('seed', 42)),
    )
    # Rebind evaluate_fn to the actual strategy instance used by Flower.
    strategy.evaluate_fn = create_centralised_evaluate_fn(model_fn, global_val_loader, device, num_classes, class_names, strategy)

    client_resources = {'num_cpus': 1, 'num_gpus': 0.1 if device.type == 'cuda' else 0.0}
    client_fn = create_client_fn(model_fn, train_loaders, val_loaders, device, num_classes, client_config, client_class_weights)
    num_rounds = int(fed_cfg['num_rounds'])
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources=client_resources,
    )

    history_df = _history_to_dataframe(history)
    history_df.to_csv(save_dir / 'metrics_history.csv', index=False)
    if strategy.epochs_history:
        pd.DataFrame(strategy.epochs_history).to_csv(save_dir / 'epochs_history.csv', index=False)
    if not history_df.empty:
        plot_training_curves(history_df, save_dir / 'accuracy_loss_curve.png')
        tracker = PerClassTracker(num_classes, class_names, save_dir)
        for _, row in history_df.iterrows():
            tracker.log_round(int(row['round']), row.to_dict())
        tracker.save()
        tracker.plot_per_class_f1_curves(save_dir / 'per_class_convergence.png')
        tracker.plot_per_class_recall_curves(save_dir / 'per_class_recall_convergence.png')
        tracker.plot_convergence_heatmap(metric='f1', save_path=save_dir / 'per_class_convergence_heatmap.png')

    best_params = strategy.best_parameters_ndarrays
    if best_params is None:
        best_params = parameters_to_ndarrays(strategy.latest_parameters)
    _save_model_from_ndarrays(model_fn, best_params, save_dir / 'best_model.pt')

    final_model = model_fn().to(device)
    set_state_dict_from_numpy(final_model, best_params)
    global_test_metrics = evaluate_model(final_model, global_test_loader, device, num_classes)
    with open(save_dir / 'global_test_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(make_jsonable(global_test_metrics), f, indent=2)
    plot_confusion_matrix(global_test_metrics['confusion_matrix'], class_names, save_dir / 'confusion_matrix.png')
    plot_per_class_metrics(global_test_metrics, class_names, save_dir / 'per_class_f1_bar.png')

    per_client_df = evaluate_per_client(model_fn, best_params, client_test_loaders, device, num_classes, train_partitions, test_partitions, test_ds)
    per_client_df.to_csv(save_dir / 'per_client_metrics.csv', index=False)
    client_summary = {
        'avg_client_balanced_accuracy': float(per_client_df['balanced_accuracy'].mean()),
        'worst_client_balanced_accuracy': float(per_client_df['balanced_accuracy'].min()),
        'best_worst_client_gap': float(per_client_df['balanced_accuracy'].max() - per_client_df['balanced_accuracy'].min()),
        'client_balanced_accuracy_std': float(per_client_df['balanced_accuracy'].std()),
    }
    with open(save_dir / 'client_summary.json', 'w', encoding='utf-8') as f:
        json.dump(client_summary, f, indent=2)
    if strategy.track_drift:
        pd.DataFrame(strategy.drift_history).to_csv(save_dir / 'drift_history.csv', index=False)
    return history, global_test_metrics, per_client_df
