from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config_loader import load_config, config_to_experiment_name
from utils.seed import set_all_seeds
from server.flower_server import run_simulation
from data.download import load_dermamnist
from data.partition import make_partition, get_all_client_distributions


def dry_run_summary(cfg):
    dataset_cfg = cfg['dataset']
    fed_cfg = cfg['federation']
    part_cfg = dict(cfg['partition'])
    seed = int(cfg.get('misc', {}).get('seed', 42))
    train_ds, val_ds, test_ds = load_dermamnist(
        size=int(dataset_cfg.get('size', 28)),
        source=str(dataset_cfg.get('source', 'package')),
        npz_path=dataset_cfg.get('npz_path', 'datasets/medmnist/dermamnist.npz'),
    )
    if cfg.get('debug_subset'):
        from torch.utils.data import Subset
        n = int(cfg['debug_subset'])
        train_ds = Subset(train_ds, list(range(min(n, len(train_ds)))))
    strategy = part_cfg.pop('strategy')
    parts = make_partition(train_ds, strategy, int(fed_cfg['num_clients']), seed=seed, **part_cfg)
    dist = get_all_client_distributions(train_ds, parts)
    print(f'Dataset sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}')
    print(f'Partition: {strategy}; clients={len(parts)}; min/max client sizes={dist.sum(axis=1).min()} / {dist.sum(axis=1).max()}')
    print(dist)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--debug_subset', type=int, default=None)
    parser.add_argument('--num_rounds_override', type=int, default=None)
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--track_drift', action='store_true')
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.setdefault('misc', {})['seed'] = args.seed
    if args.device is not None:
        cfg.setdefault('misc', {})['device'] = args.device
    if args.num_rounds_override is not None:
        cfg.setdefault('federation', {})['num_rounds'] = args.num_rounds_override
    if args.debug_subset is not None:
        cfg['debug_subset'] = args.debug_subset
    if args.track_drift:
        cfg['track_drift'] = True

    seed = int(cfg.get('misc', {}).get('seed', 42))
    set_all_seeds(seed)
    exp_name = config_to_experiment_name(cfg)
    base_save = Path(cfg.get('misc', {}).get('save_dir', 'results'))
    resolved_save = base_save / exp_name
    cfg.setdefault('misc', {})['resolved_save_dir'] = str(resolved_save)
    resolved_save.mkdir(parents=True, exist_ok=True)
    with open(resolved_save / 'config.yaml', 'w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print('=== Experiment config ===')
    print(yaml.safe_dump(cfg, sort_keys=False))
    if args.dry_run:
        dry_run_summary(cfg)
        print('Dry run complete. No training started.')
        return

    start = time.time()
    history, global_metrics, per_client_df = run_simulation(cfg)
    elapsed = (time.time() - start) / 60.0
    melanoma_recall = global_metrics.get('per_class_recall', [None] * 7)[4]
    print('\n=== Completed ===')
    print(f'Experiment: {exp_name}')
    print(f"Global: bACC={global_metrics['balanced_accuracy']:.4f} | Macro F1={global_metrics['macro_f1']:.4f} | Worst F1={global_metrics['worst_class_f1']:.4f} | Melanoma Recall={melanoma_recall:.4f}")
    print(f"Clients: Avg bACC={per_client_df['balanced_accuracy'].mean():.4f} | Worst bACC={per_client_df['balanced_accuracy'].min():.4f} | Gap={(per_client_df['balanced_accuracy'].max() - per_client_df['balanced_accuracy'].min()):.4f}")
    print(f'Completed in {elapsed:.2f} minutes')


if __name__ == '__main__':
    main()
