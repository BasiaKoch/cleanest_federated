"""Compact CPU comparison: FedAvg vs FedProx in the most FedProx-favorable setting.

Bypasses Flower/Ray entirely (direct PyTorch loop) so the whole experiment
runs in ~5-10 minutes on a laptop CPU.

Configuration is based on Li et al. 2020's strongest claim:
  - Mixed-type partition with extreme heterogeneity (specialist_dominance=0.9)
  - Partial participation (5/10 clients per round)
  - Stragglers (50% get random reduced E per round) — the paper's headline
  - High local epochs (E_max=5) for non-stragglers
  - μ=0.1 — the paper's typical winning value

Compares 4 algorithms:
  1. FedAvg-drop      : drops stragglers (paper's FedAvg behavior)
  2. FedAvg-include   : keeps stragglers' partial work (no proximal term)
  3. FedProx (μ=0.1)  : keeps stragglers' partial work + proximal term
  4. FedProx (μ=1.0)  : same with stronger proximal pull

Usage:
  python experiments/cpu_quick_fedprox.py [--seeds 3] [--rounds 30] [--debug-subset 2000]
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from scipy import stats

# Re-use the existing infrastructure
from data.download import load_dermamnist
from data.partition import mixed_type_partition
from models.simple_cnn import SimpleCNN
from metrics.evaluation import evaluate_model


# ----------------------------- Helpers ------------------------------------

def set_all_seeds(seed: int) -> None:
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def state_dict_to_vec(sd) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in sd.items()}


def weighted_average_state_dicts(state_dicts, weights):
    """Weighted average of a list of state_dicts."""
    weights = [w / sum(weights) for w in weights]
    avg = {k: torch.zeros_like(v) for k, v in state_dicts[0].items()}
    for w, sd in zip(weights, state_dicts):
        for k, v in sd.items():
            if v.dtype.is_floating_point:
                avg[k] += w * v
            else:
                # For integer buffers (e.g. BN num_batches_tracked), take the max
                avg[k] = torch.maximum(avg[k], v)
    return avg


def local_train(model, loader, num_epochs, lr, momentum, device,
                global_params=None, mu=0.0):
    """One client's local training; returns avg loss and updated state_dict."""
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    crit = nn.CrossEntropyLoss()
    model.train()
    total_loss, n_batches = 0.0, 0
    for _ in range(num_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device).view(-1).long()
            optimizer.zero_grad()
            out = model(x)
            loss = crit(out, y)
            if global_params is not None and mu > 0:
                prox = 0.0
                for lp, gp in zip(model.parameters(), global_params):
                    prox = prox + ((lp - gp) ** 2).sum()
                loss = loss + (mu / 2.0) * prox
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1
    return total_loss / max(n_batches, 1), model.state_dict()


def federated_train(
    algo: str,                # 'fedavg_drop' | 'fedavg_include' | 'fedprox'
    mu: float,
    train_ds, val_ds, test_ds,
    partitions: list[list[int]],
    num_rounds: int,
    E_max: int,
    fraction_fit: float,
    stragglers_fraction: float,
    lr: float, momentum: float, batch_size: int,
    seed: int, device: str = 'cpu',
):
    set_all_seeds(seed)
    rng = np.random.default_rng(seed)
    num_clients = len(partitions)

    # Per-client loaders
    client_loaders = []
    for cid, idxs in enumerate(partitions):
        ds = Subset(train_ds, idxs)
        # Deterministic loader order per (seed, cid)
        gen = torch.Generator().manual_seed(seed * 10000 + cid)
        client_loaders.append(DataLoader(ds, batch_size=batch_size, shuffle=True, generator=gen))

    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    # Global model — IDENTICAL initialization across paired runs (same seed)
    global_model = SimpleCNN(in_channels=3, num_classes=7).to(device)

    best_val_macro_f1 = -1.0
    best_state = state_dict_to_vec(global_model.state_dict())

    history = []
    n_strag = int(round(stragglers_fraction * num_clients))
    n_sample = max(1, int(round(fraction_fit * num_clients)))

    for rnd in range(1, num_rounds + 1):
        # Sample clients for this round (deterministic by seed)
        sampled = sorted(rng.choice(num_clients, size=n_sample, replace=False).tolist())
        # Designate stragglers among sampled (paired across algorithms via same RNG state)
        # IMPORTANT: re-seed sub-rng for straggler assignment to keep this independent
        round_rng = np.random.default_rng(seed * 100000 + rnd)
        straggler_set = set()
        if n_strag > 0:
            n_strag_this = min(len(sampled), n_strag)
            straggler_set = set(round_rng.choice(sampled, size=n_strag_this, replace=False).tolist())

        # Snapshot global params for proximal term
        global_params = [p.clone().detach() for p in global_model.parameters()]
        global_sd = state_dict_to_vec(global_model.state_dict())

        # Each sampled client trains locally
        local_results = []   # list of (cid, num_samples, E_k, train_loss, state_dict, is_straggler)
        for cid in sampled:
            is_strag = cid in straggler_set
            if is_strag:
                E_k = int(round_rng.integers(1, E_max + 1))
            else:
                E_k = E_max

            # Fresh client model loaded with current global weights
            client_model = SimpleCNN(in_channels=3, num_classes=7).to(device)
            client_model.load_state_dict(global_sd)
            # Per-round, per-client seeding for dropout / batch order
            torch.manual_seed(seed * 100000 + rnd * 100 + cid)

            mu_for_run = mu if algo == 'fedprox' else 0.0
            loss, new_sd = local_train(
                client_model, client_loaders[cid], num_epochs=E_k,
                lr=lr, momentum=momentum, device=device,
                global_params=global_params if mu_for_run > 0 else None,
                mu=mu_for_run,
            )
            local_results.append((cid, len(partitions[cid]), E_k, loss, new_sd, is_strag))

        # Aggregation
        if algo == 'fedavg_drop':
            kept = [r for r in local_results if not r[5]]   # drop stragglers
        else:
            kept = local_results

        if kept:
            weights = [r[1] for r in kept]   # num_samples
            sds = [r[4] for r in kept]
            global_model.load_state_dict(weighted_average_state_dicts(sds, weights))

        # Weighted training loss
        if kept:
            train_loss = sum(r[1] * r[3] for r in kept) / sum(r[1] for r in kept)
        else:
            train_loss = float('nan')

        # Validation
        val_metrics = evaluate_model(global_model, val_loader, device, num_classes=7)
        val_macro = val_metrics['macro_f1']
        if val_macro > best_val_macro_f1:
            best_val_macro_f1 = val_macro
            best_state = state_dict_to_vec(global_model.state_dict())

        history.append({
            'round': rnd, 'train_loss': train_loss,
            'val_macro_f1': val_macro,
            'val_balanced_accuracy': val_metrics['balanced_accuracy'],
            'val_loss': val_metrics['loss'],
            'n_sampled': len(sampled), 'n_stragglers': len(straggler_set),
            'n_kept': len(kept),
        })

    # Final test at best-val checkpoint
    test_model = SimpleCNN(in_channels=3, num_classes=7).to(device)
    test_model.load_state_dict(best_state)
    test_metrics = evaluate_model(test_model, test_loader, device, num_classes=7)

    return {
        'algo': algo, 'mu': mu, 'seed': seed,
        'best_val_macro_f1': best_val_macro_f1,
        'test_macro_f1': test_metrics['macro_f1'],
        'test_balanced_accuracy': test_metrics['balanced_accuracy'],
        'test_worst_class_f1': test_metrics['worst_class_f1'],
        'test_per_class_f1': test_metrics['per_class_f1'],
        'history': history,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    ap.add_argument('--rounds', type=int, default=30)
    ap.add_argument('--E-max', type=int, default=5)
    ap.add_argument('--fraction-fit', type=float, default=0.5)
    ap.add_argument('--stragglers', type=float, default=0.5)
    ap.add_argument('--debug-subset', type=int, default=2000,
                    help='Use only this many training samples (None = full)')
    ap.add_argument('--lr', type=float, default=0.02)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--specialist-dominance', type=float, default=0.9)
    ap.add_argument('--min-samples-per-client', type=int, default=4)
    ap.add_argument('--npz-path', default='../dermamnist_64.npz')
    ap.add_argument('--out', default='results/cpu_quick/cpu_quick_results.json')
    args = ap.parse_args()

    print('='*70)
    print('CPU QUICK FedAvg vs FedProx comparison')
    print('='*70)
    print(f'Seeds:                {args.seeds}')
    print(f'Rounds:               {args.rounds}')
    print(f'E_max:                {args.E_max}')
    print(f'Fraction-fit:         {args.fraction_fit}')
    print(f'Stragglers fraction:  {args.stragglers}')
    print(f'Specialist dominance: {args.specialist_dominance}')
    print(f'Debug subset:         {args.debug_subset}')
    print()

    # Load data once
    print('Loading DermaMNIST...')
    train_ds, val_ds, test_ds = load_dermamnist(
        size=64, source='npz', npz_path=args.npz_path,
    )
    if args.debug_subset:
        # Take first N indices (deterministic). Mixed-type partition will operate on this subset.
        train_ds = Subset(train_ds, list(range(min(args.debug_subset, len(train_ds)))))
    print(f'  train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}')
    print()

    # Algorithms to compare
    ALGOS = [
        ('fedavg_drop',    0.0,  'FedAvg (drops stragglers)'),
        ('fedavg_include', 0.0,  'FedAvg-incl. (keeps partial work, no proximal)'),
        ('fedprox',        0.1,  'FedProx μ=0.1'),
        ('fedprox',        1.0,  'FedProx μ=1.0'),
    ]

    all_results = []
    start_total = time.time()
    for seed in args.seeds:
        # Build mixed-type partition for THIS seed (paired across algorithms)
        partitions = mixed_type_partition(
            train_ds, num_clients=10, seed=seed,
            specialist_dominance=args.specialist_dominance,
            min_samples_per_client=args.min_samples_per_client,
        )
        print(f'--- seed={seed} ---')
        print(f'  partition sizes: {[len(p) for p in partitions]}')

        for algo, mu, label in ALGOS:
            t0 = time.time()
            r = federated_train(
                algo=algo, mu=mu,
                train_ds=train_ds, val_ds=val_ds, test_ds=test_ds,
                partitions=partitions,
                num_rounds=args.rounds, E_max=args.E_max,
                fraction_fit=args.fraction_fit,
                stragglers_fraction=args.stragglers,
                lr=args.lr, momentum=0.9, batch_size=args.batch_size,
                seed=seed,
            )
            elapsed = time.time() - t0
            r['label'] = label
            r['elapsed_seconds'] = elapsed
            all_results.append(r)
            print(f'  {label:50s}  test_macro_F1={r["test_macro_f1"]:.4f}  bACC={r["test_balanced_accuracy"]:.4f}  ({elapsed:.0f}s)')
        print()

    print(f'Total time: {time.time() - start_total:.0f}s\n')

    # Save raw results
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        # strip per-round history to keep file small
        for r in all_results:
            r.pop('history', None)
        json.dump(all_results, f, indent=2)
    print(f'Wrote raw results to {out_path}')

    # Paired comparison: each non-FedAvg algorithm vs fedavg_drop, per seed
    print()
    print('='*70)
    print('PAIRED COMPARISON (vs FedAvg-drop)')
    print('='*70)
    fedavg_drop = [r for r in all_results if r['algo'] == 'fedavg_drop']
    fedavg_drop_by_seed = {r['seed']: r for r in fedavg_drop}

    for algo, mu, label in ALGOS[1:]:
        runs = [r for r in all_results if r['algo'] == algo and r['mu'] == mu]
        diffs = []
        fa_vals, fp_vals = [], []
        for r in runs:
            fa_v = fedavg_drop_by_seed[r['seed']]['test_macro_f1']
            fp_v = r['test_macro_f1']
            fa_vals.append(fa_v); fp_vals.append(fp_v)
            diffs.append(fp_v - fa_v)
        diffs = np.array(diffs)
        if len(diffs) >= 2:
            try:
                _, p = stats.wilcoxon(diffs)
            except ValueError:
                p = float('nan')
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        else:
            p, sig = float('nan'), '-'
        winner = '✓ wins' if diffs.mean() > 0 else '✗ loses'
        print(f'  {label:50s}  mean Δ = {diffs.mean():+.4f}  (per-seed: {[f"{d:+.4f}" for d in diffs]})  Wilcoxon p={p:.4f} {sig}  {winner}')

    # Verdict
    print()
    fedprox_01 = [r for r in all_results if r['algo'] == 'fedprox' and r['mu'] == 0.1]
    fa = [fedavg_drop_by_seed[r['seed']]['test_macro_f1'] for r in fedprox_01]
    fp = [r['test_macro_f1'] for r in fedprox_01]
    diff = np.array(fp) - np.array(fa)
    print('VERDICT (FedProx μ=0.1 vs FedAvg-drop):')
    print(f'  FedAvg mean macro-F1 = {np.mean(fa):.4f}')
    print(f'  FedProx mean macro-F1 = {np.mean(fp):.4f}')
    print(f'  Mean improvement     = {diff.mean():+.4f}  ({diff.mean()*100:+.2f} pp)')
    if diff.mean() > 0 and (diff > 0).all():
        print('  → FedProx wins on ALL seeds (consistent improvement)')
    elif diff.mean() > 0:
        print('  → FedProx wins on average but not unanimously across seeds')
    else:
        print('  → FedProx does NOT beat FedAvg in this configuration')


if __name__ == '__main__':
    main()
