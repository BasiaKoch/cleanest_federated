"""Patch 2: per-(seed, E) best-μ selection on global val macro-F1.

For each (seed, E), look across {fedprox_mu_0001, fedprox_mu_001, fedprox_mu01,
fedprox_mu_1} and pick the run with highest val-set macro-F1 at its best
checkpoint. That value becomes the FedProx number for the headline.

Outputs a CSV with one row per (seed, E):
  seed, E, best_mu, fedprox_val_macro_f1, fedprox_test_macro_f1, fedavg_test_macro_f1
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd


MU_CANDIDATES = {
    'fedprox_mu_0001': 0.001,
    'fedprox_mu_001':  0.01,
    'fedprox_mu01':    0.1,
    'fedprox_mu_1':    1.0,
}


def best_val_macro_f1(run_dir: Path) -> float:
    """Best macro_f1 across all rounds in metrics_history.csv (= best val)."""
    f = run_dir / 'metrics_history.csv'
    if not f.exists():
        return float('-inf')
    df = pd.read_csv(f)
    if 'macro_f1' not in df.columns or df.empty:
        return float('-inf')
    return float(df['macro_f1'].max())


def test_macro_f1(run_dir: Path) -> float:
    f = run_dir / 'global_test_metrics.json'
    if not f.exists():
        return float('nan')
    with open(f) as fp:
        return float(json.load(fp)['macro_f1'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', default='results/thesis')
    ap.add_argument('--out', default='results/thesis/analysis/best_mu_per_seed.csv')
    args = ap.parse_args()

    # Discover (seed, E) combinations from FedAvg runs
    rows = []
    for fa_dir in sorted(glob.glob(f'{args.results_dir}/fedavg*_E*_s*')):
        fa_path = Path(fa_dir)
        # Parse "fedavg_E{E}_s{seed}" or "fedavg_E5_s42" or "fedavg_E20_s42"
        stem = fa_path.name
        try:
            parts = stem.split('_')
            seed = int(parts[-1][1:])
            E = int(parts[-2][1:]) if parts[-2].startswith('E') else 20
        except (ValueError, IndexError):
            continue

        fa_test = test_macro_f1(fa_path)
        fa_val = best_val_macro_f1(fa_path)

        # Find matching FedProx runs across μ candidates
        best_mu = None
        best_val = float('-inf')
        best_test = float('nan')
        mu_results = {}
        for stem_prefix, mu in MU_CANDIDATES.items():
            # Try multiple naming conventions
            for candidate in [
                Path(args.results_dir) / f'{stem_prefix}_E{E}_s{seed}',
                Path(args.results_dir) / f'{stem_prefix.replace("fedprox_", "fedprox_")}_E{E}_s{seed}',
            ]:
                if candidate.exists():
                    val = best_val_macro_f1(candidate)
                    test = test_macro_f1(candidate)
                    mu_results[mu] = {'val': val, 'test': test}
                    if val > best_val:
                        best_val = val
                        best_mu = mu
                        best_test = test
                    break

        rows.append({
            'seed': seed,
            'E': E,
            'fedavg_val_macro_f1': fa_val,
            'fedavg_test_macro_f1': fa_test,
            'fedprox_best_mu': best_mu,
            'fedprox_best_val_macro_f1': best_val,
            'fedprox_best_test_macro_f1': best_test,
            'mu_results': json.dumps(mu_results),
        })

    if not rows:
        print('No paired runs found.')
        return

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f'wrote {args.out}')
    print()
    print('=== Best-μ selection per (seed, E) ===')
    print(df[['seed', 'E', 'fedavg_test_macro_f1', 'fedprox_best_mu', 'fedprox_best_test_macro_f1']].to_string(index=False))
    print()
    print('=== Aggregate ===')
    for E in sorted(df['E'].unique()):
        sub = df[df['E'] == E]
        fa_mean = sub['fedavg_test_macro_f1'].mean()
        fp_mean = sub['fedprox_best_test_macro_f1'].mean()
        mu_counts = sub['fedprox_best_mu'].value_counts().to_dict()
        print(f'  E={E}: n={len(sub)}  FedAvg mean={fa_mean:.3f}  FedProx(best-μ) mean={fp_mean:.3f}  Δ={fp_mean-fa_mean:+.3f}  μ chosen: {mu_counts}')


if __name__ == '__main__':
    main()
