"""Patch 5 — stragglers grid plot (paper Figure 1 replication).

3 columns: stragglers_fraction = 0%, 50%, 90%
2 rows: training loss, test macro-F1
3 lines per panel: FedAvg (drops stragglers), FedProx μ=0 (keeps partial),
                   FedProx μ=0.1 (keeps partial + proximal term)
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


COLOR_FA   = '#3182ce'
COLOR_FP0  = '#9b2c2c'
COLOR_FP   = '#dd6b20'

ALGO_INFO = [
    ('fedavg',   'FedAvg (drops stragglers)',           COLOR_FA),
    ('fedprox0', 'FedProx μ=0 (keeps partial work)',    COLOR_FP0),
    ('fedprox',  'FedProx μ=0.1 (partial + proximal)',  COLOR_FP),
]
FRACS = [('00', '0% stragglers'), ('05', '50% stragglers'), ('09', '90% stragglers')]


def load_history(stem_prefix: str, results_dir: str) -> pd.DataFrame | None:
    rows = []
    for f in glob.glob(f'{results_dir}/{stem_prefix}_*_s*/metrics_history.csv'):
        df = pd.read_csv(f)
        df['seed'] = int(f.split('_s')[-1].split('/')[0])
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else None


def plot_curve(ax, metric: str, ylabel: str, frac: str, results_dir: str):
    for algo, label, color in ALGO_INFO:
        df = load_history(f'{algo}_str_{frac}', results_dir)
        if df is None or metric not in df.columns:
            continue
        agg = df.groupby('round')[metric].agg(['mean', 'std', 'count']).reset_index()
        agg['sem'] = agg['std'] / np.sqrt(agg['count'].clip(lower=1))
        ax.plot(agg['round'], agg['mean'], color=color, linewidth=1.8, label=label)
        ax.fill_between(agg['round'], agg['mean'] - agg['sem'], agg['mean'] + agg['sem'],
                        color=color, alpha=0.15)
    ax.set_xlabel('Communication round')
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', default='results/thesis')
    ap.add_argument('--out', default='results/thesis/figures/stragglers_grid.png')
    args = ap.parse_args()

    plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10})
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)

    for col, (frac, frac_label) in enumerate(FRACS):
        plot_curve(axes[0, col], 'train_loss_weighted', 'Training loss', frac, args.results_dir)
        axes[0, col].set_title(f'{frac_label}')
        plot_curve(axes[1, col], 'macro_f1', 'Test macro-F1', frac, args.results_dir)

    axes[0, 0].legend(loc='upper right', fontsize=7)
    axes[1, 0].set_title('')
    fig.suptitle('Stragglers experiment (Patch 5 / FedProx paper Figure 1)\n'
                 'Two FedProx benefits: incorporating partial work + proximal term',
                 fontsize=12, fontweight='bold', y=1.04)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches='tight')
    fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
