"""Dedicated FedAvg vs FedProx comparison: convergence curves + final-round bars."""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FIG = Path('results/figures')
FIG.mkdir(parents=True, exist_ok=True)

REGIMES = [
    ('iid',          'IID (control)'),
    ('dir05',        'Dirichlet α=0.5'),
    ('dir01',        'Dirichlet α=0.1'),
    ('pathological', 'Pathological k=2'),
]

FEDAVG_COLOR = '#3182ce'
FEDPROX_COLOR = '#dd6b20'


def load_history(stem: str) -> pd.DataFrame | None:
    """Load round-by-round history for all seeds of `stem`, stacked."""
    rows = []
    for f in glob.glob(f'results/{stem}_E5_s*/metrics_history.csv'):
        df = pd.read_csv(f)
        seed = int(f.split('_s')[-1].split('/')[0])
        df['seed'] = seed
        rows.append(df)
    if not rows:
        return None
    return pd.concat(rows, ignore_index=True)


def plot_curves(ax, regime: str, regime_label: str, metric: str, ylabel: str) -> None:
    """One subplot: FedAvg vs FedProx curves with shaded ±std across seeds."""
    for stem_prefix, color, label in [
        (f'fedavg_{regime}', FEDAVG_COLOR, 'FedAvg'),
        (f'fedprox_{regime}', FEDPROX_COLOR, 'FedProx (μ=0.01)'),
    ]:
        df = load_history(stem_prefix)
        if df is None:
            continue
        grouped = df.groupby('round')[metric].agg(['mean', 'std']).reset_index()
        ax.plot(grouped['round'], grouped['mean'], color=color, linewidth=2, label=label)
        ax.fill_between(grouped['round'],
                        grouped['mean'] - grouped['std'],
                        grouped['mean'] + grouped['std'],
                        color=color, alpha=0.18)

    ax.set_xlabel('Round')
    ax.set_ylabel(ylabel)
    ax.set_title(regime_label, fontsize=11)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 100)


def final_round_bars(ax, metric: str, ylabel: str, ylim: tuple) -> None:
    """Side-by-side bars: FedAvg vs FedProx final-round metric, all regimes."""
    fa_means, fa_stds, fp_means, fp_stds = [], [], [], []
    for regime, _ in REGIMES:
        fa = load_history(f'fedavg_{regime}')
        fp = load_history(f'fedprox_{regime}')
        if fa is not None:
            last = fa[fa['round'] == fa['round'].max()]
            fa_means.append(last[metric].mean())
            fa_stds.append(last[metric].std())
        else:
            fa_means.append(np.nan); fa_stds.append(0)
        if fp is not None:
            last = fp[fp['round'] == fp['round'].max()]
            fp_means.append(last[metric].mean())
            fp_stds.append(last[metric].std())
        else:
            fp_means.append(np.nan); fp_stds.append(0)

    x = np.arange(len(REGIMES))
    w = 0.36
    b1 = ax.bar(x - w/2, fa_means, w, yerr=fa_stds, capsize=4,
                color=FEDAVG_COLOR, edgecolor='white', label='FedAvg')
    b2 = ax.bar(x + w/2, fp_means, w, yerr=fp_stds, capsize=4,
                color=FEDPROX_COLOR, edgecolor='white', label='FedProx')

    for bars, vals in [(b1, fa_means), (b2, fp_means)]:
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.annotate(f'{v:.2f}', xy=(bar.get_x() + bar.get_width()/2, v),
                            xytext=(0, 3), textcoords='offset points',
                            ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in REGIMES], fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3, axis='y')


def main() -> None:
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
    })

    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.1])

    # Top row: bACC convergence per regime
    for i, (regime, label) in enumerate(REGIMES):
        ax = fig.add_subplot(gs[0, i])
        plot_curves(ax, regime, label, 'balanced_accuracy', 'Balanced acc.')
        if i == 0:
            ax.text(-0.18, 1.05, 'Balanced accuracy', transform=ax.transAxes,
                    fontsize=12, fontweight='bold', va='bottom')

    # Middle row: worst-class F1 convergence per regime
    for i, (regime, label) in enumerate(REGIMES):
        ax = fig.add_subplot(gs[1, i])
        plot_curves(ax, regime, label, 'worst_class_f1', 'Worst-class F1')
        if i == 0:
            ax.text(-0.18, 1.05, 'Worst-class F1', transform=ax.transAxes,
                    fontsize=12, fontweight='bold', va='bottom')

    # Bottom row: final-round summary bars (two wide subplots)
    ax_bar1 = fig.add_subplot(gs[2, :2])
    ax_bar2 = fig.add_subplot(gs[2, 2:])
    final_round_bars(ax_bar1, 'balanced_accuracy', 'Final balanced acc.', (0, 0.65))
    ax_bar1.set_title('Final-round balanced accuracy', fontsize=11, fontweight='bold')
    final_round_bars(ax_bar2, 'worst_class_f1', 'Final worst-class F1', (0, 0.35))
    ax_bar2.set_title('Final-round worst-class F1', fontsize=11, fontweight='bold')

    fig.suptitle('FedAvg vs FedProx (μ=0.01) — convergence and final-round comparison',
                 fontsize=14, fontweight='bold', y=1.01)

    out_pdf = FIG / 'fig_fedavg_vs_fedprox.pdf'
    out_png = FIG / 'fig_fedavg_vs_fedprox.png'
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out_pdf}')
    print(f'wrote {out_png}')


if __name__ == '__main__':
    main()
