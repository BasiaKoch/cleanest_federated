"""System-heterogeneity comparison: FedAvg vs FedProx when each client
runs a random number of local epochs per round.

Inspired by FedProx paper Fig. 4 and the nedeljkovicmajaa repo's system_het/
mode. This is where FedProx is most likely to show a measurable advantage.

Four panels:
  (a) Balanced accuracy over rounds — convergence
  (b) Round-to-round stability (rolling std) — FedProx's headline benefit
  (c) Worst-class F1 over rounds
  (d) Final-round bars with paired t-tests
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

FIG = Path('results/figures')
FIG.mkdir(parents=True, exist_ok=True)

FEDAVG_COLOR = '#3182ce'
FEDPROX_COLOR = '#dd6b20'

FEDAVG_STEM = 'fedavg_sysheterogeneity'
FEDPROX_STEM = 'fedprox_sysheterogeneity'


def load_history(stem: str) -> pd.DataFrame | None:
    rows = []
    for f in glob.glob(f'results/{stem}_E5_s*/metrics_history.csv'):
        df = pd.read_csv(f)
        seed = int(f.split('_s')[-1].split('/')[0])
        df['seed'] = seed
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else None


def load_finals(stem: str) -> list[dict]:
    out = []
    for f in glob.glob(f'results/{stem}_E5_s*/global_test_metrics.json'):
        with open(f) as fp:
            out.append(json.load(fp))
    return out


def plot_curve(ax, metric: str, ylabel: str, title: str, smooth: int = 5) -> None:
    for stem, color, label in [
        (FEDAVG_STEM, FEDAVG_COLOR, 'FedAvg'),
        (FEDPROX_STEM, FEDPROX_COLOR, 'FedProx (μ=0.01)'),
    ]:
        df = load_history(stem)
        if df is None:
            ax.text(0.5, 0.5, f'No data yet for {stem}',
                    ha='center', va='center', transform=ax.transAxes, fontsize=9)
            continue
        agg = df.groupby('round')[metric].agg(['mean', 'std']).reset_index()
        if smooth > 1:
            agg['m'] = agg['mean'].rolling(smooth, min_periods=1, center=True).mean()
            agg['s'] = agg['std'].rolling(smooth, min_periods=1, center=True).mean()
        else:
            agg['m'] = agg['mean']; agg['s'] = agg['std']
        ax.plot(agg['round'], agg['m'], color=color, linewidth=2.2, label=label)
        ax.fill_between(agg['round'], agg['m'] - agg['s'], agg['m'] + agg['s'],
                        color=color, alpha=0.18)
    ax.set_xlabel('Communication round')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 100)


def plot_stability(ax, window: int = 10) -> None:
    for stem, color, label in [
        (FEDAVG_STEM, FEDAVG_COLOR, 'FedAvg'),
        (FEDPROX_STEM, FEDPROX_COLOR, 'FedProx (μ=0.01)'),
    ]:
        df = load_history(stem)
        if df is None:
            return
        rolling = (df.sort_values(['seed', 'round'])
                     .groupby('seed')['balanced_accuracy']
                     .rolling(window, min_periods=window).std()
                     .reset_index(level=0, drop=True))
        df = df.assign(rolling=rolling)
        agg = df.groupby('round')['rolling'].agg(['mean', 'std']).reset_index()
        ax.plot(agg['round'], agg['mean'], color=color, linewidth=2.2, label=label)
        ax.fill_between(agg['round'],
                        agg['mean'] - agg['std'].fillna(0),
                        agg['mean'] + agg['std'].fillna(0),
                        color=color, alpha=0.18)
    ax.set_xlabel('Communication round')
    ax.set_ylabel(f'Rolling stdev of bACC (window={window})')
    ax.set_title('(b) Round-to-round stability (lower → more stable)')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)


def plot_final_bars(ax) -> None:
    fa = load_finals(FEDAVG_STEM)
    fp = load_finals(FEDPROX_STEM)
    if not fa or not fp:
        ax.text(0.5, 0.5, 'No data yet — run the SLURM jobs first',
                ha='center', va='center', transform=ax.transAxes, fontsize=11)
        return

    melanoma_idx = 4
    metrics = [
        ('Balanced\naccuracy',  lambda d: d['balanced_accuracy']),
        ('Macro F1',             lambda d: d['macro_f1']),
        ('Worst-class\nF1',      lambda d: d['worst_class_f1']),
        ('Melanoma\nrecall',     lambda d: d['per_class_recall'][melanoma_idx]),
    ]

    fa_m, fa_s, fp_m, fp_s, ps = [], [], [], [], []
    for _, getter in metrics:
        fa_v = [getter(d) for d in fa]
        fp_v = [getter(d) for d in fp]
        fa_m.append(np.mean(fa_v)); fa_s.append(np.std(fa_v))
        fp_m.append(np.mean(fp_v)); fp_s.append(np.std(fp_v))
        if len(fa_v) == len(fp_v) and len(fa_v) >= 2:
            _, p = stats.ttest_rel(fp_v, fa_v)
            ps.append(p)
        else:
            ps.append(float('nan'))

    x = np.arange(len(metrics))
    w = 0.36
    b1 = ax.bar(x - w/2, fa_m, w, yerr=fa_s, capsize=4,
                color=FEDAVG_COLOR, edgecolor='white', label='FedAvg')
    b2 = ax.bar(x + w/2, fp_m, w, yerr=fp_s, capsize=4,
                color=FEDPROX_COLOR, edgecolor='white', label='FedProx (μ=0.01)')

    for bar, v in zip(b1, fa_m):
        ax.annotate(f'{v:.3f}', xy=(bar.get_x() + bar.get_width()/2, v),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=8)
    for bar, v in zip(b2, fp_m):
        ax.annotate(f'{v:.3f}', xy=(bar.get_x() + bar.get_width()/2, v),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=8)

    for i, p in enumerate(ps):
        if np.isnan(p):
            continue
        marker = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        top = max(fa_m[i] + fa_s[i], fp_m[i] + fp_s[i])
        ax.annotate(f'{marker}\np={p:.3f}', xy=(i, top + 0.04),
                    ha='center', va='bottom', fontsize=8,
                    color='black' if marker != 'ns' else 'gray',
                    fontweight='bold' if marker != 'ns' else 'normal')

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for lbl, _ in metrics], fontsize=9)
    ax.set_ylabel('Score (mean ± std, n=3)')
    ax.set_title('(d) Final-round metrics — paired t-test, n=3')
    ax.set_ylim(0, max(0.7, max(fa_m + fp_m) + 0.15))
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3, axis='y')


def show_epoch_distribution() -> None:
    """Sanity check: verify system heterogeneity actually happened."""
    for stem in (FEDAVG_STEM, FEDPROX_STEM):
        files = glob.glob(f'results/{stem}_E5_s*/epochs_history.csv')
        if not files:
            print(f'  WARNING: no epochs_history.csv for {stem}')
            continue
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        counts = df['local_epochs'].value_counts().sort_index().to_dict()
        print(f'  {stem}: epoch distribution = {counts}, mean = {df["local_epochs"].mean():.2f}')


def main() -> None:
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10})

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    plot_curve(axes[0, 0], 'balanced_accuracy', 'Balanced accuracy',
               '(a) Balanced accuracy over rounds')
    plot_stability(axes[0, 1])
    plot_curve(axes[1, 0], 'worst_class_f1', 'Worst-class F1',
               '(c) Worst-class F1 over rounds')
    plot_final_bars(axes[1, 1])

    fig.suptitle('System heterogeneity: variable local epochs per client (∈ {1,2,5,10,20}), Dir(α=0.3)',
                 fontsize=13, fontweight='bold', y=1.02)

    out_pdf = FIG / 'fig_sysheterogeneity.pdf'
    out_png = FIG / 'fig_sysheterogeneity.png'
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out_pdf}')
    print(f'wrote {out_png}')

    print('\n=== System heterogeneity verification ===')
    show_epoch_distribution()

    fa = load_finals(FEDAVG_STEM)
    fp = load_finals(FEDPROX_STEM)
    if fa and fp:
        print('\n=== Final-round summary ===')
        melanoma_idx = 4
        for key, g in {
            'balanced_accuracy': lambda d: d['balanced_accuracy'],
            'macro_f1':          lambda d: d['macro_f1'],
            'worst_class_f1':    lambda d: d['worst_class_f1'],
            'melanoma_recall':   lambda d: d['per_class_recall'][melanoma_idx],
        }.items():
            fa_v = [g(d) for d in fa]; fp_v = [g(d) for d in fp]
            d_mean = np.mean(fp_v) - np.mean(fa_v)
            t, p = stats.ttest_rel(fp_v, fa_v) if len(fa_v) >= 2 else (float('nan'), float('nan'))
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            print(f'  {key:20s}  FedAvg={np.mean(fa_v):.3f}±{np.std(fa_v):.3f}  '
                  f'FedProx={np.mean(fp_v):.3f}±{np.std(fp_v):.3f}  '
                  f'Δ={d_mean:+.3f}  p={p:.4f} {sig}')


if __name__ == '__main__':
    main()
