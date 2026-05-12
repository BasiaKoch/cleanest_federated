"""FedProx best-case showcase plot.

Compares FedAvg vs FedProx under conditions designed to maximise
client drift (and therefore FedProx's expected advantage):
  - Dirichlet α=0.1
  - local epochs = 20
  - fraction_fit = 0.5

Four panels:
  (a) Balanced accuracy over rounds       — does FedProx reach higher?
  (b) Last-10-round rolling stdev of bACC — does FedProx stabilise?
  (c) Worst-class F1 over rounds          — minority-class rescue?
  (d) Final-round metric bars + paired t-test significance markers
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

FEDAVG_STEM = 'fedavg_showcase'
FEDPROX_STEM = 'fedprox_showcase'


def load_history(stem: str) -> pd.DataFrame | None:
    rows = []
    for f in glob.glob(f'results/{stem}_E20_s*/metrics_history.csv'):
        df = pd.read_csv(f)
        seed = int(f.split('_s')[-1].split('/')[0])
        df['seed'] = seed
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else None


def load_finals(stem: str) -> list[dict]:
    out = []
    for f in glob.glob(f'results/{stem}_E20_s*/global_test_metrics.json'):
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
            ax.text(0.5, 0.5, f'No data yet for {stem}\nRun the SLURM jobs first',
                    ha='center', va='center', transform=ax.transAxes, fontsize=9)
            return
        agg = df.groupby('round')[metric].agg(['mean', 'std']).reset_index()
        if smooth > 1:
            agg['m'] = agg['mean'].rolling(smooth, min_periods=1, center=True).mean()
            agg['s'] = agg['std'].rolling(smooth, min_periods=1, center=True).mean()
        else:
            agg['m'] = agg['mean']
            agg['s'] = agg['std']
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
    """Rolling stdev of bACC over a `window`-round window. Lower = more stable."""
    for stem, color, label in [
        (FEDAVG_STEM, FEDAVG_COLOR, 'FedAvg'),
        (FEDPROX_STEM, FEDPROX_COLOR, 'FedProx (μ=0.01)'),
    ]:
        df = load_history(stem)
        if df is None:
            return
        # Per-seed rolling std of bACC, then mean across seeds
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
    ax.set_title(f'(b) Round-to-round stability (lower → more stable)')
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
        ('Balanced\naccuracy',     lambda d: d['balanced_accuracy']),
        ('Macro F1',                lambda d: d['macro_f1']),
        ('Worst-class\nF1',         lambda d: d['worst_class_f1']),
        ('Melanoma\nrecall',        lambda d: d['per_class_recall'][melanoma_idx]),
    ]

    fa_m, fa_s, fp_m, fp_s, ps = [], [], [], [], []
    for _, getter in metrics:
        fa_vals = [getter(d) for d in fa]
        fp_vals = [getter(d) for d in fp]
        fa_m.append(np.mean(fa_vals)); fa_s.append(np.std(fa_vals))
        fp_m.append(np.mean(fp_vals)); fp_s.append(np.std(fp_vals))
        if len(fa_vals) == len(fp_vals) and len(fa_vals) >= 2:
            _, p = stats.ttest_rel(fp_vals, fa_vals)
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
    ax.set_title('(d) Final-round metrics — paired t-test across 3 seeds')
    ax.set_ylim(0, max(0.7, max(fa_m + fp_m) + 0.15))
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3, axis='y')


def print_summary() -> None:
    fa = load_finals(FEDAVG_STEM)
    fp = load_finals(FEDPROX_STEM)
    if not fa or not fp:
        print('No final-round data yet.')
        return
    print('\n=== FedProx best-case showcase — final-round summary ===')
    print('Setup: Dir(α=0.1), E=20, fraction_fit=0.5, 100 rounds, 3 seeds')
    print()
    melanoma_idx = 4
    metric_getters = {
        'balanced_accuracy': lambda d: d['balanced_accuracy'],
        'macro_f1':          lambda d: d['macro_f1'],
        'worst_class_f1':    lambda d: d['worst_class_f1'],
        'melanoma_recall':   lambda d: d['per_class_recall'][melanoma_idx],
    }
    for key, g in metric_getters.items():
        fa_v = [g(d) for d in fa]
        fp_v = [g(d) for d in fp]
        d_mean = np.mean(fp_v) - np.mean(fa_v)
        t, p = stats.ttest_rel(fp_v, fa_v) if len(fa_v) >= 2 else (float('nan'), float('nan'))
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f'  {key:20s}  FedAvg={np.mean(fa_v):.3f}±{np.std(fa_v):.3f}  '
              f'FedProx={np.mean(fp_v):.3f}±{np.std(fp_v):.3f}  '
              f'Δ={d_mean:+.3f}  p={p:.4f} {sig}')


def main() -> None:
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
    })

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    plot_curve(axes[0, 0], 'balanced_accuracy', 'Balanced accuracy',
               '(a) Balanced accuracy over rounds')
    plot_stability(axes[0, 1])
    plot_curve(axes[1, 0], 'worst_class_f1', 'Worst-class F1',
               '(c) Worst-class F1 over rounds')
    plot_final_bars(axes[1, 1])

    fig.suptitle('FedProx best-case showcase: Dir(α=0.1), E=20, fraction_fit=0.5',
                 fontsize=14, fontweight='bold', y=1.02)

    out_pdf = FIG / 'fig_showcase.pdf'
    out_png = FIG / 'fig_showcase.png'
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out_pdf}')
    print(f'wrote {out_png}')

    print_summary()


if __name__ == '__main__':
    main()
