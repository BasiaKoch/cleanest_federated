"""Compare FedAvg vs FedProx at Dirichlet α=0.3.

Generates a focused 3-panel figure:
  (a) Balanced accuracy over rounds — convergence + stability comparison
  (b) Worst-class F1 over rounds   — minority-class rescue
  (c) Final-round summary bars     — bACC, macro-F1, worst-F1, melanoma recall

Reads from results/fedavg_dir03_E5_s{42,123,456}/ and
            results/fedprox_dir03_E5_s{42,123,456}/.
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


def plot_curve(ax, regime_stem_fedavg: str, regime_stem_fedprox: str,
               metric: str, ylabel: str, title: str, smooth_window: int = 5) -> None:
    """Plot mean ± std convergence curves for FedAvg and FedProx."""
    for stem, color, label in [
        (regime_stem_fedavg, FEDAVG_COLOR, 'FedAvg'),
        (regime_stem_fedprox, FEDPROX_COLOR, 'FedProx (μ=0.01)'),
    ]:
        df = load_history(stem)
        if df is None:
            print(f'WARNING: no data for {stem}')
            continue
        agg = df.groupby('round')[metric].agg(['mean', 'std']).reset_index()
        if smooth_window > 1:
            agg['mean_s'] = agg['mean'].rolling(smooth_window, min_periods=1, center=True).mean()
            agg['std_s']  = agg['std'].rolling(smooth_window, min_periods=1, center=True).mean()
        else:
            agg['mean_s'] = agg['mean']
            agg['std_s']  = agg['std']

        ax.plot(agg['round'], agg['mean_s'], color=color, linewidth=2.2, label=label)
        ax.fill_between(agg['round'],
                        agg['mean_s'] - agg['std_s'],
                        agg['mean_s'] + agg['std_s'],
                        color=color, alpha=0.18)

    ax.set_xlabel('Communication round')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 100)


def plot_final_bars(ax) -> None:
    """Side-by-side bars: FedAvg vs FedProx final-round metrics."""
    fa = load_finals('fedavg_dir03')
    fp = load_finals('fedprox_dir03')
    if not fa or not fp:
        ax.text(0.5, 0.5, 'No data yet — run the SLURM jobs first',
                ha='center', va='center', transform=ax.transAxes, fontsize=11)
        return

    melanoma_idx = 4   # DermaMNIST class order

    metrics_def = [
        ('balanced_accuracy', 'Balanced\naccuracy',     lambda d: d['balanced_accuracy']),
        ('macro_f1',          'Macro F1',                lambda d: d['macro_f1']),
        ('worst_class_f1',    'Worst-class\nF1',         lambda d: d['worst_class_f1']),
        ('melanoma_recall',   'Melanoma\nrecall',        lambda d: d['per_class_recall'][melanoma_idx]),
    ]

    fa_means, fa_stds, fp_means, fp_stds, p_values = [], [], [], [], []
    for key, _, getter in metrics_def:
        fa_vals = [getter(d) for d in fa]
        fp_vals = [getter(d) for d in fp]
        fa_means.append(np.mean(fa_vals))
        fa_stds.append(np.std(fa_vals))
        fp_means.append(np.mean(fp_vals))
        fp_stds.append(np.std(fp_vals))
        # Paired t-test (3 seeds) — only meaningful if seeds matched
        if len(fa_vals) == len(fp_vals) and len(fa_vals) >= 2:
            t, p = stats.ttest_rel(fp_vals, fa_vals)
            p_values.append(p)
        else:
            p_values.append(float('nan'))

    x = np.arange(len(metrics_def))
    w = 0.36
    bars1 = ax.bar(x - w/2, fa_means, w, yerr=fa_stds, capsize=4,
                   color=FEDAVG_COLOR, edgecolor='white', label='FedAvg')
    bars2 = ax.bar(x + w/2, fp_means, w, yerr=fp_stds, capsize=4,
                   color=FEDPROX_COLOR, edgecolor='white', label='FedProx (μ=0.01)')

    # Value labels
    for bar, v in zip(bars1, fa_means):
        ax.annotate(f'{v:.3f}', xy=(bar.get_x() + bar.get_width()/2, v),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=8)
    for bar, v in zip(bars2, fp_means):
        ax.annotate(f'{v:.3f}', xy=(bar.get_x() + bar.get_width()/2, v),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=8)

    # Significance markers above each metric group
    for i, p in enumerate(p_values):
        if np.isnan(p):
            continue
        marker = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        top = max(fa_means[i] + fa_stds[i], fp_means[i] + fp_stds[i])
        ax.annotate(f'{marker}\np={p:.3f}', xy=(i, top + 0.04),
                    ha='center', va='bottom', fontsize=8,
                    color='black' if marker != 'ns' else 'gray',
                    fontweight='bold' if marker != 'ns' else 'normal')

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl, _ in metrics_def], fontsize=9)
    ax.set_ylabel('Score (mean ± std, n=3)')
    ax.set_title('Final-round comparison @ Dir(0.3)')
    ax.set_ylim(0, max(0.7, max(fa_means + fp_means) + 0.15))
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3, axis='y')


def main() -> None:
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
    })

    fig = plt.figure(figsize=(14, 5.5), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.2])

    ax_acc  = fig.add_subplot(gs[0, 0])
    ax_wf1  = fig.add_subplot(gs[0, 1])
    ax_bars = fig.add_subplot(gs[0, 2])

    plot_curve(ax_acc, 'fedavg_dir03', 'fedprox_dir03',
               'balanced_accuracy', 'Balanced accuracy',
               '(a) Balanced accuracy over rounds')
    plot_curve(ax_wf1, 'fedavg_dir03', 'fedprox_dir03',
               'worst_class_f1', 'Worst-class F1',
               '(b) Worst-class F1 over rounds')
    plot_final_bars(ax_bars)

    fig.suptitle('FedAvg vs FedProx @ Dirichlet α=0.3, 10 clients, 100 rounds, E=5',
                 fontsize=13, fontweight='bold', y=1.02)

    out_pdf = FIG / 'fig_dir03_comparison.pdf'
    out_png = FIG / 'fig_dir03_comparison.png'
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out_pdf}')
    print(f'wrote {out_png}')

    # Print numerical summary to console
    fa = load_finals('fedavg_dir03')
    fp = load_finals('fedprox_dir03')
    if fa and fp:
        print('\n=== Final-round summary @ Dir(0.3) ===')
        for key in ['balanced_accuracy', 'macro_f1', 'worst_class_f1']:
            fa_vals = [d[key] for d in fa]
            fp_vals = [d[key] for d in fp]
            print(f'  {key:20s}  FedAvg={np.mean(fa_vals):.3f}±{np.std(fa_vals):.3f}  '
                  f'FedProx={np.mean(fp_vals):.3f}±{np.std(fp_vals):.3f}  '
                  f'Δ={np.mean(fp_vals)-np.mean(fa_vals):+.3f}')
            if len(fa_vals) >= 2 and len(fp_vals) >= 2:
                t, p = stats.ttest_rel(fp_vals, fa_vals)
                print(f'                          paired t-test: t={t:+.3f}, p={p:.4f}')


if __name__ == '__main__':
    main()
