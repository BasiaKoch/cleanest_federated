"""Specialist-client experiment plot.

Each of 7 clients trains on a single DermaMNIST class. Aggregated model is
evaluated on the global test set (natural class distribution, all 7 classes).

Following Zhao et al. 2018 and Li et al. 2020.

Four panels:
  (a) Balanced accuracy over rounds — convergence
  (b) Per-class F1 at final round    — which classes survive aggregation
  (c) Confusion matrix grid           — visualize aggregation collapse
  (d) Bar chart of bACC / macro-F1 / worst-F1 across μ values + paired t-tests
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

CONFIGS = [
    ('fedavg_specialist',         'FedAvg',          '#3182ce'),
    ('fedprox_specialist_mu001',  'FedProx μ=0.01',  '#f6ad55'),
    ('fedprox_specialist_mu01',   'FedProx μ=0.1',   '#dd6b20'),
    ('fedprox_specialist_mu10',   'FedProx μ=1.0',   '#9b2c2c'),
]

CLASS_NAMES = [
    'actinic\nkeratoses', 'basal cell\ncarcinoma', 'benign\nkeratosis',
    'dermatofibroma', 'melanoma', 'melanocytic\nnevi', 'vascular\nlesions',
]


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


def panel_convergence(ax) -> None:
    for stem, label, color in CONFIGS:
        df = load_history(stem)
        if df is None:
            continue
        agg = df.groupby('round')['balanced_accuracy'].agg(['mean', 'std']).reset_index()
        agg['m'] = agg['mean'].rolling(5, min_periods=1, center=True).mean()
        agg['s'] = agg['std'].rolling(5, min_periods=1, center=True).mean()
        ax.plot(agg['round'], agg['m'], color=color, linewidth=2, label=label)
        ax.fill_between(agg['round'], agg['m'] - agg['s'], agg['m'] + agg['s'],
                        color=color, alpha=0.15)
    # Reference line: predict-majority-only baseline ≈ 1/7 = 0.143 bACC
    ax.axhline(1.0/7.0, linestyle=':', color='gray', alpha=0.6,
               label='Random/uniform bACC (1/7)')
    ax.set_xlabel('Communication round')
    ax.set_ylabel('Balanced accuracy')
    ax.set_title('(a) Convergence on global test set (3 seeds, smoothed)')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 100)


def panel_per_class_f1(ax) -> None:
    n = len(CLASS_NAMES)
    width = 0.20
    x = np.arange(n)
    for i, (stem, label, color) in enumerate(CONFIGS):
        finals = load_finals(stem)
        if not finals:
            continue
        per_class = np.array([d['per_class_f1'] for d in finals])
        means = per_class.mean(axis=0)
        stds = per_class.std(axis=0)
        offset = (i - (len(CONFIGS) - 1) / 2) * width
        ax.bar(x + offset, means, width, yerr=stds, label=label,
               color=color, capsize=2, edgecolor='white', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, fontsize=7)
    ax.set_ylabel('F1 score (mean ± std, n=3)')
    ax.set_title('(b) Per-class F1 at final round')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3, axis='y')


def panel_summary_bars(ax) -> None:
    """Final-round bACC, macro-F1, worst-F1 per config; t-test vs FedAvg."""
    bACC_means, bACC_stds = [], []
    mF1_means, mF1_stds = [], []
    wF1_means, wF1_stds = [], []
    labels, colors = [], []

    fedavg_finals = load_finals('fedavg_specialist')
    fedavg_bACC = [d['balanced_accuracy'] for d in fedavg_finals] if fedavg_finals else []

    for stem, label, color in CONFIGS:
        finals = load_finals(stem)
        if not finals:
            continue
        b = [d['balanced_accuracy'] for d in finals]
        m = [d['macro_f1'] for d in finals]
        w = [d['worst_class_f1'] for d in finals]
        bACC_means.append(np.mean(b)); bACC_stds.append(np.std(b))
        mF1_means.append(np.mean(m)); mF1_stds.append(np.std(m))
        wF1_means.append(np.mean(w)); wF1_stds.append(np.std(w))
        labels.append(label); colors.append(color)

    x = np.arange(len(labels))
    width = 0.27
    ax.bar(x - width, bACC_means, width, yerr=bACC_stds, capsize=3,
           color=colors, alpha=0.95, edgecolor='white', label='Balanced acc.')
    ax.bar(x, mF1_means, width, yerr=mF1_stds, capsize=3,
           color=colors, alpha=0.6, edgecolor='white', label='Macro F1')
    ax.bar(x + width, wF1_means, width, yerr=wF1_stds, capsize=3,
           color=colors, alpha=0.3, edgecolor='white', label='Worst-class F1')

    # Significance markers above bACC bars (vs FedAvg)
    if fedavg_bACC and len(fedavg_bACC) >= 2:
        for i, (stem, label, _) in enumerate(CONFIGS):
            if stem == 'fedavg_specialist':
                continue
            finals = load_finals(stem)
            if not finals or len(finals) < 2:
                continue
            b = [d['balanced_accuracy'] for d in finals]
            t, p = stats.ttest_rel(b, fedavg_bACC)
            marker = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            ax.annotate(f'{marker}\np={p:.3f}', xy=(i - width, bACC_means[i] + bACC_stds[i] + 0.02),
                        ha='center', va='bottom', fontsize=7,
                        color='black' if marker != 'ns' else 'gray')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=10)
    ax.set_ylabel('Score (mean ± std, n=3)')
    ax.set_title('(d) Final-round summary — significance vs FedAvg above bACC bars')
    ax.set_ylim(0, max(0.6, max([m + s for m, s in zip(bACC_means, bACC_stds)] + [0.2]) + 0.15))
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3, axis='y')


def panel_confusion_grid(fig, gs) -> None:
    """One confusion matrix per config (averaged across seeds)."""
    axes = [fig.add_subplot(gs[2, i]) for i in range(4)]
    for ax, (stem, label, _) in zip(axes, CONFIGS):
        finals = load_finals(stem)
        if not finals:
            ax.set_title(f'{label}\n(no data)', fontsize=8)
            ax.axis('off')
            continue
        cms = np.array([np.array(d['confusion_matrix']) for d in finals])
        cm = cms.mean(axis=0).astype(float)
        # Row-normalize for visualization
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)
        im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1, aspect='auto')
        ax.set_title(label, fontsize=9)
        ax.set_xticks(range(7)); ax.set_yticks(range(7))
        ax.set_xticklabels(['ak', 'bcc', 'bk', 'df', 'mel', 'nv', 'vasc'], fontsize=6, rotation=45)
        ax.set_yticklabels(['ak', 'bcc', 'bk', 'df', 'mel', 'nv', 'vasc'], fontsize=6)
        if ax is axes[0]:
            ax.set_ylabel('True', fontsize=7)
        ax.set_xlabel('Pred', fontsize=7)


def print_summary() -> None:
    print('\n=== Specialist-client experiment — final-round summary ===')
    print('Setup: 7 specialist clients, 100 rounds, E=5, full participation, 3 seeds')
    print()
    fedavg_finals = load_finals('fedavg_specialist')
    fedavg_bACC = [d['balanced_accuracy'] for d in fedavg_finals] if fedavg_finals else []

    for stem, label, _ in CONFIGS:
        finals = load_finals(stem)
        if not finals:
            print(f'  {label}: NO DATA')
            continue
        b = [d['balanced_accuracy'] for d in finals]
        m = [d['macro_f1'] for d in finals]
        w = [d['worst_class_f1'] for d in finals]
        if stem != 'fedavg_specialist' and fedavg_bACC and len(b) >= 2:
            t, p = stats.ttest_rel(b, fedavg_bACC)
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            extra = f'  Δ_bACC vs FedAvg = {np.mean(b) - np.mean(fedavg_bACC):+.3f}  p={p:.4f} {sig}'
        else:
            extra = ''
        print(f'  {label:18s}  bACC={np.mean(b):.3f}±{np.std(b):.3f}  '
              f'macro_F1={np.mean(m):.3f}±{np.std(m):.3f}  '
              f'worst_F1={np.mean(w):.3f}±{np.std(w):.3f}{extra}')


def main() -> None:
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10})

    fig = plt.figure(figsize=(15, 13), constrained_layout=True)
    gs = fig.add_gridspec(3, 4, height_ratios=[1.2, 1, 1.1])

    ax_conv = fig.add_subplot(gs[0, :2])
    ax_pcf1 = fig.add_subplot(gs[0, 2:])
    ax_sum  = fig.add_subplot(gs[1, :])

    panel_convergence(ax_conv)
    panel_per_class_f1(ax_pcf1)
    panel_summary_bars(ax_sum)
    panel_confusion_grid(fig, gs)
    fig.text(0.01, 0.20, '(c)', fontsize=11, fontweight='bold')

    fig.suptitle('Specialist clients: 7 clients × 1 class each (Zhao 2018-style extreme partition)',
                 fontsize=13, fontweight='bold', y=1.01)

    out_pdf = FIG / 'fig_specialist.pdf'
    out_png = FIG / 'fig_specialist.png'
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out_pdf}')
    print(f'wrote {out_png}')
    print_summary()


if __name__ == '__main__':
    main()
