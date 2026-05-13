"""Thesis-grade plots A–F for the FedProx vs FedAvg headline experiment."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CLASS_NAMES = [
    'actinic\nkeratoses', 'basal cell\ncarcinoma', 'benign\nkeratosis',
    'dermatofibroma', 'melanoma', 'melanocytic\nnevi', 'vascular\nlesions',
]

FA_COLOR = '#3182ce'
FP_COLOR = '#dd6b20'


def load_history(stem_prefix: str, results_dir: str = 'results/thesis') -> pd.DataFrame | None:
    rows = []
    for f in glob.glob(f'{results_dir}/{stem_prefix}_E*_s*/metrics_history.csv'):
        df = pd.read_csv(f)
        df['seed'] = int(f.split('_s')[-1].split('/')[0])
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else None


def load_finals(stem_prefix: str, results_dir: str = 'results/thesis') -> list[dict]:
    out = []
    for f in glob.glob(f'{results_dir}/{stem_prefix}_E*_s*/global_test_metrics.json'):
        with open(f) as fp:
            out.append(json.load(fp))
    return out


def load_local_drift(stem_prefix: str, results_dir: str = 'results/thesis') -> pd.DataFrame | None:
    rows = []
    for f in glob.glob(f'{results_dir}/{stem_prefix}_E*_s*/local_drift_per_round.csv'):
        df = pd.read_csv(f)
        df['seed'] = int(f.split('_s')[-1].split('/')[0])
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else None


def plot_train_loss_curves(ax, results_dir, fedavg_stem='fedavg', fedprox_stem='fedprox_mu01'):
    """PATCH 3 PRIMARY PANEL: training loss vs round (paper Fig. 1 style)."""
    for stem, color, label in [(fedavg_stem, FA_COLOR, 'FedAvg'),
                                (fedprox_stem, FP_COLOR, 'FedProx (μ=0.1)')]:
        df = load_history(stem, results_dir)
        if df is None or 'train_loss_weighted' not in df.columns:
            continue
        agg = df.groupby('round')['train_loss_weighted'].agg(['mean', 'std', 'count']).reset_index()
        agg['sem'] = agg['std'] / np.sqrt(agg['count'].clip(lower=1))
        ax.plot(agg['round'], agg['mean'], color=color, linewidth=2, label=label)
        ax.fill_between(agg['round'], agg['mean'] - agg['sem'], agg['mean'] + agg['sem'],
                        color=color, alpha=0.20)
    ax.set_xlabel('Communication round')
    ax.set_ylabel('Training loss (size-weighted across clients)')
    ax.set_title('(A) PRIMARY — Training loss vs round (paper Fig. 1 style)')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)


def plot_headline_curves(ax, results_dir, fedavg_stem='fedavg', fedprox_stem='fedprox_mu01'):
    """Panel A2 (now secondary): global val macro-F1 vs round."""
    for stem, color, label in [(fedavg_stem, FA_COLOR, 'FedAvg'),
                                (fedprox_stem, FP_COLOR, 'FedProx (μ=0.1)')]:
        df = load_history(stem, results_dir)
        if df is None:
            continue
        agg = df.groupby('round')['macro_f1'].agg(['mean', 'std', 'count']).reset_index()
        agg['sem'] = agg['std'] / np.sqrt(agg['count'].clip(lower=1))
        ax.plot(agg['round'], agg['mean'], color=color, linewidth=2, label=label)
        ax.fill_between(agg['round'], agg['mean'] - agg['sem'], agg['mean'] + agg['sem'],
                        color=color, alpha=0.20)
    ax.set_xlabel('Communication round')
    ax.set_ylabel('Global validation macro-F1')
    ax.set_title('(A2) SECONDARY — Macro-F1 vs round')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)


def plot_grad_variance(ax, results_dir, fedavg_stem='fedavg', fedprox_stem='fedprox_mu01'):
    """PATCH 1: gradient-variance dissimilarity vs round."""
    for stem, color, label in [(fedavg_stem, FA_COLOR, 'FedAvg'),
                                (fedprox_stem, FP_COLOR, 'FedProx (μ=0.1)')]:
        df = load_history(stem, results_dir)
        if df is None or 'grad_variance' not in df.columns:
            continue
        agg = df.groupby('round')['grad_variance'].agg(['mean', 'std', 'count']).reset_index()
        agg['sem'] = agg['std'] / np.sqrt(agg['count'].clip(lower=1))
        ax.plot(agg['round'], agg['mean'], color=color, linewidth=2, label=label)
        ax.fill_between(agg['round'], agg['mean'] - agg['sem'], agg['mean'] + agg['sem'],
                        color=color, alpha=0.20)
    ax.set_xlabel('Communication round')
    ax.set_ylabel('E_k[||∇F_k − ∇f||²]  (lower = aligned)')
    ax.set_title('(G) Gradient-variance dissimilarity (Patch 1 / paper Fig. 2)')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_yscale('symlog', linthresh=1e-3)
    ax.grid(alpha=0.3)


def plot_test_f1_vs_E(ax, results_dir):
    """Panel B: final selected test macro-F1 vs E."""
    Es = [1, 5, 10, 20, 40]
    fa_means, fa_stds = [], []
    fp_means, fp_stds = [], []
    for E in Es:
        fa_stem = 'fedavg' if E == 20 else f'fedavg_E{E}'
        fp_stem = 'fedprox_mu01' if E == 20 else f'fedprox_E{E}'
        fa = load_finals(fa_stem, results_dir)
        fp = load_finals(fp_stem, results_dir)
        fa_v = [d['macro_f1'] for d in fa]
        fp_v = [d['macro_f1'] for d in fp]
        fa_means.append(np.mean(fa_v) if fa_v else np.nan)
        fa_stds.append(np.std(fa_v) if fa_v else 0)
        fp_means.append(np.mean(fp_v) if fp_v else np.nan)
        fp_stds.append(np.std(fp_v) if fp_v else 0)
    x = np.arange(len(Es))
    ax.errorbar(x, fa_means, yerr=fa_stds, marker='o', linewidth=2,
                color=FA_COLOR, label='FedAvg', capsize=4, markersize=8)
    ax.errorbar(x, fp_means, yerr=fp_stds, marker='s', linewidth=2,
                color=FP_COLOR, label='FedProx (μ=0.1)', capsize=4, markersize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(E) for E in Es])
    ax.set_xlabel('Local epochs E')
    ax.set_ylabel('Test macro-F1 at best-val checkpoint')
    ax.set_title('(B) Test macro-F1 vs local epochs')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)


def plot_per_class_f1(ax, results_dir, fedavg_stem='fedavg', fedprox_stem='fedprox_mu01'):
    """Panel C: per-class test F1 bars."""
    fa = load_finals(fedavg_stem, results_dir)
    fp = load_finals(fedprox_stem, results_dir)
    if not fa or not fp:
        ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes)
        return
    fa_pc = np.array([d['per_class_f1'] for d in fa])
    fp_pc = np.array([d['per_class_f1'] for d in fp])
    x = np.arange(len(CLASS_NAMES))
    w = 0.38
    ax.bar(x - w/2, fa_pc.mean(0), w, yerr=fa_pc.std(0), capsize=3,
           color=FA_COLOR, label='FedAvg', edgecolor='white')
    ax.bar(x + w/2, fp_pc.mean(0), w, yerr=fp_pc.std(0), capsize=3,
           color=FP_COLOR, label='FedProx (μ=0.1)', edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, fontsize=8)
    ax.set_ylabel('Test F1 (mean ± std)')
    ax.set_title('(C) Per-class test F1')
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(fa_pc.max(), fp_pc.max(), 0.5) + 0.05)
    ax.grid(alpha=0.3, axis='y')


def plot_drift(ax, results_dir, fedavg_stem='fedavg', fedprox_stem='fedprox_mu01'):
    """Panel D: per-round std across local models of global val loss."""
    for stem, color, label in [(fedavg_stem, FA_COLOR, 'FedAvg'),
                                (fedprox_stem, FP_COLOR, 'FedProx (μ=0.1)')]:
        df = load_local_drift(stem, results_dir)
        if df is None:
            continue
        agg = df.groupby('round')['val_loss_std'].agg(['mean', 'std', 'count']).reset_index()
        agg['sem'] = agg['std'] / np.sqrt(agg['count'].clip(lower=1))
        ax.plot(agg['round'], agg['mean'], color=color, linewidth=2, label=label)
        ax.fill_between(agg['round'],
                        agg['mean'] - agg['sem'], agg['mean'] + agg['sem'],
                        color=color, alpha=0.20)
    ax.set_xlabel('Communication round')
    ax.set_ylabel('std across local models of global val loss')
    ax.set_title('(D) Drift diagnostic (lower → more aligned clients)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)


def plot_partition_heatmap(ax, results_dir):
    """Panel E: client × class distribution heatmap (from one seed's run)."""
    # Use first available run
    candidates = glob.glob(f'{results_dir}/fedavg*_E*_s*/data_distributions/client_train_distributions.csv')
    if not candidates:
        ax.text(0.5, 0.5, 'no partition data', ha='center', va='center', transform=ax.transAxes)
        return
    df = pd.read_csv(candidates[0], index_col=0)
    df.columns = [c[:8] for c in df.columns]
    im = ax.imshow(df.values, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([f'C{i}' for i in df.index], fontsize=8)
    for i in range(len(df)):
        for j in range(len(df.columns)):
            v = df.values[i, j]
            ax.text(j, i, str(int(v)), ha='center', va='center',
                    fontsize=7, color='black' if v < df.values.max() * 0.5 else 'white')
    ax.set_title('(E) Client-class distribution')
    plt.colorbar(im, ax=ax, shrink=0.7)


def plot_mu_sweep(ax, results_dir):
    """Panel F: test macro-F1 vs μ (μ=0 is FedAvg)."""
    mu_runs = [
        (0.0,   'fedavg'),
        (0.001, 'fedprox_mu_0001'),
        (0.01,  'fedprox_mu_001'),
        (0.1,   'fedprox_mu01'),
        (0.5,   'fedprox_mu_05'),
        (1.0,   'fedprox_mu_1'),
        (2.0,   'fedprox_mu_2'),
    ]
    mus, means, stds = [], [], []
    for mu, stem in mu_runs:
        finals = load_finals(stem, results_dir)
        v = [d['macro_f1'] for d in finals]
        if v:
            mus.append(mu); means.append(np.mean(v)); stds.append(np.std(v))
    if not mus:
        ax.text(0.5, 0.5, 'no μ sweep data', ha='center', va='center', transform=ax.transAxes)
        return
    ax.errorbar(range(len(mus)), means, yerr=stds, marker='D', linewidth=2,
                color=FP_COLOR, capsize=4, markersize=9)
    ax.set_xticks(range(len(mus)))
    ax.set_xticklabels([str(mu) for mu in mus])
    ax.set_xlabel('Proximal coefficient μ  (μ=0 ≡ FedAvg)')
    ax.set_ylabel('Test macro-F1 (mean ± std)')
    ax.set_title('(F) μ sweep — selected by validation')
    ax.grid(alpha=0.3)
    # Highlight peak
    best = int(np.argmax(means))
    ax.annotate(f'best μ = {mus[best]}', xy=(best, means[best]),
                xytext=(best, means[best] + 0.04),
                ha='center', fontsize=9, fontweight='bold',
                arrowprops=dict(arrowstyle='->'))


def plot_heterogeneity_trend(ax, results_dir):
    """PATCH 4: final macro-F1 vs heterogeneity level + (FedProx − FedAvg) gap."""
    levels = [('low', 0.2), ('medium', 0.4), ('high', 0.6), ('extreme', 0.9)]
    fa_means, fa_stds, fp_means, fp_stds = [], [], [], []
    for level, _ in levels:
        fa = load_finals(f'fedavg_het_{level}', results_dir)
        fp = load_finals(f'fedprox_het_{level}', results_dir)
        fa_v = [d['macro_f1'] for d in fa]
        fp_v = [d['macro_f1'] for d in fp]
        fa_means.append(np.mean(fa_v) if fa_v else np.nan)
        fa_stds.append(np.std(fa_v) if fa_v else 0)
        fp_means.append(np.mean(fp_v) if fp_v else np.nan)
        fp_stds.append(np.std(fp_v) if fp_v else 0)
    x = np.arange(len(levels))
    labels = [f'{n}\n(dom={d:.0%})' for n, d in levels]
    ax.errorbar(x, fa_means, yerr=fa_stds, marker='o', linewidth=2,
                color=FA_COLOR, label='FedAvg', capsize=4)
    ax.errorbar(x, fp_means, yerr=fp_stds, marker='s', linewidth=2,
                color=FP_COLOR, label='FedProx (μ=0.1)', capsize=4)
    # Annotate gap above each level
    for i in range(len(levels)):
        gap = fp_means[i] - fa_means[i] if not np.isnan(fp_means[i]) and not np.isnan(fa_means[i]) else np.nan
        if not np.isnan(gap):
            ax.annotate(f'Δ={gap:+.3f}', xy=(i, max(fa_means[i], fp_means[i]) + 0.02),
                        ha='center', fontsize=8, color='black')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel('Heterogeneity level')
    ax.set_ylabel('Final test macro-F1')
    ax.set_title('(H) Heterogeneity trend — FedProx gap should grow with skew (Patch 4)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', default='results/thesis')
    ap.add_argument('--out', default='results/thesis/figures')
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11})

    fig, axes = plt.subplots(4, 2, figsize=(14, 20), constrained_layout=True)
    plot_train_loss_curves(axes[0, 0], args.results_dir)        # NEW (Patch 3 primary)
    plot_headline_curves(axes[0, 1], args.results_dir)           # was primary
    plot_grad_variance(axes[1, 0], args.results_dir)             # NEW (Patch 1)
    plot_drift(axes[1, 1], args.results_dir)
    plot_per_class_f1(axes[2, 0], args.results_dir)
    plot_test_f1_vs_E(axes[2, 1], args.results_dir)
    plot_partition_heatmap(axes[3, 0], args.results_dir)
    plot_mu_sweep(axes[3, 1], args.results_dir)
    fig.suptitle('Thesis FedProx-favorable experiment — DermaMNIST, mixed-type partition',
                 fontsize=13, fontweight='bold', y=1.01)
    out_png = out_dir / 'thesis_panel.png'
    out_pdf = out_dir / 'thesis_panel.pdf'
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out_png}')
    print(f'wrote {out_pdf}')


if __name__ == '__main__':
    main()
