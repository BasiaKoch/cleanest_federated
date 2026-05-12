"""Polished dissertation plots: aggregated Pareto and per-class F1 grouped bar."""
from __future__ import annotations

import json
import glob
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

FIG = Path('results/figures')
FIG.mkdir(parents=True, exist_ok=True)

# Class names from DermaMNIST INFO (loaded once below)
CLASS_NAMES = [
    'actinic keratoses',
    'basal cell carcinoma',
    'benign keratosis',
    'dermatofibroma',
    'melanoma',
    'melanocytic nevi',
    'vascular lesions',
]

MINORITY = {'dermatofibroma', 'vascular lesions'}


def load_aggregated(stem: str) -> dict:
    """Aggregate metrics for a config across its seeds."""
    rows = []
    for f in glob.glob(f'results/{stem}_E5_s*/global_test_metrics.json'):
        with open(f) as fp:
            rows.append(json.load(fp))
    if not rows:
        return {}
    out = {}
    for key in ['balanced_accuracy', 'macro_f1', 'worst_class_f1']:
        vals = [r[key] for r in rows]
        out[key + '_mean'] = float(np.mean(vals))
        out[key + '_std'] = float(np.std(vals))
    # Per-class F1 — mean across seeds, per class
    per_class = np.array([r['per_class_f1'] for r in rows])
    out['per_class_f1_mean'] = per_class.mean(axis=0)
    out['per_class_f1_std'] = per_class.std(axis=0)
    # Equity gap = max - min across classes (for the seed mean)
    out['equity_gap'] = float(per_class.mean(axis=0).max() - per_class.mean(axis=0).min())
    out['n'] = len(rows)
    return out


def pareto_plot(ax, data: dict[str, dict]) -> None:
    """x = bACC, y = equity gap (lower better), colour = method family."""
    family_color = {
        'iid':            ('#000000', 'o', 'IID baseline'),
        'fedavg_dirichlet':   ('#3182ce', 'o', 'FedAvg + Dirichlet'),
        'fedprox_dirichlet':  ('#dd6b20', 's', 'FedProx + Dirichlet'),
        'pathological':       ('#9b2c2c', 'D', 'Pathological'),
        'mitigation_avg':     ('#2f855a', '^', 'FedAvg + weighted CE'),
        'mitigation_prox':    ('#22543d', 'v', 'FedProx + weighted CE'),
        'focal':              ('#805ad5', 'X', 'FedProx + focal'),
    }

    def family_of(name: str) -> str:
        if 'iid' in name:
            return 'iid'
        if 'pathological' in name:
            return 'pathological'
        if 'weighted' in name and 'fedavg' in name:
            return 'mitigation_avg'
        if 'weighted' in name and 'fedprox' in name:
            return 'mitigation_prox'
        if 'focal' in name:
            return 'focal'
        if 'fedprox' in name:
            return 'fedprox_dirichlet'
        return 'fedavg_dirichlet'

    seen = set()
    points = []
    for cfg, d in data.items():
        if not d:
            continue
        fam = family_of(cfg)
        color, marker, lbl = family_color[fam]
        x, y = d['balanced_accuracy_mean'], d['equity_gap']
        xerr = d['balanced_accuracy_std']
        label = lbl if fam not in seen else None
        seen.add(fam)
        ax.errorbar(x, y, xerr=xerr, marker=marker, color=color,
                    markersize=10, markeredgecolor='white', markeredgewidth=1.2,
                    linewidth=0, elinewidth=1.5, capsize=3, label=label, zorder=3)
        points.append((cfg, x, y, fam))

    # Pareto frontier (low y, high x). Points are Pareto-optimal if no other point
    # has higher x AND lower y.
    pareto = []
    for cfg, x, y, fam in points:
        dominated = any(x2 >= x and y2 <= y and (x2 > x or y2 < y)
                        for cfg2, x2, y2, fam2 in points if cfg2 != cfg)
        if not dominated:
            pareto.append((cfg, x, y))
    pareto = sorted(pareto, key=lambda t: t[1])
    if pareto:
        ax.plot([p[1] for p in pareto], [p[2] for p in pareto],
                color='gray', linestyle='--', linewidth=1.5, alpha=0.7,
                zorder=2, label='Pareto frontier')

    # Annotate the best point
    if pareto:
        best = max(pareto, key=lambda t: t[1])  # rightmost on the frontier
        ax.annotate(best[0].replace('fedprox_dir05_weighted', 'FedProx + weighted CE @ Dir(0.5)'),
                    xy=(best[1], best[2]), xytext=(best[1] - 0.18, best[2] - 0.05),
                    fontsize=9, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.2),
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray'))

    ax.set_xlabel('Balanced accuracy (higher → better)')
    ax.set_ylabel('Equity gap = max F1 − min F1 (lower → fairer)')
    ax.set_title('Accuracy–fairness Pareto frontier')
    ax.legend(loc='lower right', fontsize=8, framealpha=0.95)
    ax.grid(alpha=0.3)
    ax.invert_yaxis()  # lower equity gap is better → top of plot


def per_class_f1_plot(ax, data: dict[str, dict]) -> None:
    """Grouped bar chart of per-class F1: FedAvg / FedProx / Weighted CE / IID baseline."""
    cfgs_to_show = [
        ('fedavg_iid', 'IID baseline', '#000000'),
        ('fedavg_dir05', 'FedAvg @ Dir(0.5)', '#3182ce'),
        ('fedprox_dir05', 'FedProx @ Dir(0.5)', '#dd6b20'),
        ('fedprox_dir05_weighted', 'FedProx + weighted CE', '#22543d'),
    ]
    n_classes = len(CLASS_NAMES)
    n_cfgs = len(cfgs_to_show)
    x = np.arange(n_classes)
    width = 0.18

    for i, (cfg, lbl, color) in enumerate(cfgs_to_show):
        d = data.get(cfg, {})
        if not d:
            continue
        means = d['per_class_f1_mean']
        stds = d['per_class_f1_std']
        offset = (i - (n_cfgs - 1) / 2) * width
        ax.bar(x + offset, means, width, yerr=stds, label=lbl,
               color=color, capsize=2, edgecolor='white', linewidth=0.5)

    # Highlight minority classes
    for j, name in enumerate(CLASS_NAMES):
        if name in MINORITY:
            ax.axvspan(j - 0.5, j + 0.5, alpha=0.07, color='red', zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(' ', '\n', 1) for n in CLASS_NAMES],
                       fontsize=8, rotation=0)
    ax.set_ylabel('F1 score (mean ± std, n=3)')
    ax.set_title('Per-class F1: minority classes (pink) rescued by weighted CE')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3, axis='y')


def main() -> None:
    plt.rcParams.update({'font.size': 10})

    # Load aggregated data
    configs = [
        'fedavg_iid', 'fedavg_dir05', 'fedavg_dir01', 'fedavg_pathological',
        'fedprox_dir05', 'fedprox_dir01', 'fedprox_pathological',
        'fedavg_dir05_weighted', 'fedprox_dir05_weighted', 'fedprox_dir05_focal',
    ]
    data = {c: load_aggregated(c) for c in configs}

    # Pareto plot
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    pareto_plot(ax, data)
    fig.savefig(FIG / 'fig_pareto_clean.pdf', bbox_inches='tight')
    fig.savefig(FIG / 'fig_pareto_clean.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {FIG}/fig_pareto_clean.{{pdf,png}}')

    # Per-class F1 plot
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    per_class_f1_plot(ax, data)
    fig.savefig(FIG / 'fig_per_class_f1_clean.pdf', bbox_inches='tight')
    fig.savefig(FIG / 'fig_per_class_f1_clean.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {FIG}/fig_per_class_f1_clean.{{pdf,png}}')


if __name__ == '__main__':
    main()
