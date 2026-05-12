"""One-shot dissertation story plot: heterogeneity ladder, FedAvg-vs-FedProx, mitigation, participation."""
from __future__ import annotations

import json
import glob
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

RESULTS = Path('results')
FIG = RESULTS / 'figures'
FIG.mkdir(parents=True, exist_ok=True)


def load_metrics(pattern: str) -> list[dict]:
    rows = []
    for f in sorted(glob.glob(f'results/{pattern}/global_test_metrics.json')):
        with open(f) as fp:
            rows.append(json.load(fp))
    return rows


def mean_std(rows: list[dict], key: str) -> tuple[float, float]:
    if not rows:
        return float('nan'), float('nan')
    vals = [r[key] for r in rows]
    return float(np.mean(vals)), float(np.std(vals))


def collect(stem: str, key: str) -> tuple[float, float]:
    return mean_std(load_metrics(f'{stem}_E5_s*'), key)


def collect_ablation(prefix: str, key: str) -> dict[str, float]:
    out = {}
    for d in sorted(Path(f'results/ablations/{prefix}').glob('abl_*_E5_s*')):
        with open(d / 'global_test_metrics.json') as f:
            out[d.name] = json.load(f)[key]
    return out


def panel_heterogeneity(ax) -> None:
    """FedAvg bACC across heterogeneity regimes."""
    cfgs = ['fedavg_iid', 'fedavg_dir05', 'fedavg_dir01', 'fedavg_pathological']
    labels = ['IID', 'Dir(α=0.5)', 'Dir(α=0.1)', 'Pathological']
    means, stds = zip(*[collect(c, 'balanced_accuracy') for c in cfgs])
    wmeans, wstds = zip(*[collect(c, 'worst_class_f1') for c in cfgs])

    x = np.arange(len(labels))
    ax.errorbar(x, means, yerr=stds, marker='o', linewidth=2,
                color='#2b6cb0', label='Balanced accuracy', capsize=4)
    ax.errorbar(x, wmeans, yerr=wstds, marker='s', linewidth=2,
                color='#c53030', label='Worst-class F1', capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel('Score (mean ± std, n=3)')
    ax.set_title('(a) FedAvg degrades monotonically with heterogeneity')
    ax.set_ylim(-0.02, 0.7)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)


def panel_fedavg_vs_fedprox(ax) -> None:
    """FedAvg vs FedProx side-by-side, three regimes."""
    regimes = ['dir05', 'dir01', 'pathological']
    labels = ['Dir(0.5)', 'Dir(0.1)', 'Pathological']

    fa_b = [collect(f'fedavg_{r}', 'balanced_accuracy') for r in regimes]
    fp_b = [collect(f'fedprox_{r}', 'balanced_accuracy') for r in regimes]

    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width/2, [m for m, _ in fa_b], width,
           yerr=[s for _, s in fa_b], label='FedAvg', color='#3182ce', capsize=4)
    ax.bar(x + width/2, [m for m, _ in fp_b], width,
           yerr=[s for _, s in fp_b], label='FedProx (μ=0.01)', color='#dd6b20', capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Balanced accuracy (mean ± std)')
    ax.set_title('(b) FedProx does not improve over FedAvg (p > 0.18 all)')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(0, 0.6)
    ax.grid(alpha=0.3, axis='y')


def panel_mitigation(ax) -> None:
    """The headline: weighted CE rescues minority class."""
    cfgs = [
        ('fedavg_dir05', 'FedAvg\n(vanilla)'),
        ('fedprox_dir05', 'FedProx\n(vanilla)'),
        ('fedprox_dir05_focal', 'FedProx\n+focal'),
        ('fedavg_dir05_weighted', 'FedAvg\n+weighted CE'),
        ('fedprox_dir05_weighted', 'FedProx\n+weighted CE'),
    ]
    bACC_means, bACC_stds = zip(*[collect(c, 'balanced_accuracy') for c, _ in cfgs])
    w_means, w_stds = zip(*[collect(c, 'worst_class_f1') for c, _ in cfgs])
    labels = [lbl for _, lbl in cfgs]

    x = np.arange(len(labels))
    width = 0.35
    colors_bACC = ['#3182ce'] * 3 + ['#38a169'] * 2  # green = mitigated
    colors_worst = ['#bee3f8'] * 3 + ['#9ae6b4'] * 2
    bars1 = ax.bar(x - width/2, bACC_means, width, yerr=bACC_stds,
                   color=colors_bACC, capsize=4, label='Balanced acc.')
    bars2 = ax.bar(x + width/2, w_means, width, yerr=w_stds,
                   color=colors_worst, capsize=4, label='Worst-class F1')

    for bar, m in zip(bars1, bACC_means):
        ax.annotate(f'{m:.2f}', xy=(bar.get_x() + bar.get_width()/2, m),
                    ha='center', va='bottom', fontsize=8)
    for bar, m in zip(bars2, w_means):
        if m > 0.01:
            ax.annotate(f'{m:.2f}', xy=(bar.get_x() + bar.get_width()/2, m),
                        ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Score')
    ax.set_title('(c) Class-weighted CE rescues minority class (Dir 0.5)')
    ax.set_ylim(0, 0.7)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=0.3, axis='y')


def panel_participation(ax) -> None:
    """Participation ablation: full participation is the worst case."""
    fracs = [0.3, 0.5, 0.7, 1.0]
    bACC = []
    worst = []
    for f in fracs:
        d = Path(f'results/ablations/participation/abl_part{f}_E5_s42/global_test_metrics.json')
        if d.exists():
            with open(d) as fp:
                m = json.load(fp)
            bACC.append(m['balanced_accuracy'])
            worst.append(m['worst_class_f1'])
        else:
            bACC.append(float('nan'))
            worst.append(float('nan'))

    ax.plot(fracs, bACC, marker='o', linewidth=2, color='#2b6cb0',
            label='Balanced accuracy', markersize=10)
    ax.plot(fracs, worst, marker='s', linewidth=2, color='#c53030',
            label='Worst-class F1', markersize=10)
    ax.set_xlabel('Client participation fraction')
    ax.set_ylabel('Score (n=1 per point)')
    ax.set_title('(d) Full participation is the worst case (FedAvg, Dir 0.5)')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.02, 0.6)
    ax.axvline(1.0, color='gray', linestyle=':', alpha=0.5)
    ax.annotate('Your\ndefault', xy=(1.0, 0.05), xytext=(0.85, 0.15),
                fontsize=8, ha='center',
                arrowprops=dict(arrowstyle='->', alpha=0.5))


def main() -> None:
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
    })

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    panel_heterogeneity(axes[0, 0])
    panel_fedavg_vs_fedprox(axes[0, 1])
    panel_mitigation(axes[1, 0])
    panel_participation(axes[1, 1])
    fig.suptitle('Federated Learning under Class Imbalance — DermaMNIST',
                 fontsize=13, fontweight='bold', y=1.02)

    out_pdf = FIG / 'fig_story.pdf'
    out_png = FIG / 'fig_story.png'
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out_pdf}')
    print(f'wrote {out_png}')


if __name__ == '__main__':
    main()
