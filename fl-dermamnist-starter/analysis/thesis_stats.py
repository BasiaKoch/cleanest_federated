"""Thesis-grade paired statistical comparison: FedProx vs FedAvg.

Reads test metrics from results/thesis/fedavg_E5_s* and fedprox_mu01_E5_s*
(or equivalent at other E values), then computes:
  - Paired Wilcoxon signed-rank test
  - Rank-biserial effect size
  - Mean paired difference + bootstrap CI
  - Per-class F1 differences
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CLASS_NAMES = [
    'actinic_keratoses', 'basal_cell', 'benign_keratosis',
    'dermatofibroma', 'melanoma', 'melanocytic_nevi', 'vascular_lesions',
]


def load_finals_by_seed(stem_prefix: str, results_dir: str = 'results/thesis') -> dict[int, dict]:
    """Return {seed: metrics_dict} for all completed runs matching the prefix."""
    out = {}
    pattern = f'{results_dir}/{stem_prefix}_E*_s*/global_test_metrics.json'
    for f in glob.glob(pattern):
        # parse seed from path
        seed = int(f.split('_s')[-1].split('/')[0])
        with open(f) as fp:
            out[seed] = json.load(fp)
    return out


def rank_biserial(differences: np.ndarray) -> float:
    """Rank-biserial correlation for paired data — robust effect size."""
    diffs = differences[differences != 0]
    if len(diffs) == 0:
        return float('nan')
    ranks = stats.rankdata(np.abs(diffs))
    W_pos = float(np.sum(ranks[diffs > 0]))
    W_neg = float(np.sum(ranks[diffs < 0]))
    return (W_pos - W_neg) / (W_pos + W_neg)


def bootstrap_mean_ci(differences: np.ndarray, n_boot: int = 10000, alpha: float = 0.05) -> tuple:
    """Two-sided percentile bootstrap CI for the mean paired difference."""
    rng = np.random.default_rng(42)
    means = np.array([
        rng.choice(differences, size=len(differences), replace=True).mean()
        for _ in range(n_boot)
    ])
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def compare(fedavg_stem: str, fedprox_stem: str, results_dir: str = 'results/thesis') -> dict:
    fa = load_finals_by_seed(fedavg_stem, results_dir)
    fp = load_finals_by_seed(fedprox_stem, results_dir)
    shared = sorted(set(fa.keys()) & set(fp.keys()))
    if len(shared) < 2:
        return {'error': f'Need ≥2 paired seeds; found {len(shared)} ({shared})'}

    metrics = ['macro_f1', 'balanced_accuracy', 'worst_class_f1', 'accuracy']
    summary = {'paired_seeds': shared, 'n': len(shared), 'fedavg_stem': fedavg_stem, 'fedprox_stem': fedprox_stem}

    for m in metrics:
        fa_v = np.array([fa[s][m] for s in shared])
        fp_v = np.array([fp[s][m] for s in shared])
        diffs = fp_v - fa_v

        # Two-sided Wilcoxon signed-rank
        try:
            w_stat, w_p = stats.wilcoxon(fp_v, fa_v, zero_method='wilcox', alternative='two-sided')
        except ValueError:
            w_stat, w_p = float('nan'), float('nan')

        # One-sided "FedProx > FedAvg" Wilcoxon
        try:
            _, w_p_greater = stats.wilcoxon(fp_v, fa_v, alternative='greater')
        except ValueError:
            w_p_greater = float('nan')

        # Paired t-test (sanity)
        t_stat, t_p = stats.ttest_rel(fp_v, fa_v) if len(shared) > 1 else (float('nan'), float('nan'))

        ci_lo, ci_hi = bootstrap_mean_ci(diffs)

        summary[m] = {
            'fedavg_mean': float(fa_v.mean()),
            'fedavg_std': float(fa_v.std(ddof=1)) if len(fa_v) > 1 else 0.0,
            'fedprox_mean': float(fp_v.mean()),
            'fedprox_std': float(fp_v.std(ddof=1)) if len(fp_v) > 1 else 0.0,
            'mean_diff': float(diffs.mean()),
            'paired_diffs_per_seed': diffs.tolist(),
            'wilcoxon_stat': float(w_stat),
            'wilcoxon_p_two_sided': float(w_p),
            'wilcoxon_p_greater': float(w_p_greater),
            'rank_biserial': rank_biserial(diffs),
            'cohens_d': float(diffs.mean() / diffs.std(ddof=1)) if (len(diffs) > 1 and diffs.std(ddof=1) > 0) else float('nan'),
            'ci95_low': ci_lo,
            'ci95_high': ci_hi,
            't_stat': float(t_stat),
            't_p': float(t_p),
        }

    # Per-class F1 differences
    per_class_diffs = []
    for s in shared:
        fa_pc = fa[s].get('per_class_f1', [])
        fp_pc = fp[s].get('per_class_f1', [])
        if len(fa_pc) == len(fp_pc) == len(CLASS_NAMES):
            per_class_diffs.append([fp_pc[i] - fa_pc[i] for i in range(len(CLASS_NAMES))])
    if per_class_diffs:
        per_class_diffs = np.array(per_class_diffs)
        summary['per_class_f1'] = {
            CLASS_NAMES[i]: {
                'fedavg_mean': float(np.mean([fa[s]['per_class_f1'][i] for s in shared])),
                'fedprox_mean': float(np.mean([fp[s]['per_class_f1'][i] for s in shared])),
                'mean_diff': float(per_class_diffs[:, i].mean()),
            }
            for i in range(len(CLASS_NAMES))
        }

    return summary


def format_report(summary: dict) -> str:
    if 'error' in summary:
        return f"ERROR: {summary['error']}"
    n = summary['n']
    out = []
    out.append(f"\n{'='*70}\nPaired comparison: {summary['fedprox_stem']} vs {summary['fedavg_stem']}\nn = {n} paired seeds: {summary['paired_seeds']}\n{'='*70}")
    for m in ['macro_f1', 'balanced_accuracy', 'worst_class_f1', 'accuracy']:
        d = summary[m]
        sig = '***' if d['wilcoxon_p_two_sided'] < 0.001 else '**' if d['wilcoxon_p_two_sided'] < 0.01 else '*' if d['wilcoxon_p_two_sided'] < 0.05 else 'ns'
        out.append(f"\n[{m}]")
        out.append(f"  FedAvg  : {d['fedavg_mean']:.4f} ± {d['fedavg_std']:.4f}")
        out.append(f"  FedProx : {d['fedprox_mean']:.4f} ± {d['fedprox_std']:.4f}")
        out.append(f"  Δ (FP-FA): {d['mean_diff']:+.4f}   95% bootstrap CI: [{d['ci95_low']:+.4f}, {d['ci95_high']:+.4f}]")
        out.append(f"  Wilcoxon two-sided p = {d['wilcoxon_p_two_sided']:.4f}  {sig}")
        out.append(f"  Wilcoxon greater p = {d['wilcoxon_p_greater']:.4f}")
        out.append(f"  Rank-biserial r = {d['rank_biserial']:+.3f}    Cohen's d = {d['cohens_d']:+.3f}")
        out.append(f"  Paired diffs per seed: {[f'{x:+.4f}' for x in d['paired_diffs_per_seed']]}")
    if 'per_class_f1' in summary:
        out.append('\n[per-class F1 differences]')
        for cname, d in summary['per_class_f1'].items():
            out.append(f"  {cname:22s}  FA={d['fedavg_mean']:.3f}  FP={d['fedprox_mean']:.3f}  Δ={d['mean_diff']:+.3f}")
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fedavg', default='fedavg', help='Stem prefix for FedAvg runs')
    ap.add_argument('--fedprox', default='fedprox_mu01', help='Stem prefix for FedProx runs')
    ap.add_argument('--results-dir', default='results/thesis')
    ap.add_argument('--out', default='results/thesis/analysis/headline_stats.json')
    args = ap.parse_args()

    summary = compare(args.fedavg, args.fedprox, args.results_dir)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'wrote {args.out}')
    print(format_report(summary))


if __name__ == '__main__':
    main()
