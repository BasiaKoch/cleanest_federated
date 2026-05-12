from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon, shapiro, kruskal


SEEDS = [42, 123, 456]


def _read_metrics(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _cohens_d(a, b):
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(diff.mean() / (diff.std(ddof=1) + 1e-12))


def main():
    out = Path('results/analysis/statistics')
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for metrics_path in Path('results').glob('**/global_test_metrics.json'):
        name = metrics_path.parent.name
        seed = next((s for s in SEEDS if name.endswith(f'_s{s}')), None)
        if seed is None:
            continue
        metrics = _read_metrics(metrics_path)
        rows.append({
            'config': name.removesuffix(f'_s{seed}'),
            'seed': seed,
            'balanced_accuracy': metrics.get('balanced_accuracy'),
            'macro_f1': metrics.get('macro_f1'),
            'worst_class_f1': metrics.get('worst_class_f1'),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out / 'seed_results.csv', index=False)
    if df.empty:
        print('No seeded global_test_metrics.json files found.')
        return

    summary = df.groupby('config').agg(['mean', 'std'])
    summary.columns = ['_'.join(c).strip('_') for c in summary.columns]
    summary.to_csv(out / 'results_summary.csv')
    with open(out / 'master_results.tex', 'w', encoding='utf-8') as f:
        f.write(summary.reset_index().to_latex(index=False, float_format='%.3f'))

    comparisons = []
    configs = sorted(df['config'].unique())
    for cfg in configs:
        if not cfg.startswith('fedavg_'):
            continue
        peer = cfg.replace('fedavg_', 'fedprox_', 1)
        if peer not in configs:
            continue
        a = df[df['config'] == cfg].sort_values('seed')
        b = df[df['config'] == peer].sort_values('seed')
        common = sorted(set(a['seed']) & set(b['seed']))
        if len(common) < 2:
            continue
        av = a[a['seed'].isin(common)]['balanced_accuracy'].to_numpy(float)
        bv = b[b['seed'].isin(common)]['balanced_accuracy'].to_numpy(float)
        diff = av - bv
        normal = len(diff) >= 3 and shapiro(diff).pvalue >= 0.05
        stat = ttest_rel(av, bv) if normal else wilcoxon(av, bv)
        comparisons.append({'method_a': cfg, 'method_b': peer, 'metric': 'balanced_accuracy',
                            'p_value': float(stat.pvalue), 'cohens_d': _cohens_d(av, bv)})
    pd.DataFrame(comparisons).to_csv(out / 'pairwise_comparisons.csv', index=False)

    fedavg = df[df['config'].str.startswith('fedavg_dir')]
    groups = [g['balanced_accuracy'].dropna().values for _, g in fedavg.groupby('config') if len(g) > 0]
    anova_rows = []
    if len(groups) >= 2:
        stat = kruskal(*groups)
        anova_rows.append({'test': 'kruskal_fedavg_alpha', 'statistic': float(stat.statistic), 'p_value': float(stat.pvalue)})
    pd.DataFrame(anova_rows).to_csv(out / 'anova_results.csv', index=False)
    with open(out / 'results_summary.txt', 'w', encoding='utf-8') as f:
        f.write(summary.to_string())
        if comparisons:
            f.write('\n\nPairwise comparisons:\n')
            f.write(pd.DataFrame(comparisons).to_string(index=False))


if __name__ == '__main__':
    main()
