from __future__ import annotations

from pathlib import Path
import json
import sys
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.download import load_dermamnist, get_dataset_info
from data.partition import iid_partition, dirichlet_partition, pathological_partition, get_all_client_distributions
from data.visualise import plot_global_class_distribution, plot_partition_comparison


def _save_both(fig, stem: Path):
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix('.png'), dpi=300, bbox_inches='tight')
    fig.savefig(stem.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def _metric(path, key):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f).get(key)


def _write_table(df, stem):
    stem.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(stem.with_suffix('.csv'), index=False)
    with open(stem.with_suffix('.tex'), 'w', encoding='utf-8') as f:
        f.write(df.to_latex(index=False, float_format='%.3f'))


def main():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.size': 11, 'axes.labelsize': 11, 'xtick.labelsize': 9, 'ytick.labelsize': 9})
    out = Path('results/figures')
    table_out = Path('results/tables')
    out.mkdir(parents=True, exist_ok=True)
    table_out.mkdir(parents=True, exist_ok=True)

    train, _, _ = load_dermamnist(size=28)
    names = get_dataset_info('dermamnist')['class_names']
    fig = plot_global_class_distribution(train, names, None, figsize=(3.5, 3.0))
    _save_both(fig, out / 'fig_class_distribution')

    parts = {
        'IID': iid_partition(train, 10, seed=42),
        'Dir(1.0)': dirichlet_partition(train, 10, 1.0, seed=42),
        'Dir(0.5)': dirichlet_partition(train, 10, 0.5, seed=42),
        'Dir(0.3)': dirichlet_partition(train, 10, 0.3, seed=42),
        'Dir(0.1)': dirichlet_partition(train, 10, 0.1, seed=42),
        'Pathological': pathological_partition(train, 10, 2, seed=42),
    }
    fig = plot_partition_comparison({k: get_all_client_distributions(train, v) for k, v in parts.items()}, names, None, figsize=(7.2, 4.8))
    _save_both(fig, out / 'fig_partition_comparison')

    rows = []
    for metrics_path in Path('results').glob('**/global_test_metrics.json'):
        with open(metrics_path, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
        rows.append({
            'experiment': metrics_path.parent.name,
            'balanced_accuracy': metrics.get('balanced_accuracy'),
            'macro_f1': metrics.get('macro_f1'),
            'worst_class_f1': metrics.get('worst_class_f1'),
            'per_class_f1': metrics.get('per_class_f1'),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        table_df = df.drop(columns=['per_class_f1'])
        _write_table(table_df, table_out / 'table_core_comparison')
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        df.sort_values('balanced_accuracy').tail(20).plot.barh(x='experiment', y='balanced_accuracy', ax=ax, legend=False)
        ax.set_xlabel('Balanced accuracy')
        fig.tight_layout()
        _save_both(fig, out / 'fig_strategy_comparison')

        best = df.sort_values('balanced_accuracy').dropna(subset=['per_class_f1']).tail(3)
        if not best.empty:
            fig, ax = plt.subplots(figsize=(7.2, 4.0))
            x = range(len(names))
            width = 0.25
            for offset, (_, row) in enumerate(best.iterrows()):
                ax.bar([i + (offset - 1) * width for i in x], row['per_class_f1'], width=width, label=row['experiment'])
            ax.set_xticks(list(x))
            ax.set_xticklabels(names, rotation=35, ha='right')
            ax.set_ylabel('F1')
            ax.set_ylim(0, 1)
            ax.legend(fontsize=8)
            fig.tight_layout()
            _save_both(fig, out / 'fig_per_class_f1')

        mitigation = table_df[table_df['experiment'].str.contains('weighted|focal|sampler|logit|fedavg_dir05|fedprox_dir05', case=False, na=False)]
        if not mitigation.empty:
            _write_table(mitigation, table_out / 'table_mitigation')
            fig, ax = plt.subplots(figsize=(7.2, 4.0))
            mitigation.plot.bar(x='experiment', y=['balanced_accuracy', 'worst_class_f1'], ax=ax)
            ax.set_ylabel('Metric')
            ax.set_ylim(0, 1)
            fig.tight_layout()
            _save_both(fig, out / 'fig_mitigation_comparison')

        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        pareto_df = table_df.dropna(subset=['balanced_accuracy', 'worst_class_f1']).copy()
        if not pareto_df.empty:
            pareto_df['equity_gap'] = 1.0 - pareto_df['worst_class_f1']
            ax.scatter(pareto_df['balanced_accuracy'], pareto_df['equity_gap'])
            for _, row in pareto_df.iterrows():
                ax.annotate(row['experiment'], (row['balanced_accuracy'], row['equity_gap']), fontsize=7)
            ax.set_xlabel('Balanced accuracy')
            ax.set_ylabel('Equity gap')
            fig.tight_layout()
            _save_both(fig, out / 'fig_pareto')

    histories = list(Path('results').glob('**/per_class_history.csv'))
    if histories:
        hist = pd.read_csv(histories[0])
        f1_cols = [c for c in hist.columns if c.startswith('f1_')]
        if f1_cols:
            fig, ax = plt.subplots(figsize=(3.5, 3.5))
            hist.plot(x='round', y=f1_cols, ax=ax)
            ax.set_ylabel('F1')
            ax.set_ylim(0, 1)
            ax.legend(fontsize=6, bbox_to_anchor=(1.02, 1), loc='upper left')
            fig.tight_layout()
            _save_both(fig, out / 'fig_per_class_convergence')

    per_client_files = list(Path('results').glob('**/per_client_metrics.csv'))
    if per_client_files:
        rows = []
        for path in per_client_files:
            pc = pd.read_csv(path)
            for _, row in pc.iterrows():
                rows.append({'experiment': path.parent.name, 'num_test_samples': row.get('num_test_samples'), 'balanced_accuracy': row.get('balanced_accuracy')})
        pcdf = pd.DataFrame(rows)
        if not pcdf.empty:
            fig, ax = plt.subplots(figsize=(3.5, 3.5))
            ax.scatter(pcdf['num_test_samples'], pcdf['balanced_accuracy'])
            ax.set_xlabel('Client test samples')
            ax.set_ylabel('Client bACC')
            fig.tight_layout()
            _save_both(fig, out / 'fig_client_scatter')

    central = Path('results/centralised/dermamnist/comparison/summary.csv')
    if central.exists():
        _write_table(pd.read_csv(central), table_out / 'table_centralised')

    ablations = list(Path('results/ablations').glob('**/global_test_metrics.json'))
    if ablations:
        adf = pd.DataFrame([{'experiment': p.parent.name, 'balanced_accuracy': _metric(p, 'balanced_accuracy'),
                             'worst_class_f1': _metric(p, 'worst_class_f1')} for p in ablations])
        _write_table(adf, table_out / 'table_ablation_summary')
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        adf.plot.bar(x='experiment', y='balanced_accuracy', ax=ax, legend=False)
        ax.set_ylabel('Balanced accuracy')
        fig.tight_layout()
        _save_both(fig, out / 'fig_ablation_panel')


if __name__ == '__main__':
    main()
