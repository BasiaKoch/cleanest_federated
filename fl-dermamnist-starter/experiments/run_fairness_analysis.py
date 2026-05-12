from __future__ import annotations

from pathlib import Path
import json
import sys
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from metrics.fairness import equity_gap, minority_macro_f1, worst_client_gap


def _load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _pareto(df):
    keep = []
    for i, row in df.iterrows():
        dominated = ((df['balanced_accuracy'] >= row['balanced_accuracy']) & (df['equity_gap'] <= row['equity_gap']) &
                     ((df['balanced_accuracy'] > row['balanced_accuracy']) | (df['equity_gap'] < row['equity_gap']))).any()
        keep.append(not dominated)
    return keep


def main():
    out = Path('results/analysis/fairness')
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for metrics_path in Path('results').glob('**/global_test_metrics.json'):
        run_dir = metrics_path.parent
        per_client_path = run_dir / 'per_client_metrics.csv'
        metrics = _load_json(metrics_path)
        per_class_f1 = metrics.get('per_class_f1', [])
        row = {
            'experiment': run_dir.name,
            'balanced_accuracy': metrics.get('balanced_accuracy'),
            'macro_f1': metrics.get('macro_f1'),
            'worst_f1': metrics.get('worst_class_f1'),
            'minority_f1': minority_macro_f1(per_class_f1) if per_class_f1 else None,
            'equity_gap': equity_gap(per_class_f1) if per_class_f1 else None,
            'melanoma_recall': metrics.get('per_class_recall', [None] * 7)[4],
        }
        if per_client_path.exists():
            pc = pd.read_csv(per_client_path)
            row['client_gap'] = worst_client_gap(pc['balanced_accuracy'])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / 'fairness_summary.csv', index=False)
    if not df.empty:
        df['pareto_optimal'] = False
        valid = df.dropna(subset=['balanced_accuracy', 'equity_gap'])
        df.loc[valid.index, 'pareto_optimal'] = _pareto(valid)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(df['balanced_accuracy'], df['equity_gap'], c=df['pareto_optimal'].map({True: 'tab:green', False: 'tab:blue'}))
        for _, row in df.iterrows():
            ax.annotate(row['experiment'], (row['balanced_accuracy'], row['equity_gap']), fontsize=7)
        ax.set_xlabel('Balanced accuracy')
        ax.set_ylabel('Equity gap')
        fig.tight_layout()
        fig.savefig(out / 'pareto_frontier.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
