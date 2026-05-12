from pathlib import Path
import json
import subprocess
import sys
import pandas as pd


def _experiment_name(config_path: Path, seed: int) -> str:
    return f'{config_path.stem}_E5_s{seed}'


def _summarise_results(configs, seeds, results_root=Path('results')):
    rows = []
    for cfg in configs:
        for seed in seeds:
            metrics_path = results_root / _experiment_name(cfg, seed) / 'global_test_metrics.json'
            if not metrics_path.exists():
                continue
            with open(metrics_path, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
            rows.append({
                'config': cfg.stem,
                'seed': seed,
                'balanced_accuracy': metrics.get('balanced_accuracy'),
                'accuracy': metrics.get('accuracy'),
                'macro_f1': metrics.get('macro_f1'),
                'worst_class_f1': metrics.get('worst_class_f1'),
                'macro_auroc': metrics.get('macro_auroc'),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    summary = df.groupby('config').agg(['mean', 'std'])
    summary.columns = ['_'.join(col).strip('_') for col in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(results_root / 'master_comparison.csv', index=False)
    with open(results_root / 'master_comparison.tex', 'w', encoding='utf-8') as f:
        f.write(summary.to_latex(index=False, float_format='%.3f'))
    return summary


def main():
    configs = sorted(Path('configs').glob('fedavg_*.yaml')) + sorted(Path('configs').glob('fedprox_*.yaml'))
    seeds = [42, 123, 456]
    failures = []
    for cfg in configs:
        for seed in seeds:
            cmd = [sys.executable, 'experiments/run_experiment.py', '--config', str(cfg), '--seed', str(seed)]
            print('Running:', ' '.join(cmd))
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as exc:
                failures.append((str(cfg), seed, exc.returncode))
                print(f'FAILED: {cfg} seed={seed}')
    if failures:
        print('Failures:')
        for f in failures:
            print(f)
    summary = _summarise_results(configs, seeds)
    if not summary.empty:
        print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
