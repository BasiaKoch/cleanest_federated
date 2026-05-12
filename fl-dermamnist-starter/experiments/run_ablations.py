from pathlib import Path
import argparse
import subprocess
import sys
import yaml


def write_temp_config(base_cfg_path, overrides, out_path):
    with open(base_cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    def merge(a, b):
        for k, v in b.items():
            if isinstance(v, dict) and isinstance(a.get(k), dict):
                merge(a[k], v)
            else:
                a[k] = v
    merge(cfg, overrides)
    with open(out_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ablation', required=True, choices=['local_epochs', 'mu', 'alpha', 'clients', 'participation'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--num_rounds_override', type=int, default=None)
    args = parser.parse_args()
    tmp_dir = Path('configs/tmp_ablations')
    tmp_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    if args.ablation == 'local_epochs':
        for e in [1, 5, 10, 20]:
            jobs.append((f'abl_localE{e}', {'federation': {'local_epochs': e}, 'client_objective': {'proximal_mu': 0.01}, 'partition': {'strategy': 'dirichlet', 'alpha': 0.5}}))
    elif args.ablation == 'mu':
        for mu in [0.0, 0.001, 0.01, 0.1, 1.0]:
            jobs.append((f'abl_mu{mu}', {'client_objective': {'proximal_mu': mu}, 'partition': {'strategy': 'dirichlet', 'alpha': 0.5}}))
    elif args.ablation == 'alpha':
        for alpha in [0.05, 0.1, 0.3, 0.5, 1.0, 5.0]:
            jobs.append((f'abl_alpha{alpha}', {'client_objective': {'proximal_mu': 0.0}, 'partition': {'strategy': 'dirichlet', 'alpha': alpha}}))
    elif args.ablation == 'clients':
        for k in [3, 5, 10, 20]:
            jobs.append((f'abl_clients{k}', {'federation': {'num_clients': k}, 'partition': {'strategy': 'dirichlet', 'alpha': 0.5}}))
    elif args.ablation == 'participation':
        for frac in [0.3, 0.5, 0.7, 1.0]:
            jobs.append((f'abl_part{frac}', {'federation': {'fraction_fit': frac}, 'partition': {'strategy': 'dirichlet', 'alpha': 0.5}}))

    for name, override in jobs:
        override.setdefault('misc', {})['save_dir'] = f'results/ablations/{args.ablation}'
        cfg_path = tmp_dir / f'{name}.yaml'
        write_temp_config('configs/base.yaml', override, cfg_path)
        cmd = [sys.executable, 'experiments/run_experiment.py', '--config', str(cfg_path), '--seed', str(args.seed), '--device', args.device]
        if args.num_rounds_override is not None:
            cmd += ['--num_rounds_override', str(args.num_rounds_override)]
        print('Running:', ' '.join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
