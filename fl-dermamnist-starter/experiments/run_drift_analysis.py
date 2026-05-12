from __future__ import annotations

from pathlib import Path
import argparse
import sys
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config_loader import load_config
from utils.seed import set_all_seeds
from server.flower_server import run_simulation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/fedavg_dir01.yaml')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_rounds_override', type=int, default=50)
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg.setdefault('misc', {})['seed'] = args.seed
    cfg.setdefault('misc', {})['device'] = args.device
    cfg.setdefault('misc', {})['save_dir'] = 'results/analysis/drift'
    cfg.setdefault('federation', {})['num_rounds'] = args.num_rounds_override
    cfg['track_drift'] = True
    set_all_seeds(args.seed)
    run_simulation(cfg)
    save_dir = Path(cfg['misc']['save_dir'])
    drift_files = sorted(save_dir.glob('**/drift_history.csv'))
    if not drift_files:
        drift_files = [save_dir / 'drift_history.csv']
    for path in drift_files:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        fig, ax = plt.subplots(figsize=(8, 4))
        for cid, group in df.groupby('client_id'):
            ax.plot(group['round'], group['l2_distance'], label=str(cid), alpha=0.8)
        ax.set_xlabel('Round')
        ax.set_ylabel('L2 drift')
        ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc='upper left')
        fig.tight_layout()
        fig.savefig(path.with_name('drift_l2_by_client.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved drift analysis from {path}')


if __name__ == '__main__':
    main()
