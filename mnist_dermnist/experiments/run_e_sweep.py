"""CLI: optional E sweep — E ∈ {1, 5, 10, 20, 40} × {FedAvg, FedProx} × seeds.

Useful for replicating paper Fig 4 style "FedProx advantage grows with E".
By default uses a smaller seed set (3 seeds) because the matrix is wider.

Example
-------
python -m mnist_dermnist.experiments.run_e_sweep --seeds 42 123 456
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_SEEDS = [42, 123, 456]
DEFAULT_ES = [1, 5, 10, 20, 40]
PAIRS = [("fedavg", 0.0), ("fedprox", 0.1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--local-epochs", type=int, nargs="+", default=DEFAULT_ES)
    ap.add_argument("--num-rounds", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--partition", default="medical_skew_7_clients")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", default="mnist_dermnist/results/e_sweep")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_total = len(args.seeds) * len(args.local_epochs) * len(PAIRS)
    print(f"E sweep: {len(args.seeds)} seeds × {len(args.local_epochs)} E × {len(PAIRS)} algorithms = {n_total} runs")

    submitted, skipped, failed = 0, 0, []
    for E in args.local_epochs:
        for seed in args.seeds:
            for algo, mu in PAIRS:
                stem = f"{algo}_mu{mu}_E{E}_s{seed}"
                test_json = out_dir / f"test_at_best_{stem}.json"
                if test_json.exists():
                    print(f"  [SKIP] {test_json.name}")
                    skipped += 1
                    continue
                cmd = [
                    sys.executable, "-m", "mnist_dermnist.experiments.run_one",
                    "--algorithm", algo, "--mu", str(mu), "--seed", str(seed),
                    "--local-epochs", str(E), "--num-rounds", str(args.num_rounds),
                    "--lr", str(args.lr), "--batch-size", str(args.batch_size),
                    "--partition", args.partition, "--device", args.device,
                    "--out-dir", str(out_dir),
                ]
                print(f"  [RUN] {algo} μ={mu} seed={seed} E={E}")
                if args.dry_run:
                    print("        " + " ".join(cmd))
                    continue
                r = subprocess.run(cmd)
                if r.returncode == 0:
                    submitted += 1
                else:
                    failed.append(stem)
    print(f"\nE sweep done: submitted={submitted}, skipped={skipped}, failed={len(failed)}")
    if failed:
        for s in failed:
            print(f"  FAILED: {s}")
        sys.exit(1)


if __name__ == "__main__":
    main()
