"""CLI: full headline sweep — 10 paired seeds × {FedAvg, FedProx(μ=0.1)} at E=20.

Per the spec:
  - algorithms: fedavg, fedprox
  - seeds: 42, 123, 456, 789, 999, 2024, 31337, 8675309, 161803, 271828
  - local_epochs: 20
  - partition: medical_skew_7_clients
  - model: DermMNISTCNN
  - loss: cross-entropy (no class weighting)
  - optimizer: SGD lr=0.01 momentum=0.9
  - batch_size: 32 (capped per-client by client_size)
  - rounds: 150

Runs are paired by `seed` — both algorithms share initial weights, partition,
client sampling, and minibatch order for any given seed. Skips runs whose
output JSON already exists, so it's safe to resume.

Example
-------
python -m mnist_dermnist.experiments.run_headline_sweep
python -m mnist_dermnist.experiments.run_headline_sweep --device cuda --out-dir mnist_dermnist/results/headline
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


HEADLINE_SEEDS = [42, 123, 456, 789, 999, 2024, 31337, 8675309, 161803, 271828]

PAIRS = [
    ("fedavg",  0.0),
    ("fedprox", 0.1),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=HEADLINE_SEEDS)
    ap.add_argument("--local-epochs", type=int, default=20)
    ap.add_argument("--num-rounds", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--partition", default="medical_skew_7_clients")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", default="mnist_dermnist/results/headline")
    ap.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    ap.add_argument("--skip-existing", action="store_true", default=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_total = len(args.seeds) * len(PAIRS)
    print(f"Headline sweep: {len(args.seeds)} seeds × {len(PAIRS)} algorithms = {n_total} runs")
    print(f"  seeds: {args.seeds}")
    print(f"  algorithms: {[(a, m) for a, m in PAIRS]}")
    print(f"  E={args.local_epochs}, R={args.num_rounds}, lr={args.lr}, batch={args.batch_size}")
    print(f"  partition: {args.partition}")
    print(f"  out_dir: {out_dir.resolve()}")
    print()

    submitted = 0
    skipped = 0
    failed: list[str] = []
    for seed in args.seeds:
        for algo, mu in PAIRS:
            stem = f"{algo}_mu{mu}_E{args.local_epochs}_s{seed}"
            test_json = out_dir / f"test_at_best_{stem}.json"
            if args.skip_existing and test_json.exists():
                print(f"  [SKIP existing] {test_json.name}")
                skipped += 1
                continue
            cmd = [
                sys.executable, "-m", "mnist_dermnist.experiments.run_one",
                "--algorithm", algo,
                "--mu", str(mu),
                "--seed", str(seed),
                "--local-epochs", str(args.local_epochs),
                "--num-rounds", str(args.num_rounds),
                "--lr", str(args.lr),
                "--batch-size", str(args.batch_size),
                "--partition", args.partition,
                "--device", args.device,
                "--out-dir", str(out_dir),
            ]
            print(f"  [RUN] {algo} μ={mu} seed={seed}")
            if args.dry_run:
                print("        " + " ".join(cmd))
                continue
            r = subprocess.run(cmd)
            if r.returncode == 0:
                submitted += 1
            else:
                failed.append(stem)
                print(f"        FAILED with code {r.returncode}")

    print()
    print(f"Sweep finished: submitted={submitted}, skipped={skipped}, failed={len(failed)}")
    if failed:
        print("  Failed runs:")
        for s in failed:
            print(f"    - {s}")
        sys.exit(1)


if __name__ == "__main__":
    main()
