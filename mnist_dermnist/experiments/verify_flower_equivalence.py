"""Verify the Flower-framework runtime produces results equivalent to the
pure-PyTorch FL loop.

Runs both paths on the same (seed, partition, hyperparameter) configuration
and compares the resulting test macro-F1 values. Differences arising from
RNG-iteration order in Ray-managed Flower workers vs the sequential PyTorch
loop are expected to be < 0.005 macro-F1 (within the CUDA non-determinism
noise floor estimated for the headline sweep).

The point is NOT to demand bit-identity, but to demonstrate that the
underlying FedAvg/FedProx algorithm produces equivalent statistical
inference under either orchestration framework.

Usage:
    PYTHONPATH=. python -m mnist_dermnist.experiments.verify_flower_equivalence \\
        --seed 42 --local-epochs 5 --num-rounds 10
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def run_path(runner: str, args, out_dir: Path) -> dict:
    """Run one of the two runtime paths and return the test_at_best JSON."""
    cmd = [
        sys.executable, "-m", runner,
        "--algorithm", args.algorithm,
        "--mu", str(args.mu),
        "--seed", str(args.seed),
        "--local-epochs", str(args.local_epochs),
        "--num-rounds", str(args.num_rounds),
        "--partition", args.partition,
        "--device", "cpu",
        "--npz-path", args.npz_path,
        "--out-dir", str(out_dir),
    ]
    print(f"\n--- Running {runner} ---")
    print("  " + " ".join(cmd[2:]))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("STDERR:", result.stderr[-2000:])
        raise RuntimeError(f"{runner} failed")
    # Find the resulting JSON
    jsons = list(out_dir.glob("test_at_best_*.json"))
    if not jsons:
        raise RuntimeError(f"{runner} produced no test_at_best JSON in {out_dir}")
    return json.load(open(jsons[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--algorithm", choices=["fedavg", "fedprox"], default="fedprox")
    ap.add_argument("--mu", type=float, default=0.01)
    ap.add_argument("--local-epochs", type=int, default=5,
                    help="Keep small for CPU equivalence test")
    ap.add_argument("--num-rounds", type=int, default=10,
                    help="Keep small for CPU equivalence test")
    ap.add_argument("--partition", default="balanced_paired_7_clients")
    ap.add_argument("--npz-path",
                    default="/Users/basiakoch/cleanest_federated/dermamnist_64.npz")
    ap.add_argument("--tolerance", type=float, default=0.005,
                    help="Max macro-F1 difference treated as equivalent")
    args = ap.parse_args()

    print("=" * 70)
    print("FLOWER vs PURE-PYTORCH EQUIVALENCE TEST")
    print("=" * 70)
    print(f"Config: algo={args.algorithm} mu={args.mu} seed={args.seed} "
          f"E={args.local_epochs} R={args.num_rounds} partition={args.partition}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        pt_out = tmp_path / "pure_pytorch"; pt_out.mkdir()
        fl_out = tmp_path / "flower";        fl_out.mkdir()

        pt_result = run_path("mnist_dermnist.experiments.run_one", args, pt_out)
        fl_result = run_path("mnist_dermnist.experiments.run_one_flower", args, fl_out)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    keys = ["macro_f1", "balanced_accuracy", "accuracy", "loss"]
    print(f"{'metric':<22} {'pure_pytorch':>14} {'flower':>14} {'|diff|':>10}")
    print("-" * 62)
    all_pass = True
    for k in keys:
        pt_v = float(pt_result[k]); fl_v = float(fl_result[k])
        d = abs(pt_v - fl_v)
        ok = d <= args.tolerance
        all_pass = all_pass and ok
        mark = "✓" if ok else "✗"
        print(f"{k:<22} {pt_v:>14.6f} {fl_v:>14.6f} {d:>10.6f} {mark}")

    print(f"\nTolerance: {args.tolerance}  ({'PASS' if all_pass else 'FAIL'})")
    print(f"Pure-PyTorch best-val round: {pt_result.get('selected_round')}")
    print(f"Flower      best-val round: {fl_result.get('selected_round')}")
    print(f"\nInterpretation: differences smaller than {args.tolerance} are within "
          f"the estimated CUDA/RNG-order noise floor and confirm algorithmic "
          f"equivalence between the two runtimes.")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
