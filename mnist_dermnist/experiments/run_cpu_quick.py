"""CPU-friendly back-to-back FedAvg vs FedProx run.

Self-contained: load data, partition, run both algorithms with the SAME seed,
print a side-by-side comparison + delta. Saves both runs' CSV/JSON to
`mnist_dermnist/results/cpu_quick/`.

The μ value is set to the literature-default of 0.1 (Li et al. 2020). Override
with `--mu` if you want to spot-check another value while the HPC μ sweep runs.

Default settings (chosen for ~30 min CPU runtime, single seed):
  - partition: medical_skew_7_clients
  - num_rounds: 20         (vs 150 on HPC)
  - local_epochs: 10        (vs 20 on HPC — still in the drift-prone regime)
  - lr: 0.01
  - batch_size: 32
  - seed: 42

Examples
--------
# Defaults: FedAvg vs FedProx(μ=0.1), 20 rounds, E=10
PYTHONPATH=. python -m mnist_dermnist.experiments.run_cpu_quick

# Match HPC headline E
PYTHONPATH=. python -m mnist_dermnist.experiments.run_cpu_quick --local-epochs 20

# Try a different μ
PYTHONPATH=. python -m mnist_dermnist.experiments.run_cpu_quick --mu 1.0
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from mnist_dermnist.data.load import load_dermmnist
from mnist_dermnist.data.partition import medical_skew_7_clients
from mnist_dermnist.fl.server_loop import FLConfig, run_fl, save_run_outputs
from mnist_dermnist.models import DermMNISTCNN


def _build_cfg(*, algo: str, mu: float, args) -> FLConfig:
    return FLConfig(
        seed=args.seed,
        algorithm=algo,
        mu=mu,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        fraction_fit=1.0,
        lr=args.lr,
        momentum=0.9,
        weight_decay=0.0,
        batch_size=args.batch_size,
        num_classes=7,
        device="cpu",
    )


def _run_and_summarize(cfg: FLConfig, *, train, val_loader, test_loader, partitions, out_dir: Path) -> dict:
    t0 = time.time()
    result = run_fl(
        cfg,
        model_builder=DermMNISTCNN,
        train_dataset=train,
        val_loader=val_loader,
        test_loader=test_loader,
        client_indices=partitions,
    )
    elapsed = time.time() - t0
    save_run_outputs(result, out_dir)
    tm = result["test_metrics"]
    return {
        "algo": cfg.algorithm,
        "mu": cfg.mu,
        "elapsed_s": elapsed,
        "best_round": tm["selected_round"],
        "best_val_macro_f1": tm["best_val_macro_f1"],
        "test_macro_f1": tm["macro_f1"],
        "test_balanced_accuracy": tm["balanced_accuracy"],
        "test_accuracy": tm["accuracy"],
        "test_loss": tm["loss"],
        "test_per_class_f1": tm["per_class_f1"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mu", type=float, default=0.1,
                    help="FedProx proximal coefficient (default 0.1 per Li et al. 2020)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-rounds", type=int, default=20)
    ap.add_argument("--local-epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--image-size", type=int, default=28)
    ap.add_argument("--npz-path", default="/Users/basiakoch/cleanest_federated/dermamnist_64.npz")
    ap.add_argument("--out-dir", default="mnist_dermnist/results/cpu_quick")
    args = ap.parse_args()

    print("=" * 72)
    print("CPU quick FedAvg vs FedProx — paired run, one seed")
    print("=" * 72)
    print(f"  seed={args.seed}  rounds={args.num_rounds}  local_epochs={args.local_epochs}")
    print(f"  lr={args.lr}  batch_size={args.batch_size}  partition=medical_skew_7_clients")
    print(f"  μ (FedProx) = {args.mu}")
    print()

    # ---- data + partition (paired across both runs via shared seed) ----
    print("Loading DermMNIST and partitioning ...", flush=True)
    train, val, test = load_dermmnist(args.npz_path, image_size=args.image_size)
    val_loader = DataLoader(val, batch_size=128, shuffle=False, num_workers=0)
    test_loader = DataLoader(test, batch_size=128, shuffle=False, num_workers=0)
    partitions, _ = medical_skew_7_clients(train.labels, seed=args.seed)
    print(f"  train={len(train)}  val={len(val)}  test={len(test)}")
    print(f"  client sizes: {[len(c) for c in partitions]}")
    print(f"  size ratio max/min: {max(len(c) for c in partitions) / min(len(c) for c in partitions):.1f}x")
    print()

    out_dir = Path(args.out_dir)

    # ---- run FedAvg ----
    print("→ Running FedAvg (μ=0) ...", flush=True)
    fa = _run_and_summarize(
        _build_cfg(algo="fedavg", mu=0.0, args=args),
        train=train, val_loader=val_loader, test_loader=test_loader,
        partitions=partitions, out_dir=out_dir,
    )
    print(f"  elapsed: {fa['elapsed_s']/60:.1f} min", flush=True)
    print(f"  test macro-F1   = {fa['test_macro_f1']:.4f}  (best @ round {fa['best_round']})")
    print(f"  test bACC       = {fa['test_balanced_accuracy']:.4f}")
    print(f"  test accuracy   = {fa['test_accuracy']:.4f}")
    print()

    # ---- run FedProx ----
    print(f"→ Running FedProx (μ={args.mu}) ...", flush=True)
    fp = _run_and_summarize(
        _build_cfg(algo="fedprox", mu=args.mu, args=args),
        train=train, val_loader=val_loader, test_loader=test_loader,
        partitions=partitions, out_dir=out_dir,
    )
    print(f"  elapsed: {fp['elapsed_s']/60:.1f} min", flush=True)
    print(f"  test macro-F1   = {fp['test_macro_f1']:.4f}  (best @ round {fp['best_round']})")
    print(f"  test bACC       = {fp['test_balanced_accuracy']:.4f}")
    print(f"  test accuracy   = {fp['test_accuracy']:.4f}")
    print()

    # ---- comparison ----
    print("=" * 72)
    print("Head-to-head (paired by seed)")
    print("=" * 72)
    delta_f1 = fp["test_macro_f1"] - fa["test_macro_f1"]
    delta_bacc = fp["test_balanced_accuracy"] - fa["test_balanced_accuracy"]
    print(f"  test macro-F1       : FedAvg {fa['test_macro_f1']:.4f}  vs  FedProx {fp['test_macro_f1']:.4f}   Δ = {delta_f1:+.4f}")
    print(f"  test balanced acc   : FedAvg {fa['test_balanced_accuracy']:.4f}  vs  FedProx {fp['test_balanced_accuracy']:.4f}   Δ = {delta_bacc:+.4f}")
    print(f"  test accuracy       : FedAvg {fa['test_accuracy']:.4f}  vs  FedProx {fp['test_accuracy']:.4f}   Δ = {fp['test_accuracy']-fa['test_accuracy']:+.4f}")
    print()

    classes = ("actinic", "basal", "benign_k", "dermato", "melanoma", "mel_nevi", "vascular")
    print("  per-class test F1 (FedAvg → FedProx):")
    for c, name in enumerate(classes):
        d = fp["test_per_class_f1"][c] - fa["test_per_class_f1"][c]
        print(f"    {name:>10s}: {fa['test_per_class_f1'][c]:.3f} → {fp['test_per_class_f1'][c]:.3f}  Δ = {d:+.3f}")
    print()

    # Honest verdict
    print("Verdict (single seed — descriptive only, not a statistical claim):")
    if delta_f1 > 0.005:
        print(f"  FedProx beats FedAvg on macro-F1 by {delta_f1:+.4f}. Consistent with hypothesis.")
    elif delta_f1 < -0.005:
        print(f"  FedAvg beats FedProx on macro-F1 by {-delta_f1:+.4f}. μ={args.mu} may be sub-optimal.")
    else:
        print(f"  Difference within noise (|Δ| ≤ 0.005). Try a different μ or rely on HPC sweep.")
    print()
    print(f"Outputs written to: {out_dir.resolve()}")

    # Save the comparison as a small JSON for easy reading later
    comparison = {
        "seed": args.seed, "num_rounds": args.num_rounds, "local_epochs": args.local_epochs,
        "mu": args.mu, "lr": args.lr, "batch_size": args.batch_size,
        "fedavg": fa, "fedprox": fp,
        "delta_macro_f1": delta_f1,
        "delta_balanced_accuracy": delta_bacc,
    }
    with open(out_dir / f"comparison_seed{args.seed}_mu{args.mu}.json", "w") as f:
        json.dump(comparison, f, indent=2)


if __name__ == "__main__":
    main()
