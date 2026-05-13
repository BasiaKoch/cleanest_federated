"""CPU-friendly back-to-back FedAvg vs FedProx via Flower simulation.

Mirror of `run_cpu_quick.py` but uses Flower (fl.simulation.start_simulation)
for the orchestration instead of a hand-rolled PyTorch loop. Same components:
  - DermMNISTCNN (GroupNorm)
  - medical_skew_7_clients partition
  - per-(seed, round, cid) DataLoader generators
  - best-val macro-F1 checkpoint selection
  - test-set evaluation only at the selected checkpoint

The two implementations should agree closely; use this as a Flower-side
sanity-check of the pure-PyTorch result.

Examples
--------
PYTHONPATH=. python -m mnist_dermnist.experiments.run_cpu_flower
PYTHONPATH=. python -m mnist_dermnist.experiments.run_cpu_flower --local-epochs 20 --num-rounds 30
PYTHONPATH=. python -m mnist_dermnist.experiments.run_cpu_flower --mu 1.0
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List

import flwr as fl
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, Subset

from mnist_dermnist.data.load import load_dermmnist
from mnist_dermnist.data.partition import (
    medical_skew_7_clients,
    balanced_specialist_7_clients,
    balanced_paired_7_clients,
    simple_pathological_3_clients,
    quantity_skew_improved,
)
from mnist_dermnist.fl_flower.client import FlClient, numpy_to_state_dict
from mnist_dermnist.models import DermMNISTCNN


# ----- centralized evaluation (sklearn for macro-F1 / balanced accuracy) ----

CLASS_NAMES = ("actinic", "basal", "benign_k", "dermato", "melanoma", "mel_nevi", "vascular")


@torch.no_grad()
def evaluate_global(parameters: List[np.ndarray], loader: DataLoader, dropout: float = 0.2,
                    device: str = "cpu") -> Dict:
    dev = torch.device(device)
    model = DermMNISTCNN(num_classes=7, dropout=dropout).to(dev)
    numpy_to_state_dict(model, parameters)
    model.eval()
    crit = torch.nn.CrossEntropyLoss()
    total_loss, total = 0.0, 0
    all_preds, all_targets = [], []
    for x, y in loader:
        x = x.to(dev)
        y = y.to(dev).view(-1).long()
        logits = model(x)
        total_loss += float(crit(logits, y).item()) * y.size(0)
        total += y.size(0)
        all_preds.extend(logits.argmax(1).cpu().numpy().tolist())
        all_targets.extend(y.cpu().numpy().tolist())
    labels = list(range(7))
    return {
        "loss": total_loss / max(total, 1),
        "accuracy": float(np.mean(np.array(all_preds) == np.array(all_targets))),
        "balanced_accuracy": float(balanced_accuracy_score(all_targets, all_preds)),
        "macro_f1": float(f1_score(all_targets, all_preds, average="macro", labels=labels, zero_division=0)),
        "per_class_f1": f1_score(all_targets, all_preds, average=None, labels=labels, zero_division=0).tolist(),
    }


# ----- per-run driver ------------------------------------------------------

def run_one(*, algorithm: str, mu: float, seed: int,
            num_rounds: int, local_epochs: int, lr: float, batch_size: int,
            train, val_loader, test_loader, partitions, device: str = "cpu") -> Dict:
    """Run one FL experiment via Flower simulation; return summary dict."""
    num_clients = len(partitions)

    # Per-client DataLoaders — generators seeded by (seed, cid). For per-round
    # variation we'd reseed each round, but Flower's simulation calls
    # client_fn fresh each round, so a per-cid seed suffices (and matches
    # the spec's "minibatch order where possible" clause).
    train_loaders = []
    for cid, idxs in enumerate(partitions):
        gen = torch.Generator().manual_seed(seed * 10000 + cid)
        train_loaders.append(DataLoader(
            Subset(train, list(idxs)),
            batch_size=min(batch_size, max(1, len(idxs))),
            shuffle=True,
            generator=gen,
            num_workers=0,
        ))

    def client_fn(cid: str):
        cid_int = int(cid)
        return FlClient(
            cid=cid_int,
            train_loader=train_loaders[cid_int],
            num_local_epochs=local_epochs,
            lr=lr,
            momentum=0.9,
            proximal_mu=mu,
            device=device,
        ).to_client()

    # ---- best-val checkpoint tracking via mutable closure state ----
    best = {"val_macro_f1": -1.0, "params": None, "round": -1}

    def fit_metrics_agg(metrics_list):
        total = sum(n for n, _ in metrics_list) or 1
        return {
            "train_loss_weighted": sum(n * m.get("train_loss", 0.0) for n, m in metrics_list) / total
        }

    def evaluate_fn(server_round, parameters, _config):
        m = evaluate_global(parameters, val_loader, device=device)
        # Update best-val checkpoint (selected by val macro-F1)
        if m["macro_f1"] > best["val_macro_f1"]:
            best["val_macro_f1"] = m["macro_f1"]
            best["params"] = [p.copy() for p in parameters]
            best["round"] = int(server_round)
        return m["loss"], {
            "val_loss": m["loss"],
            "val_accuracy": m["accuracy"],
            "val_balanced_accuracy": m["balanced_accuracy"],
            "val_macro_f1": m["macro_f1"],
        }

    # Seed everything we control (model init inside FlClient uses torch RNG)
    torch.manual_seed(seed)
    np.random.seed(seed)

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=0,
        min_available_clients=num_clients,
        evaluate_fn=evaluate_fn,
        fit_metrics_aggregation_fn=fit_metrics_agg,
    )

    # Suppress Flower's banner / DEBUG noise on stdout if you want a quieter run:
    os.environ.setdefault("RAY_DEDUP_LOGS", "1")

    t0 = time.time()
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 1.0 if device == "cuda" else 0.0},
    )
    elapsed = time.time() - t0

    # ---- final test eval at best-val checkpoint ----
    if best["params"] is None:
        # Shouldn't happen, but fall back to last-round params
        raise RuntimeError("No best checkpoint captured — was validation evaluated at all?")
    test_metrics = evaluate_global(best["params"], test_loader, device=device)

    # ---- flatten Flower history into a per-round DataFrame ----
    rows: Dict[int, Dict] = {}
    for k, vals in history.metrics_centralized.items():
        for rnd, v in vals:
            rows.setdefault(int(rnd), {"round": int(rnd)})[k] = float(v)
    for rnd, v in history.losses_centralized:
        rows.setdefault(int(rnd), {"round": int(rnd)})["val_loss_flower"] = float(v)
    if hasattr(history, "metrics_distributed_fit"):
        for k, vals in history.metrics_distributed_fit.items():
            for rnd, v in vals:
                rows.setdefault(int(rnd), {"round": int(rnd)})[k] = float(v)
    history_df = pd.DataFrame([rows[k] for k in sorted(rows)])
    history_df["algorithm"] = algorithm
    history_df["mu"] = mu
    history_df["seed"] = seed
    history_df["local_epochs"] = local_epochs

    return {
        "algorithm": algorithm, "mu": mu, "seed": seed,
        "num_rounds": num_rounds, "local_epochs": local_epochs,
        "elapsed_s": elapsed,
        "best_round": best["round"],
        "best_val_macro_f1": best["val_macro_f1"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
        "test_accuracy": test_metrics["accuracy"],
        "test_loss": test_metrics["loss"],
        "test_per_class_f1": test_metrics["per_class_f1"],
        "history_df": history_df,
        "history": history,
    }


# ----- main ----------------------------------------------------------------

def _load_prior_fedavg(out_dir: Path, seed: int) -> dict | None:
    """Find a previously-completed FedAvg result in `out_dir` for this seed.

    Looks at every comparison_seed{seed}_mu*.json — the FedAvg section is the
    same across all of them (μ=0). Returns the fedavg dict or None.
    """
    candidates = sorted(out_dir.glob(f"flower_comparison_seed{seed}_mu*.json"))
    for c in candidates:
        try:
            with open(c) as f:
                blob = json.load(f)
            if "fedavg" in blob and blob["fedavg"].get("seed") == seed:
                return blob["fedavg"], c
        except Exception:
            continue
    return None, None


def main():
    ap = argparse.ArgumentParser()
    # Lowered from 0.1 → 0.01 per evidence that 0.1 was too strong on this partition
    ap.add_argument("--mu", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-rounds", type=int, default=20)
    ap.add_argument("--local-epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--image-size", type=int, default=28)
    ap.add_argument("--npz-path", default="/Users/basiakoch/cleanest_federated/dermamnist_64.npz")
    ap.add_argument("--out-dir", default="mnist_dermnist/results/cpu_flower")
    ap.add_argument("--skip-fedavg", action="store_true",
                    help="Don't re-run FedAvg; reuse a previous result from out-dir (any μ).")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                    help="Device to train on. 'cuda' if a GPU is available.")
    ap.add_argument("--partition",
                    choices=["medical_skew_7_clients", "balanced_specialist_7_clients",
                             "balanced_paired_7_clients",
                             "simple_pathological_3_clients", "quantity_skew_improved"],
                    default="balanced_paired_7_clients",
                    help="Default: balanced_paired_7_clients (every class held by ≥2 clients).")
    args = ap.parse_args()

    print("=" * 72)
    print("CPU Flower FedAvg vs FedProx — paired single-seed run")
    print("=" * 72)
    print(f"  seed={args.seed}  rounds={args.num_rounds}  E={args.local_epochs}")
    print(f"  lr={args.lr}  batch={args.batch_size}  μ={args.mu}")
    print(f"  framework: Flower simulation  device: cpu  num_workers=0")
    print(f"  partition: {args.partition}")
    print()

    train, val, test = load_dermmnist(args.npz_path, image_size=args.image_size)
    val_loader = DataLoader(val, batch_size=128, shuffle=False, num_workers=0)
    test_loader = DataLoader(test, batch_size=128, shuffle=False, num_workers=0)
    partitioners = {
        "medical_skew_7_clients": medical_skew_7_clients,
        "balanced_specialist_7_clients": balanced_specialist_7_clients,
        "balanced_paired_7_clients": balanced_paired_7_clients,
        "simple_pathological_3_clients": simple_pathological_3_clients,
        "quantity_skew_improved": quantity_skew_improved,
    }
    partitions, _ = partitioners[args.partition](train.labels, seed=args.seed)
    print(f"  train={len(train)}  val={len(val)}  test={len(test)}")
    print(f"  client sizes: {[len(c) for c in partitions]}")
    print()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    fa = None
    if args.skip_fedavg:
        fa, src = _load_prior_fedavg(out_dir, args.seed)
        if fa is None:
            raise SystemExit(
                f"--skip-fedavg requested but no prior FedAvg result found for seed={args.seed} "
                f"in {out_dir}. Run without --skip-fedavg first, or remove the flag."
            )
        print(f"→ Reusing prior FedAvg result from {src.name}", flush=True)
        print(f"  prior FedAvg: test_macro_F1 = {fa['test_macro_f1']:.4f}  "
              f"(rounds={fa.get('num_rounds','?')}, E={fa.get('local_epochs','?')})")
        # Sanity check that the prior FedAvg used the same (rounds, E); otherwise
        # the comparison is unfair.
        if fa.get("num_rounds") != args.num_rounds or fa.get("local_epochs") != args.local_epochs:
            print(f"  WARNING: prior FedAvg config differs from this run's "
                  f"(rounds {fa.get('num_rounds')} vs {args.num_rounds}, "
                  f"E {fa.get('local_epochs')} vs {args.local_epochs}). "
                  f"Comparison will not be paired-fair.")
        print()
    else:
        print("→ Running FedAvg (μ=0.0) via Flower ...", flush=True)
        fa = run_one(algorithm="fedavg", mu=0.0, seed=args.seed,
                     num_rounds=args.num_rounds, local_epochs=args.local_epochs,
                     lr=args.lr, batch_size=args.batch_size,
                     train=train, val_loader=val_loader, test_loader=test_loader,
                     partitions=partitions, device=args.device)
        # Save per-round CSV for curve plotting
        fa_history_path = out_dir / f"history_fedavg_mu0.0_E{args.local_epochs}_s{args.seed}.csv"
        fa["history_df"].to_csv(fa_history_path, index=False)
        print(f"  wrote per-round history: {fa_history_path}")
        # Strip Flower internal history before storing in JSON (large)
        fa = {k: v for k, v in fa.items() if k not in ("history", "history_df")}
        print(f"  elapsed: {fa['elapsed_s']/60:.1f} min   test macro-F1 = {fa['test_macro_f1']:.4f}")
        print()

    print(f"→ Running FedProx (μ={args.mu}) via Flower ...", flush=True)
    fp = run_one(algorithm="fedprox", mu=args.mu, seed=args.seed,
                 num_rounds=args.num_rounds, local_epochs=args.local_epochs,
                 lr=args.lr, batch_size=args.batch_size,
                 train=train, val_loader=val_loader, test_loader=test_loader,
                 partitions=partitions, device=args.device)
    fp_history_path = out_dir / f"history_fedprox_mu{args.mu}_E{args.local_epochs}_s{args.seed}.csv"
    fp["history_df"].to_csv(fp_history_path, index=False)
    print(f"  wrote per-round history: {fp_history_path}")
    fp = {k: v for k, v in fp.items() if k not in ("history", "history_df")}
    print(f"  elapsed: {fp['elapsed_s']/60:.1f} min   test macro-F1 = {fp['test_macro_f1']:.4f}")
    print()

    delta_f1 = fp["test_macro_f1"] - fa["test_macro_f1"]
    delta_b  = fp["test_balanced_accuracy"] - fa["test_balanced_accuracy"]
    print("=" * 72)
    print("Head-to-head (paired by seed, Flower simulation)")
    print("=" * 72)
    print(f"  test macro-F1     : FedAvg {fa['test_macro_f1']:.4f} vs FedProx {fp['test_macro_f1']:.4f}   Δ = {delta_f1:+.4f}")
    print(f"  test balanced acc : FedAvg {fa['test_balanced_accuracy']:.4f} vs FedProx {fp['test_balanced_accuracy']:.4f}   Δ = {delta_b:+.4f}")
    print(f"  test accuracy     : FedAvg {fa['test_accuracy']:.4f} vs FedProx {fp['test_accuracy']:.4f}")
    print()
    print("  per-class test F1 (FedAvg → FedProx):")
    for c, name in enumerate(CLASS_NAMES):
        d = fp["test_per_class_f1"][c] - fa["test_per_class_f1"][c]
        print(f"    {name:>10s}: {fa['test_per_class_f1'][c]:.3f} → {fp['test_per_class_f1'][c]:.3f}  Δ = {d:+.3f}")
    print()
    if delta_f1 > 0.005:
        print(f"Verdict (single seed — descriptive): FedProx beats FedAvg on macro-F1 by {delta_f1:+.4f}.")
    elif delta_f1 < -0.005:
        print(f"Verdict (single seed — descriptive): FedAvg beats FedProx on macro-F1 by {-delta_f1:+.4f}. μ={args.mu} may be sub-optimal.")
    else:
        print(f"Verdict (single seed — descriptive): difference within noise (|Δ| ≤ 0.005).")
    print()

    # ---- write comparison JSON ----
    comparison = {
        "framework": "flower-simulation",
        "device": "cpu",
        "seed": args.seed, "num_rounds": args.num_rounds, "local_epochs": args.local_epochs,
        "mu": args.mu, "lr": args.lr, "batch_size": args.batch_size,
        "fedavg_reused": args.skip_fedavg,
        "fedavg":  fa,
        "fedprox": fp,
        "delta_macro_f1": delta_f1,
        "delta_balanced_accuracy": delta_b,
    }
    out = out_dir / f"flower_comparison_seed{args.seed}_mu{args.mu}.json"
    with open(out, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
