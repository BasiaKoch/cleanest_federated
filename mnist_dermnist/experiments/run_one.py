"""CLI: one paired-fair FL run.

Reuses the partition (medical_skew_7_clients by default), model
(DermMNISTCNN), and FL loop already in this package. Writes:
  - history_<algo>_mu<mu>_E<E>_s<seed>.csv
  - test_at_best_<algo>_mu<mu>_E<E>_s<seed>.json

Examples
--------
# FedAvg, seed=42, E=20
python -m mnist_dermnist.experiments.run_one --algorithm fedavg  --seed 42 --local-epochs 20

# FedProx, μ=0.1, seed=42, E=20
python -m mnist_dermnist.experiments.run_one --algorithm fedprox --mu 0.1 --seed 42 --local-epochs 20
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mnist_dermnist.data.load import load_dermmnist
from mnist_dermnist.data.partition import (
    balanced_paired_7_clients,
    balanced_specialist_7_clients,
    dirichlet_7_clients,
    iid_7_clients,
    medical_skew_7_clients,
    quantity_skew_improved,
    simple_pathological_3_clients,
)
from mnist_dermnist.fl.server_loop import FLConfig, run_fl, save_run_outputs
from mnist_dermnist.models import DermMNISTCNN


def _dir_a01(labels, seed=42):
    return dirichlet_7_clients(labels, seed=seed, alpha=0.1)


def _dir_a05(labels, seed=42):
    return dirichlet_7_clients(labels, seed=seed, alpha=0.5)


PARTITIONERS = {
    "medical_skew_7_clients": medical_skew_7_clients,
    "simple_pathological_3_clients": simple_pathological_3_clients,
    "balanced_specialist_7_clients": balanced_specialist_7_clients,
    "balanced_paired_7_clients": balanced_paired_7_clients,
    "quantity_skew_improved": quantity_skew_improved,
    "iid_7_clients": iid_7_clients,
    "dirichlet_alpha01_7_clients": _dir_a01,
    "dirichlet_alpha05_7_clients": _dir_a05,
}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="One paired-fair FedAvg/FedProx run.")
    ap.add_argument("--algorithm", choices=["fedavg", "fedprox"], required=True)
    ap.add_argument("--mu", type=float, default=0.0,
                    help="Proximal coefficient; ignored / forced to 0 for fedavg")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--local-epochs", type=int, default=20)
    ap.add_argument("--num-rounds", type=int, default=150)
    ap.add_argument("--fraction-fit", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=32,
                    help="Capped per-client to min(--batch-size, client_size)")
    ap.add_argument("--image-size", type=int, default=28)
    ap.add_argument("--partition", choices=list(PARTITIONERS), default="medical_skew_7_clients")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--npz-path", default="/Users/basiakoch/cleanest_federated/dermamnist_64.npz")
    ap.add_argument("--out-dir", default="mnist_dermnist/results/headline")
    # --- System heterogeneity (per-client variable local epochs) ---
    # See mnist_dermnist/fl/system_het.py for the canonical reference and the
    # connection to Li et al. (2020) §5.2. mode='uniform' is the default and
    # bit-identical to the pre-system-het code path.
    ap.add_argument("--system-het-mode",
                    choices=["uniform", "fixed_stragglers", "random_stragglers"],
                    default="uniform",
                    help="System-heterogeneity scenario (default: uniform = no system het)")
    ap.add_argument("--straggler-epochs", type=int, default=5,
                    help="E for stragglers in 'fixed_stragglers' mode")
    ap.add_argument("--fixed-straggler-ids", default=None,
                    help="Comma-separated client ids for fixed stragglers (default: last 2)")
    ap.add_argument("--straggler-fraction", type=float, default=0.5,
                    help="Fraction of clients per round designated stragglers in random mode")
    return ap


def main():
    args = build_parser().parse_args()

    # FedAvg: force μ=0 regardless of --mu
    mu = 0.0 if args.algorithm == "fedavg" else float(args.mu)

    # Data
    train, val, test = load_dermmnist(args.npz_path, image_size=args.image_size)
    val_loader = DataLoader(val, batch_size=128, shuffle=False, num_workers=0)
    test_loader = DataLoader(test, batch_size=128, shuffle=False, num_workers=0)

    # Partition (seeded → identical across paired FedAvg / FedProx runs)
    partitioner = PARTITIONERS[args.partition]
    client_indices, _ = partitioner(train.labels, seed=args.seed)

    # Build the system-het config (default 'uniform' = no system het)
    from mnist_dermnist.fl.system_het import SystemHetConfig
    fixed_ids = None
    if args.fixed_straggler_ids:
        fixed_ids = [int(x) for x in args.fixed_straggler_ids.split(",")]
    system_het_cfg = SystemHetConfig(
        mode=args.system_het_mode,
        E_max=args.local_epochs,
        E_straggler=args.straggler_epochs,
        fixed_straggler_ids=fixed_ids,
        random_straggler_fraction=args.straggler_fraction,
    )

    cfg = FLConfig(
        seed=args.seed,
        algorithm=args.algorithm,
        mu=mu,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        fraction_fit=args.fraction_fit,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        num_classes=7,
        device=args.device,
        system_het=system_het_cfg,
    )

    print(f"\n=== {args.algorithm.upper()} (μ={mu}) seed={args.seed} E={args.local_epochs} ===")
    print(f"  partition={args.partition}  rounds={args.num_rounds}  fraction_fit={args.fraction_fit}")
    print(f"  client sizes: {[len(c) for c in client_indices]}")
    print(f"  system_het: mode={args.system_het_mode}", end="")
    if args.system_het_mode == "fixed_stragglers":
        print(f"  stragglers={fixed_ids if fixed_ids else 'last_2'}  E_straggler={args.straggler_epochs}")
    elif args.system_het_mode == "random_stragglers":
        print(f"  frac={args.straggler_fraction}  E_i~Uniform[1, {args.local_epochs - 1}]")
    else:
        print()
    print(f"  device={args.device}\n")

    result = run_fl(
        cfg,
        model_builder=DermMNISTCNN,
        train_dataset=train,
        val_loader=val_loader,
        test_loader=test_loader,
        client_indices=client_indices,
    )

    out_dir = Path(args.out_dir)
    # Pass provenance metadata that isn't carried inside FLConfig so each
    # output JSON is fully self-documenting (mixing risk under shared
    # out-dirs flagged in audit).
    extra_metadata = {
        "partition": args.partition,
        "image_size": int(args.image_size),
        "npz_path": str(args.npz_path),
        "framework": "pure-pytorch-reference-loop",
        "framework_version": None,
        "loss_type": "cross_entropy",
        "runner_script": "run_one.py",
    }
    save_run_outputs(result, out_dir, extra_metadata=extra_metadata)
    tm = result["test_metrics"]
    if tm is not None:
        print(f"Test @ best-val checkpoint (round {tm['selected_round']}, val_macro_f1={tm['best_val_macro_f1']:.4f}):")
        print(f"  test_macro_f1   = {tm['macro_f1']:.4f}")
        print(f"  test_balanced_a = {tm['balanced_accuracy']:.4f}")
        print(f"  test_accuracy   = {tm['accuracy']:.4f}")
        print(f"  test_loss       = {tm['loss']:.4f}")
    print(f"\nWrote outputs to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
