"""Flower-framework entry point — paired-fair FedAvg / FedProx on DermaMNIST.

Mirrors the CLI of `run_one.py` exactly, but routes through Flower's
`flwr.simulation.start_simulation` runtime. The produced results are
equivalent (within floating-point noise) to those from the pure-PyTorch
runner — both implement the same FedAvg and FedProx mathematics; they
differ only in the orchestration framework around the per-round
broadcast / local-train / aggregate cycle.

Usage:
    PYTHONPATH=. python -m mnist_dermnist.experiments.run_one_flower \\
        --algorithm fedprox --mu 0.01 --seed 42 \\
        --local-epochs 20 --num-rounds 150 \\
        --partition balanced_paired_7_clients \\
        --device cuda \\
        --out-dir mnist_dermnist/results/headline_flower

When `--system-het-mode` is set to a non-uniform value, the server's
configure_fit() function reads from the per-(round, client) schedule
generated in `mnist_dermnist.fl.system_het`, identical to the pure-PyTorch
path.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
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
from mnist_dermnist.fl.evaluation import evaluate
from mnist_dermnist.fl.system_het import SystemHetConfig, build_epoch_schedule
from mnist_dermnist.fl_flower.client import FlClient, state_dict_to_numpy, numpy_to_state_dict
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
    ap = argparse.ArgumentParser(
        description="Flower-framework FedAvg/FedProx run (paired-fair).")
    ap.add_argument("--algorithm", choices=["fedavg", "fedprox"], required=True)
    ap.add_argument("--mu", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--local-epochs", type=int, default=20)
    ap.add_argument("--num-rounds", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--image-size", type=int, default=28)
    ap.add_argument("--partition", choices=list(PARTITIONERS),
                    default="balanced_paired_7_clients")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--npz-path",
                    default="/Users/basiakoch/cleanest_federated/dermamnist_64.npz")
    ap.add_argument("--out-dir", default="mnist_dermnist/results/headline_flower")
    ap.add_argument("--fraction-fit", type=float, default=1.0,
                    help="Fraction of clients sampled per round (C). "
                         "Default 1.0 = full participation (cross-silo). "
                         "Values < 1 enable partial-participation sensitivity.")
    ap.add_argument("--system-het-mode",
                    choices=["uniform", "fixed_stragglers", "random_stragglers"],
                    default="uniform")
    ap.add_argument("--straggler-epochs", type=int, default=5)
    ap.add_argument("--fixed-straggler-ids", default=None)
    ap.add_argument("--straggler-fraction", type=float, default=0.5)
    # Loss-side imbalance baselines (audit HV2). 'ce' = headline. Set to
    # 'class_weighted_ce' or 'focal' for the imbalance-aware comparators.
    ap.add_argument("--loss-type",
                    choices=["ce", "class_weighted_ce", "focal"],
                    default="ce")
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    return ap


def main():
    args = build_parser().parse_args()
    mu = 0.0 if args.algorithm == "fedavg" else float(args.mu)
    seed = int(args.seed)

    # --- Reproducibility ---
    # NOTE: Python's `random` module is seeded explicitly here because
    # Flower's ClientManager.sample() internally calls random.sample(...)
    # for partial-participation client selection. Without this seed, two
    # runs with --fraction-fit < 1.0 (and otherwise identical
    # configurations) will select different client subsets per round.
    # The pure-PyTorch reference loop in fl/server_loop.py uses a
    # separate np.random.default_rng(seed + 9_000_001) for client
    # sampling; the two runtimes therefore sample DIFFERENT subsets even
    # with the same seed under partial participation. This is acceptable
    # for the present thesis (only C=1.0 results are claimed in the
    # headline; the partial-participation sensitivity is descriptive)
    # but must be acknowledged in any comparison.
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device_str = args.device
    device = torch.device(device_str)

    # --- Data + partition ---
    train, val, test = load_dermmnist(args.npz_path, image_size=args.image_size)
    val_loader = DataLoader(val, batch_size=128, shuffle=False, num_workers=0)
    test_loader = DataLoader(test, batch_size=128, shuffle=False, num_workers=0)

    partitioner = PARTITIONERS[args.partition]
    client_indices, _ = partitioner(train.labels, seed=seed)
    num_clients = len(client_indices)

    # --- System-heterogeneity schedule ---
    fixed_ids = None
    if args.fixed_straggler_ids:
        fixed_ids = [int(x) for x in args.fixed_straggler_ids.split(",")]
    sh_cfg = SystemHetConfig(
        mode=args.system_het_mode,
        E_max=args.local_epochs,
        E_straggler=args.straggler_epochs,
        fixed_straggler_ids=fixed_ids,
        random_straggler_fraction=args.straggler_fraction,
    )
    epoch_schedule = build_epoch_schedule(
        sh_cfg, num_clients=num_clients,
        num_rounds=args.num_rounds, seed=seed,
    )

    # --- Global model: initialise ONCE, deterministically, before clients ---
    torch.manual_seed(seed)  # ensure init is paired
    global_model = DermMNISTCNN(num_classes=7, dropout=0.2).to(device)
    initial_params = state_dict_to_numpy(global_model)

    # --- Tracking state for best-val checkpoint ---
    best_val_macro_f1 = {"value": -1.0, "round": -1,
                         "params": [arr.copy() for arr in initial_params]}
    history_rows: List[Dict] = []
    # Populated by fit_metrics_aggregation_fn (below), keyed by 1-based round.
    # Read out post-simulation to back-fill the train_loss column in
    # history_rows so the Flower CSV matches the pure-PyTorch CSV schema.
    train_loss_by_round: Dict[int, float] = {}

    # --- Centralised evaluation function called every round by the server ---
    def evaluate_fn(server_round: int, parameters: List[np.ndarray], config):
        # Reconstruct model and evaluate on val
        eval_model = DermMNISTCNN(num_classes=7, dropout=0.2).to(device)
        numpy_to_state_dict(eval_model, parameters)
        metrics = evaluate(eval_model, val_loader, device, num_classes=7)

        # Flower invokes evaluate_fn once at server_round=0 with the
        # untrained initial parameters AND after every fit round 1..R.
        # The pure-PyTorch reference loop (fl/server_loop.py) only
        # evaluates after rounds 1..R, so we skip writing the round-0
        # row here to keep history CSVs cross-runtime comparable.
        # The best-val tracker is also skipped for round 0; untrained
        # params would not beat any trained checkpoint in practice.
        if int(server_round) >= 1:
            row = {
                "seed": seed,
                "algorithm": args.algorithm,
                "mu": mu,
                "local_epochs": args.local_epochs,
                "round": server_round,
                "val_loss": metrics["loss"],
                "val_accuracy": metrics["accuracy"],
                "val_balanced_accuracy": metrics["balanced_accuracy"],
                "val_macro_f1": metrics["macro_f1"],
            }
            for c, f1c in enumerate(metrics["per_class_f1"]):
                row[f"val_f1_class_{c}"] = float(f1c)
            history_rows.append(row)

            if metrics["macro_f1"] > best_val_macro_f1["value"]:
                best_val_macro_f1["value"] = float(metrics["macro_f1"])
                best_val_macro_f1["round"] = int(server_round)
                best_val_macro_f1["params"] = [arr.copy() for arr in parameters]

        return metrics["loss"], {
            "val_macro_f1": metrics["macro_f1"],
            "val_balanced_accuracy": metrics["balanced_accuracy"],
            "val_accuracy": metrics["accuracy"],
        }

    # --- Per-round configure_fit: pass per-client local_epochs ---
    def on_fit_config_fn(server_round: int) -> Dict:
        # Note: server_round is 1-based in Flower
        return {"round": server_round}

    # --- Aggregate per-client train_loss so it lands in the history CSV.
    # FedAvg's default strategy DROPS client fit-metrics unless we provide
    # an aggregator. Without this, the Flower history would lack train_loss
    # entirely (the pure-PyTorch CSV has it; the cross-runtime mismatch was
    # the original B3 from the audit).
    def fit_metrics_aggregation_fn(metrics):
        total_n = sum(int(n) for n, _ in metrics)
        if total_n <= 0:
            return {}
        return {
            "train_loss": float(
                sum(float(m.get("train_loss", 0.0)) * int(n) for n, m in metrics)
                / total_n
            ),
        }

    # --- Strategy: standard FedAvg aggregation (FedProx uses same aggregation;
    #     the proximal term is applied client-side in fit()) ---
    n_fit = max(1, int(round(args.fraction_fit * num_clients)))
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=float(args.fraction_fit),
        fraction_evaluate=0.0,  # we don't run per-client eval
        min_fit_clients=n_fit,
        min_evaluate_clients=0,
        min_available_clients=num_clients,
        initial_parameters=fl.common.ndarrays_to_parameters(initial_params),
        evaluate_fn=evaluate_fn,
        on_fit_config_fn=on_fit_config_fn,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        accept_failures=False,
    )

    # --- Client factory passes per-(round, cid) local-epoch override
    #     through a custom config dict per-client. Flower's NumPyClient API
    #     reads the config in fit(). We inject the schedule by composing
    #     a per-client-cid wrapper around on_fit_config_fn. The simplest
    #     approach is for each client to read its own row from the schedule
    #     at fit-time, given the round number from config["round"].
    # Flower 1.29 prefers a Context-based client_fn; we accept both shapes
    # so this runner is portable across 1.7–1.29.
    # The full per-(round, client) epoch schedule is captured in the closure
    # and passed to each client at construction; no fragile config-string
    # round-tripping required.
    # Build the local-training loss once and share across all clients
    # (the inverse-frequency weights depend only on the global training
    # labels, which all clients agree on by design).
    def build_criterion():
        if args.loss_type == "ce":
            return torch.nn.CrossEntropyLoss()
        from mnist_dermnist.fl.class_imbalance import (
            make_class_weighted_ce, make_focal_loss,
        )
        if args.loss_type == "class_weighted_ce":
            return make_class_weighted_ce(train.labels, num_classes=7,
                                          device=device_str)
        return make_focal_loss(gamma=args.focal_gamma, labels=train.labels,
                               num_classes=7, device=device_str)

    def client_fn(context_or_cid) -> fl.client.Client:
        if hasattr(context_or_cid, "node_config"):
            cid_int = int(context_or_cid.node_config.get("partition-id", 0))
        elif hasattr(context_or_cid, "cid"):
            cid_int = int(context_or_cid.cid)
        else:
            cid_int = int(context_or_cid)
        return FlClient(
            cid=cid_int,
            train_dataset=train,
            indices=client_indices[cid_int],
            model_builder=lambda: DermMNISTCNN(num_classes=7, dropout=0.2),
            seed=seed,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            proximal_mu=mu,
            device=device_str,
            epoch_schedule=epoch_schedule,
            criterion=build_criterion(),
        ).to_client()

    # --- Launch Flower simulation ---
    print(f"\n=== Flower runtime: {args.algorithm.upper()} (μ={mu}) seed={seed} ===")
    print(f"  partition={args.partition}  rounds={args.num_rounds}  E_max={args.local_epochs}  C={args.fraction_fit}")
    print(f"  client sizes: {[len(c) for c in client_indices]}")
    print(f"  system_het: mode={args.system_het_mode}")
    print(f"  device={device_str}\n")

    client_resources = ({"num_cpus": 1, "num_gpus": 1.0 / num_clients}
                        if device_str == "cuda" else {"num_cpus": 1, "num_gpus": 0.0})

    t0 = time.time()
    history_obj = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        client_resources=client_resources,
    )
    elapsed = time.time() - t0

    # --- Back-fill train_loss into history_rows from the aggregated fit metrics.
    # Flower stores aggregated fit metrics in `metrics_distributed_fit`:
    # a dict {metric_name -> List[Tuple[round, value]]}.
    if hasattr(history_obj, "metrics_distributed_fit"):
        for r, v in history_obj.metrics_distributed_fit.get("train_loss", []):
            train_loss_by_round[int(r)] = float(v)
    for row in history_rows:
        row["train_loss"] = train_loss_by_round.get(int(row["round"]), float("nan"))

    # --- Final test at best-val checkpoint ---
    # `return_predictions=True` keeps per-sample argmax + target arrays
    # for downstream confusion-matrix analysis (audit P2 fix). They're
    # serialised to a sibling .npz, NOT inlined into the JSON.
    test_model = DermMNISTCNN(num_classes=7, dropout=0.2).to(device)
    numpy_to_state_dict(test_model, best_val_macro_f1["params"])
    test_metrics = evaluate(
        test_model, test_loader, device, num_classes=7,
        return_predictions=True,
    )
    test_metrics["selected_round"] = best_val_macro_f1["round"]
    test_metrics["best_val_macro_f1"] = best_val_macro_f1["value"]
    # Extract predictions before constructing the JSON payload.
    _preds = test_metrics.pop("predictions", None)
    _targets = test_metrics.pop("targets", None)

    # --- Write outputs (mirror save_run_outputs from pure-PyTorch path) ---
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sh_tag = "" if args.system_het_mode == "uniform" else f"_sh-{args.system_het_mode}"
    # Encode fraction_fit only when it differs from full participation, so files
    # produced under C=1.0 retain their existing names and don't break the
    # already-collected headline data.
    c_tag = "" if abs(args.fraction_fit - 1.0) < 1e-9 else f"_C{args.fraction_fit}"
    stem = f"{args.algorithm}_mu{mu}_E{args.local_epochs}{sh_tag}{c_tag}_s{seed}"

    import pandas as pd
    pd.DataFrame(history_rows).to_csv(out_dir / f"history_{stem}.csv", index=False)
    predictions_file = None
    if _preds is not None and _targets is not None:
        predictions_file = f"test_predictions_{stem}.npz"
        np.savez_compressed(
            out_dir / predictions_file,
            predictions=np.asarray(_preds, dtype=np.int64),
            targets=np.asarray(_targets, dtype=np.int64),
            selected_round=int(test_metrics["selected_round"]),
            seed=int(seed),
            algorithm=args.algorithm,
        )
    with open(out_dir / f"test_at_best_{stem}.json", "w") as f:
        json.dump({
            **test_metrics,
            "predictions_file": predictions_file,
            "seed": seed, "algorithm": args.algorithm, "mu": mu,
            "local_epochs": args.local_epochs, "num_rounds": args.num_rounds,
            "lr": args.lr, "momentum": args.momentum, "weight_decay": args.weight_decay,
            "batch_size": args.batch_size, "device": device_str,
            "fraction_fit": float(args.fraction_fit),
            # Provenance fields (mixing-risk mitigation per audit):
            "partition": args.partition,
            "image_size": int(args.image_size),
            "npz_path": str(args.npz_path),
            "framework": "flower-simulation",
            "framework_version": fl.__version__,
            "runner_script": "run_one_flower.py",
            "loss_type": args.loss_type,
            "focal_gamma": args.focal_gamma if args.loss_type == "focal" else None,
            "system_het": sh_cfg.to_dict(),
            "elapsed_s": elapsed,
        }, f, indent=2)

    print(f"\nTest @ best-val (round {test_metrics['selected_round']}, val_macro_f1={test_metrics['best_val_macro_f1']:.4f}):")
    print(f"  test_macro_f1   = {test_metrics['macro_f1']:.4f}")
    print(f"  test_balanced_a = {test_metrics['balanced_accuracy']:.4f}")
    print(f"  test_accuracy   = {test_metrics['accuracy']:.4f}")
    print(f"  elapsed: {elapsed:.1f}s")
    print(f"\nWrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
